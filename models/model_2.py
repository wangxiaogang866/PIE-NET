import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../tf_ops/nn_distance'))
import tensorflow as tf
import numpy as np
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_fp_module
import tf_nndistance
import math


def smooth_l1_dist(deltas, sigma2=2.0, name='smooth_l1_dist'):
    with tf.name_scope(name=name) as scope:
        deltas_abs = tf.abs(deltas)
        smoothL1_sign = tf.cast(tf.less(deltas_abs, 1.0/sigma2), tf.float32)
        return tf.square(deltas) * 0.5 * sigma2 * smoothL1_sign + \
                   (deltas_abs - 0.5 / sigma2) * tf.abs(smoothL1_sign - 1)

def get_feature(point_cloud, is_training,stage,bn_decay=None):
    """ Part segmentation PointNet, input is BxNx6 (XYZ NormalX NormalY NormalZ), output Bx50 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = tf.slice(point_cloud, [0,0,0], [-1,-1,3])
    l0_points = tf.slice(point_cloud, [0,0,3], [-1,-1,0])

    # Layer 1
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=4096, radius=0.05, nsample=32, mlp=[32,32,64], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=2048, radius=0.1, nsample=32, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=1024, radius=0.2, nsample=32, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer3')
    l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=512, radius=0.4, nsample=32, mlp=[256,256,512], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer4')

    # with FPN
    #if return_fullfea:
    #    end_points['sem_fea_full_l4'] = pointnet_fp_module(xyz, l4_xyz, points, l4_points, [], is_training, bn_decay, scope='fa_layer1_fpn')
    #    end_points['sem_fea_full_l3'] = pointnet_fp_module(xyz, l3_xyz, points, l3_points, [], is_training, bn_decay, scope='fa_layer2_fpn')
    #    end_points['sem_fea_full_l2'] = pointnet_fp_module(xyz, l2_xyz, points, l2_points, [], is_training, bn_decay, scope='fa_layer3_fpn')
    #    end_points['sem_fea_full_l1'] = pointnet_fp_module(xyz, l1_xyz, points, l1_points, [], is_training, bn_decay, scope='fa_layer4_fpn')
    
    
    # Feature Propagation layers
    l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [256,256], is_training, bn_decay, scope='fa_layer1')
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256,256], is_training, bn_decay, scope='fa_layer2')
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,128], is_training, bn_decay, scope='fa_layer3')
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128,128,128], is_training, bn_decay, scope='fa_layer4')

    # FC layers
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='pointnet/fc1', bn_decay=bn_decay)

    end_points['feats'] = net 
    if stage==1:
    	net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='pointnet/dp1')
    
    # dof_feature
    dof_feat = tf_util.conv1d(net, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='pointnet/fc_dof', bn_decay=bn_decay)
    # simmat_feature
    simmat_feat = tf_util.conv1d(net, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='pointnet/fc_simmat', bn_decay=bn_decay)
    return end_points,dof_feat,simmat_feat

def placeholder_inputs_stage_1(batch_size,num_point,num_open_pair,num_point_per_open_pair,num_open_gt_sample, num_closed_point,num_point_per_closed_point,num_closed_gt_sample):
    pointclouds_pl = tf.placeholder(tf.float32,shape=(batch_size,num_point,3))  # input
    labels_key_p = tf.placeholder(tf.int32,shape=(batch_size,num_point))  # edge points label 0/1
    labels_corner_p = tf.placeholder(tf.int32,shape=(batch_size,num_point))
        
    open_gt_256_64_idx = tf.placeholder(tf.int32,shape=(batch_size, num_open_pair, num_point_per_open_pair))
    open_gt_mask = tf.placeholder(tf.int32,shape=(batch_size, num_open_pair, num_point_per_open_pair))
    open_gt_type = tf.placeholder(tf.int32,shape=(batch_size, num_open_pair))
    open_gt_res = tf.placeholder(tf.float32,shape=(batch_size, num_open_pair, 3*2))
    open_gt_sample_points = tf.placeholder(tf.float32,shape=(batch_size, num_open_pair, num_open_gt_sample, 3))
    open_gt_valid_mask = tf.placeholder(tf.int32,shape=(batch_size, num_open_pair))
    open_gt_pair_idx = tf.placeholder(tf.int32,shape=(batch_size, num_open_pair, 2))
    
    closed_gt_256_64_idx = tf.placeholder(tf.int32,shape=(batch_size, num_closed_point, num_point_per_closed_point))
    closed_gt_mask =  tf.placeholder(tf.int32,shape=(batch_size, num_closed_point, num_point_per_closed_point))
    closed_gt_type =  tf.placeholder(tf.int32,shape=(batch_size, num_closed_point))
    closed_gt_res =  tf.placeholder(tf.float32,shape=(batch_size, num_closed_point, 3))
    closed_gt_sample_points =  tf.placeholder(tf.float32,shape=(batch_size, num_closed_point, num_closed_gt_sample, 3))
    closed_gt_valid_mask = tf.placeholder(tf.int32,shape=(batch_size, num_closed_point))
    closed_gt_pair_idx = tf.placeholder(tf.int32,shape=(batch_size, num_closed_point,1))
    return pointclouds_pl,labels_key_p,labels_corner_p, \
            open_gt_256_64_idx, open_gt_mask, open_gt_type, open_gt_res, open_gt_sample_points, open_gt_valid_mask, open_gt_pair_idx, \
            closed_gt_256_64_idx, closed_gt_mask, closed_gt_type, closed_gt_res, closed_gt_sample_points, closed_gt_valid_mask, closed_gt_pair_idx
            

##############################################################################################################################
##############################################################################################################################
def Line_predict(para_line, para_start_end, num_point_pre):
    ''' 
    Segmentation head
    Inputs:
        para_line: [B, NUM_ROIS, NUM_CONTROL_PARAMETERS(2*3)]   # B * 512 * (start_point_delta, end_point_delta)
        para_start_end: [B, NUM_ROIS, 3+3]   # B * 512 * 3+3;
		num_point_pre: int32; # 64
    Returns:
        points_pre: [B, NUM_ROIS, num_point_pre, 3]   # B*512*64*3	
    '''
    Norm_start_points = tf.slice(para_start_end,[0,0,0],[-1,-1,3])
    Norm_end_points = tf.slice(para_start_end, [0,0,3], [-1,-1,-1])
    	
    start_points_delta = tf.slice(para_line, [0,0,0], [-1,-1,3])
    end_points_delta = tf.slice(para_line, [0,0,3], [-1,-1,3])
    
    Norm_start_points = Norm_start_points + start_points_delta
    Norm_end_points = Norm_end_points + end_points_delta
    	
    Batch_num = para_line.get_shape()[0].value
    Roi_num = para_line.get_shape()[1].value
    one_d_tensor = tf.range(0,num_point_pre) #tf.constant(0)
    theta_mat = tf.tile(tf.expand_dims(tf.expand_dims(one_d_tensor,axis=0),axis =0),[Batch_num,Roi_num,1]) 
    theta_mat = tf.cast(theta_mat,tf.float32)
    
    intern = 1/tf.cast((num_point_pre-1),tf.float32)
    theta_mat = intern*theta_mat  #[Batch_num,Roi_num,num_point_pre]
    	
    theta_mat_t = tf.tile(tf.expand_dims(theta_mat,3),[1,1,1,3]) # [B, num_roi, num_pre_points, num_x_y_z(3)]
    #print(sess.run(theta_mat_t))
    
    theta_mat_1_t = 1-theta_mat_t
    #print(sess.run(theta_mat_1_t))
    	
    Norm_start_points_t = tf.tile(tf.expand_dims(Norm_start_points,2),[1,1,num_point_pre,1]) 
    Norm_end_points_t = tf.tile(tf.expand_dims(Norm_end_points,2),[1,1,num_point_pre,1]) 
    	
    points_pre = tf.multiply(theta_mat_t,Norm_start_points_t) + tf.multiply(theta_mat_1_t,Norm_end_points_t)
    #print('points_pre:')
    #print(sess.run(points_pre))
    return points_pre
	

##############################################################################################################################
##############################################################################################################################
def B_spline_predict(para_spline, para_start_end,num_point_pre):
    ''' 
    b_spline
    Inputs:
        para_spline: [B, NUM_ROIS, NUM_CONTROL_PARAMETERS(4*3)]   # B * 512 * (start_point_delta, end_point_delta, control_point1, control_point2)
        para_start_end: [B, NUM_ROIS, 3+3]   # B * 512 * 3+3;
		num_point_pre: int32; # 64
    Returns:
        points_pre: [B, NUM_ROIS, num_point_pre, 3]   # B*512*64*3
    '''	
    #para_spline = [[[0,0,0,0,0,0,1,4,1,5,-4,1],[0,0,0,0,0,0,5,4,1,10,-4,1]]]
    para_spline = tf.cast(para_spline,tf.float32)
    #print('para_spline:',sess.run(para_spline))
    #para_start_end = [[[0,0,0,10,0,0],[1,0,0,15,0,0]]]
    para_start_end = tf.cast(para_start_end,tf.float32)
    #print('para_start_end:',sess.run(para_start_end))    
    
    #num_point_pre = tf.cast(num_point_pre,tf.float32)
    Norm_start_points = tf.slice(para_start_end,[0,0,0],[-1,-1,3])
    Norm_end_points = tf.slice(para_start_end, [0,0,3], [-1,-1,-1])
    	
    start_points_delta = tf.slice(para_spline, [0,0,0], [-1,-1,3])
    end_points_delta = tf.slice(para_spline, [0,0,3], [-1,-1,3])
    control_points_1 = tf.slice(para_spline, [0,0,6], [-1,-1,3])
    control_points_2 = tf.slice(para_spline, [0,0,9], [-1,-1,-1])
    
    Norm_start_points = Norm_start_points + start_points_delta
    Norm_end_points = Norm_end_points + end_points_delta
    	
    Batch_num = para_spline.get_shape()[0].value
    Roi_num = para_spline.get_shape()[1].value
    one_d_tensor = tf.range(0,num_point_pre) #tf.constant(0)
    theta_mat = tf.tile(tf.expand_dims(tf.expand_dims(one_d_tensor,axis=0),axis =0),[Batch_num,Roi_num,1]) 
    theta_mat = tf.cast(theta_mat,tf.float32)	
    	
    intern = 1/tf.cast((num_point_pre-1),tf.float32)
    theta_mat = intern*theta_mat  #[Batch_num,Roi_num,num_point_pre]
    
    theta_mat_t_3 =  tf.pow(theta_mat,3)
    theta_mat_t_2 =  tf.pow(theta_mat,2)	
    theta_mat_t_1 =  tf.pow(theta_mat,1)
    theta_mat_t_0 =  tf.pow(theta_mat,0)
    theta_mat_T = tf.stack([theta_mat_t_3,theta_mat_t_2,theta_mat_t_1,theta_mat_t_0],3)   # [B,NUM_ROI,NUM_PRE_POINTS,[T3,T2,T1,T0]]
    #print('theta_mat_T:')
    #print(sess.run(theta_mat_T))
    	
    B_mat_t = tf.constant([[[[-1,3,-3,1],[3,-6,3,0],[-3,3,0,0],[1,0,0,0]]]],tf.float32)
    B_mat_T = tf.tile(B_mat_t,[Batch_num,Roi_num,1,1])  # [B,NUM_ROI, 4,4]
    	
    Key_points = tf.stack([Norm_start_points,control_points_1,control_points_2,Norm_end_points],2) # [B,NUM_ROI,[P1,P2,P3,P4],3(X,Y,Z)] 
    	
    points_pre = tf.matmul(tf.matmul(theta_mat_T,B_mat_T),Key_points)
    #print('points_pre:')
    #print(sess.run(points_pre))
    return points_pre
	

##############################################################################################################################
##############################################################################################################################	
def Cycle_predict(para_cycle, num_point_pre):
    ''' 
    cycle
    Inputs:
        para_cycle: [B, NUM_ROIS, 7]   # B * 512 * 7;   center, normal,radius
        # para_start_end: [B, NUM_ROIS, 3+3]   # B * 512 * 3+3;
		# para_end: [B, NUM_ROIS, 3]   # B * 512 * 3;     
		num_point_pre: int32; # 256
    Returns:
        points_pre: [B, NUM_ROIS, num_point_pre, 3]   # B*512*64*3
    '''   
    para_cycle = tf.cast(para_cycle,tf.float32)
    
    #Norm_start_points = tf.slice(para_cycle,[0,0,4],[-1,-1,3])     # here we need to refine to remove start points
    #Norm_end_points = tf.slice(para_start_end, [0,0,3], [-1,-1,-1])


    cycle_center = tf.slice(para_cycle, [0,0,0], [-1,-1,3])
    cycle_normal = tf.slice(para_cycle, [0,0,3], [-1,-1,3])
    cycle_radius = tf.slice(para_cycle, [0,0,6], [-1,-1,-1])
    Norm_start_points = tf.ones_like(cycle_center)
    
    Batch_num = para_cycle.get_shape()[0].value
    Roi_num = para_cycle.get_shape()[1].value
    #print('Roi_num:',sess.run(Roi_num))
    
    vector_a_norm_temp = Norm_start_points - cycle_center
    vector_a_norm_temp_num = vector_a_norm_temp.get_shape()[0].value
    #print('vector_a_norm_temp_num:',sess.run(vector_a_norm_temp_num))
    
    vector_b = tf.cross(cycle_normal, vector_a_norm_temp)
    #print('vector_b:',sess.run(vector_b))
    #print('vector_b:',sess.run(vector_b.get_shape()[0].value))
    
    vector_b_norm = tf.nn.l2_normalize(vector_b, dim=2)
    #print('vector_b_norm:',sess.run(vector_b_norm))
    
    vector_a = tf.cross(cycle_normal, vector_b_norm)	
    vector_a_norm = tf.nn.l2_normalize(vector_a, dim=2)
    	
    Batch_num = para_cycle.get_shape()[0].value
    Roi_num = para_cycle.get_shape()[1].value
    one_d_tensor = tf.range(0,num_point_pre) #tf.constant(0)
    theta_mat = tf.tile(tf.expand_dims(tf.expand_dims(one_d_tensor,axis=0),axis =0),[Batch_num,Roi_num,1]) 
    theta_mat = tf.cast(theta_mat,tf.float32)
    #print('theta_mat:',sess.run(theta_mat))
    	
    PI = tf.constant(math.pi, tf.float32)
    #print('PI:',sess.run(PI))
    intern = 2*PI/tf.cast((num_point_pre-1),tf.float32)
    theta_mat = intern*theta_mat
    	
    cos_theta_mat = tf.cos(theta_mat)
    sin_theta_mat = tf.sin(theta_mat)
        
        
    c0 = tf.tile(tf.expand_dims(tf.gather(cycle_center,0,axis=2),axis = 2),[1,1,num_point_pre])
    c1 = tf.tile(tf.expand_dims(tf.gather(cycle_center,1,axis=2),axis = 2),[1,1,num_point_pre])
    c2 = tf.tile(tf.expand_dims(tf.gather(cycle_center,2,axis=2),axis = 2),[1,1,num_point_pre])
    
    a0 = tf.tile(tf.expand_dims(tf.gather(vector_a_norm,0,axis=2),axis = 2),[1,1,num_point_pre])
    a1 = tf.tile(tf.expand_dims(tf.gather(vector_a_norm,1,axis=2),axis = 2),[1,1,num_point_pre])
    a2 = tf.tile(tf.expand_dims(tf.gather(vector_a_norm,2,axis=2),axis = 2),[1,1,num_point_pre])	
    	
    b0 = tf.tile(tf.expand_dims(tf.gather(vector_b_norm,0,axis=2),axis = 2),[1,1,num_point_pre])
    b1 = tf.tile(tf.expand_dims(tf.gather(vector_b_norm,1,axis=2),axis = 2),[1,1,num_point_pre])
    b2 = tf.tile(tf.expand_dims(tf.gather(vector_b_norm,2,axis=2),axis = 2),[1,1,num_point_pre])
    
    radius_mat = tf.tile(cycle_radius,[1,1,num_point_pre])
    
    x = c0 + radius_mat*a0*cos_theta_mat + radius_mat*b0*sin_theta_mat
    y = c1 + radius_mat*a1*cos_theta_mat + radius_mat*b1*sin_theta_mat
    z = c2 + radius_mat*a2*cos_theta_mat + radius_mat*b2*sin_theta_mat
	
    points_pre = tf.concat([tf.expand_dims(x,axis=3),tf.expand_dims(y,axis=3),tf.expand_dims(z,axis=3)],axis=3)    
    return points_pre
            
##############################################################################################################################
##############################################################################################################################
# For high-dim idx to extract pc and pc_feat;  idx: [B, NUM_ROIS, NUM_POINT_PER_ROI]  and pc: [B, NUM_POINT, 3], pc_fea: [B, NUM_POINT, NFEA]
def points_cropping(pc, pc_fea, masks_selection_idx, normalize_crop_region=True):
    ''' Crop points for network heads, in analogy to ROIAlign
    Inputs:
        pc: [B, NUM_POINT, 3]
        pc_fea: [B, NUM_POINT, NFEA]
        #pc_center: [B, NUM_POINT, 3]
        #rois: [B, NUM_ROIS, 6], zero padded
		radius: [B, NUM_ROIS], zero padded
		center: [B, NUM_ROIS, 3], zero padded
        masks_selection_idx: [B, NUM_ROIS, NUM_POINT_PER_ROI]
    Returns:
        pc_fea_cropped: [B, NUM_ROIS, NUM_POINT_PER_ROI, NFEA]
        #pc_center_cropped: [B, NUM_ROIS, NUM_POINT_PER_ROI, 3]
        pc_coord_cropped: [B, NUM_ROIS, NUM_POINT_PER_ROI, 3]
    '''
    batch_size = masks_selection_idx.get_shape()[0].value
    num_rois = masks_selection_idx.get_shape()[1].value
    num_point_per_roi = masks_selection_idx.get_shape()[2].value
    smp_idx = tf.reshape(tf.tile(tf.reshape(tf.range(batch_size),[-1,1]),[1, num_rois*num_point_per_roi]),[-1,1])
    smp_idx = tf.concat((smp_idx, tf.reshape(masks_selection_idx,[-1,1])),1)
    pc_fea_cropped = tf.reshape(tf.gather_nd(pc_fea, smp_idx),[batch_size, num_rois, num_point_per_roi, -1])
    pc_coord_cropped_unnormalized = tf.reshape(tf.gather_nd(pc, smp_idx),[batch_size, num_rois, num_point_per_roi, -1])
    pc_coord_cropped = pc_coord_cropped_unnormalized

    # convert world coord to local
    bbox_size = tf.reduce_max(pc_coord_cropped, axis=2, keep_dims=True)-tf.reduce_min(pc_coord_cropped, axis=2, keep_dims=True)
    radius = 1e-8 + tf.sqrt(tf.reduce_sum(tf.square(bbox_size/2), axis=-1, keep_dims=True)) # [B, nsmp, 1, 1]	
    center = tf.reduce_min(pc_coord_cropped, axis=2, keep_dims=True) + bbox_size/2   # [B, nsmp, 1, 3]
	
    # normalization
    rois_center = center
    pc_coord_cropped = pc_coord_cropped-rois_center
    if normalize_crop_region:
        # scale box to [1,1,1]
        pc_coord_cropped = tf.divide(pc_coord_cropped, radius)
    return pc_fea_cropped, pc_coord_cropped, pc_coord_cropped_unnormalized, radius, center            

######################################################################################
######################################################################################
# For low-dim idx to extract pc and pc_feat; idx:[B, num_closed]
def points_cropping_low(pc, pc_fea, masks_selection_idx):
    ''' Crop points for network heads, in analogy to ROIAlign
    Inputs:
        pc: [B, NUM_POINT, 3]
        pc_fea: [B, NUM_POINT, NFEA]
        masks_selection_idx: [B, num_closed]
    Returns:
        pc_fea_cropped: [B, num_closed, NFEA]
        pc_coord_cropped: [B, num_closed, 3]  B*512*3
		pc_coord_corpped_knn_idx:  [B, num_closed, num_sample_knn]
    '''
    batch_size = pc.get_shape()[0].value
    num_closed = masks_selection_idx.get_shape()[1].value
    smp_idx = tf.reshape(tf.tile(tf.reshape(tf.range(batch_size),[-1,1]),[1, num_closed]),[-1,1])
    smp_idx = tf.concat((smp_idx, tf.reshape(masks_selection_idx,[-1,1])),1)
    pc_fea_cropped = tf.reshape(tf.gather_nd(pc_fea, smp_idx),[batch_size, num_closed, -1])
    pc_coord_cropped = tf.reshape(tf.gather_nd(pc, smp_idx),[batch_size, num_closed, -1])
    
    return pc_fea_cropped, pc_coord_cropped

######################################################################################
######################################################################################
def classification_head_open(pc, pc_fea, num_category, num_sample_pre, mlp_list, mlp_list2, is_training, bn_decay, scope, bn=True):
    ''' Classification head for both class id prediction and bbox delta regression
    Inputs:
        pc: [B, NUM_ROIS, NUM_POINT_PER_ROI, 3]
        pc_fea: [B, NUM_ROIS, NUM_POINT_PER_ROI, NFEA]
        num_category: scalar
		num_sample_pre: scalar  # predict points for each curve
    Returns:
        logits: [B, NUM_ROIS, NUM_CATEGORY]
        probs: [B, NUM_ROIS, NUM_CATEGORY]
        bbox_deltas: [B, NUM_ROIS, NUM_CATEGORY, (dz, dy, dx, log(dh), log(dw), log(dl))]
		open_curve_pred: [B, NUM_ROIS, NUM_CATEGORY, num_sample_pre,3]  # 1+3 is zeros/cycle/line/b_spline
    '''
    with tf.variable_scope(scope) as myscope:
        num_rois = pc.get_shape()[1].value
        grouped_points = tf.concat((pc_fea, pc), -1)
        for i,num_out_channel in enumerate(mlp_list):
            grouped_points = tf_util.conv2d(grouped_points, num_out_channel, [1, 1],
                                            padding='VALID', stride=[1,1], bn=bn, is_training=is_training,
                                            scope='conv_prev_%d'%i, bn_decay=bn_decay)
        new_points = tf.reduce_max(grouped_points, axis=2)
        for i,num_out_channel in enumerate(mlp_list2):
            new_points = tf_util.conv1d(new_points, num_out_channel, 1,
                                        padding='VALID', stride=1, bn=bn, is_training=is_training,
                                        scope='conv_post_%d'%i, bn_decay=bn_decay)
        logits = tf_util.conv1d(new_points, num_category, 1, padding='VALID',
                                stride=1, scope='conv_classify', activation_fn=None)
        probs = tf.nn.softmax(logits, 2)
        bbox_deltas = tf_util.conv1d(new_points, num_category*6, 1, padding='VALID',
                                     stride=1, scope='conv_bbox_regress', activation_fn=None)
        bbox_deltas = tf.reshape(bbox_deltas, [-1, num_rois, num_category, 6])
		

        
        # num_sample_points
        Batch_num = pc.get_shape()[0].value
        Roi_num = pc.get_shape()[1].value
        num_sample_points = 64 #tf.cast(tf.constant(64),tf.float32)
        
        # para_start_end extract from pc (frist and second colums)
        para_start_end = tf.gather(pc, [0,1], axis = 2)
        para_start_end = tf.reshape(para_start_end, [Batch_num, Roi_num, -1])  #[batch,roi_num,6]
        
        
        # para_cycle: [B, NUM_ROIS, 7]   # B * 512 * 7;   center, normal,radius  output: [B, num_roi, num_points,3]
        para_cycle = tf_util.conv1d(new_points, 7, 1, padding='VALID',
                                     stride=1, scope='conv_open_cycle', activation_fn=None)
        cycle_curve_pre = Cycle_predict(para_cycle, num_sample_points)	

        # para_spline: [B, NUM_ROIS, NUM_CONTROL_PARAMETERS(4*3)]   # B * 512 * (start_point_delta, end_point_delta, control_point1, control_point2)
        # para_start_end: [B, NUM_ROIS, 3+3]   # B * 512 * 3+3;  
        # output: [B, num_roi, num_points,3]
        para_spline = tf_util.conv1d(new_points, 12, 1, padding='VALID',
                                     stride=1, scope='conv_open_spline', activation_fn=None)        
        b_spline_curve_pre = B_spline_predict(para_spline, para_start_end, num_sample_points)

        #para_line: [B, NUM_ROIS, NUM_CONTROL_PARAMETERS(2*3)]   # B * 512 * (start_point_delta, end_point_delta)
        #para_start_end: [B, NUM_ROIS, 3+3]   # B * 512 * 3+3;
        # output: [B, num_roi, num_points,3] 
        para_line = tf_util.conv1d(new_points, 6, 1, padding='VALID',
                                     stride=1, scope='conv_open_line', activation_fn=None)         		
        line_curve_pre = Line_predict(para_line,para_start_end, num_sample_points)
        
        #generate all zeros matrix for 0 catergory; output: zeros[B, num_roi, num_points,3] 
        # pc: [B, NUM_ROIS, NUM_POINT_PER_ROI, 3]
        category_zero = tf.zeros([Batch_num,Roi_num,num_sample_points,3])
        		
        open_curve_pred = tf.stack([category_zero, cycle_curve_pre, b_spline_curve_pre, line_curve_pre], 2)
		
        return logits, probs, bbox_deltas, open_curve_pred, para_cycle, para_spline, para_line       
        

######################################################################################
######################################################################################
def classification_head_closed(pc, pc_fea, num_category, mlp_list, mlp_list2, is_training, bn_decay, scope, bn=True):
    ''' Classification head for both class id prediction and bbox delta regression
    Inputs:
        pc: [B, NUM_ROIS, NUM_POINT_PER_ROI, 3]
        pc_fea: [B, NUM_ROIS, NUM_POINT_PER_ROI, NFEA]
        num_category: scalar
    Returns:
        logits: [B, NUM_ROIS, NUM_CATEGORY]
        probs: [B, NUM_ROIS, NUM_CATEGORY]
        bbox_deltas: [B, NUM_ROIS, NUM_CATEGORY, (dz, dy, dx, log(dh), log(dw), log(dl))]   #
		closed_curve_pred: [B, NUM_ROIS, num_sample_pre,3]
    '''
    with tf.variable_scope(scope) as myscope:
        num_rois = pc.get_shape()[1].value
        grouped_points = tf.concat((pc_fea, pc), -1)
        for i,num_out_channel in enumerate(mlp_list):
            grouped_points = tf_util.conv2d(grouped_points, num_out_channel, [1, 1],
                                            padding='VALID', stride=[1,1], bn=bn, is_training=is_training,
                                            scope='conv_prev_%d'%i, bn_decay=bn_decay)
        new_points = tf.reduce_max(grouped_points, axis=2)
        for i,num_out_channel in enumerate(mlp_list2):
            new_points = tf_util.conv1d(new_points, num_out_channel, 1,
                                        padding='VALID', stride=1, bn=bn, is_training=is_training,
                                        scope='conv_post_%d'%i, bn_decay=bn_decay)
        logits = tf_util.conv1d(new_points, num_category, 1, padding='VALID',
                                stride=1, scope='conv_classify', activation_fn=None)
        probs = tf.nn.softmax(logits, 2)
        bbox_deltas = tf_util.conv1d(new_points, num_category*6, 1, padding='VALID',
                                     stride=1, scope='conv_bbox_regress', activation_fn=None)
        bbox_deltas = tf.reshape(bbox_deltas, [-1, num_rois, num_category, 6])
		
        # num_sample_points
        Batch_num = pc.get_shape()[0].value
        Roi_num = pc.get_shape()[1].value
        num_sample_points = 64 #tf.cast(tf.constant(64),tf.float32)
        
        # para_start_end extract from pc (frist and second colums)
        para_start_end = tf.gather(pc, [0,1], axis = 2)
        para_start_end = tf.reshape(para_start_end, [Batch_num, Roi_num, -1])  #[batch,roi_num,6]        
        
        # para_cycle: [B, NUM_ROIS, 7]   # B * 512 * 7;   center, normal,radius  output: [B, num_roi, num_points,3]
        para_cycle = tf_util.conv1d(new_points, 7, 1, padding='VALID',
                                     stride=1, scope='conv_closed_cycle', activation_fn=None)        		
        cycle_curve_pre = Cycle_predict(para_cycle, num_sample_points)
        
        #generate all zeros matrix for 0 catergory; output: zeros[B, num_roi, num_points,3] 
        # pc: [B, NUM_ROIS, NUM_POINT_PER_ROI, 3]
        category_zero = tf.zeros([Batch_num,Roi_num,num_sample_points,3])
        		
        closed_curve_pred = tf.stack([category_zero, cycle_curve_pre], 2)        
        
        return logits, probs, bbox_deltas, closed_curve_pred, para_cycle
        

######################################################################################
######################################################################################
def segmentation_head(pc, pc_fea, num_category, mlp_list, mlp_list2, mlp_list3, is_training, bn_decay, scope, bn=True):
    ''' Segmentation head
    Inputs:
        pc: [B, NUM_ROIS, NUM_POINT_PER_ROI, 3]
        pc_fea: [B, NUM_ROIS, NUM_POINT_PER_ROI, NFEA]
        num_category: scalar
    Returns:
        masks: [B, NUM_ROIS, NUM_POINT_PER_ROI, NUM_CATEGORY]
    '''
    with tf.variable_scope(scope) as myscope:
        num_rois = pc.get_shape()[1].value
        num_point_per_roi = pc.get_shape()[2].value
        grouped_points = tf.concat((pc_fea, pc), -1)
        for i,num_out_channel in enumerate(mlp_list):
            grouped_points = tf_util.conv2d(grouped_points, num_out_channel, [1, 1],
                                            padding='VALID', stride=[1,1], bn=bn, is_training=is_training,
                                            scope='conv_prev_%d'%i, bn_decay=bn_decay)
        local_feat = grouped_points
        for i,num_out_channel in enumerate(mlp_list2):
            grouped_points = tf_util.conv2d(grouped_points, num_out_channel, [1, 1],
                                            padding='VALID', stride=[1,1], bn=bn, is_training=is_training,
                                            scope='conv_%d'%i, bn_decay=bn_decay)
        global_feat = tf.reduce_max(grouped_points, axis=2, keep_dims=True)
        global_feat_expanded = tf.tile(global_feat, [1, 1, num_point_per_roi, 1])
        new_points = tf.concat((global_feat_expanded, local_feat), -1)
        for i,num_out_channel in enumerate(mlp_list3):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1, 1],
                                            padding='VALID', stride=[1,1], bn=bn, is_training=is_training,
                                            scope='conv_post_%d'%i, bn_decay=bn_decay)

        masks = tf_util.conv2d(new_points, num_category, [1, 1], padding='VALID',
                               stride=[1,1], scope='conv_seg', activation_fn=None)
        return masks


######################################################################################
######################################################################################
def sem_net(xyz, points, num_category, end_points, scope, is_training, bn_decay=None, return_fullfea=False):
    ''' Encode multiple context.
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            num_category: int32 -- #output category
        Return:
            sem_class_logits: (batch_size, npoint_sem, num_category)
    '''        
    l0_xyz = xyz
    l0_points = points

    # Layer 1
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=512, radius=0.1, nsample=32, mlp=[32,32,64], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='sem_layer1')
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=256, radius=0.2, nsample=32, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='sem_layer2')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=128, radius=0.4, nsample=32, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='sem_layer3')
    #l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=32, radius=0.8, nsample=32, mlp=[256,256,512], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='sem_layer4')

    # with FPN
    if return_fullfea:
        #end_points['sem_fea_full_l4'] = pointnet_fp_module(xyz, l4_xyz, points, l4_points, [], is_training, bn_decay, scope='sem_fa_layer1_fpn')
        end_points['sem_fea_full_l3'] = pointnet_fp_module(xyz, l3_xyz, points, l3_points, [], is_training, bn_decay, scope='sem_fa_layer2_fpn')
        end_points['sem_fea_full_l2'] = pointnet_fp_module(xyz, l2_xyz, points, l2_points, [], is_training, bn_decay, scope='sem_fa_layer3_fpn')
        end_points['sem_fea_full_l1'] = pointnet_fp_module(xyz, l1_xyz, points, l1_points, [], is_training, bn_decay, scope='sem_fa_layer4_fpn')

    # Feature Propagation layers
    #l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [256,256], is_training, bn_decay, scope='sem_fa_layer1')
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256,256], is_training, bn_decay, scope='sem_fa_layer2')
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,128], is_training, bn_decay, scope='sem_fa_layer3')
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128,128,128], is_training, bn_decay, scope='sem_fa_layer4')

    # FC layers
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='sem_fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    sem_class_logits = tf_util.conv1d(net, num_category, 1, padding='VALID', activation_fn=None, scope='sem_fc2')
    end_points['sem_class_logits'] = sem_class_logits

    return end_points

######################################################################################
######################################################################################
def get_rpointnet_class_loss(rpointnet_class_logits, gt_class_ids, roi_valid_mask):
    '''
    Inputs:
       rpointnet_class_logits: [B, NUM_ROIS, NUM_CATEGORY]
       gt_class_ids: [B, NUM_ROIS], zero padded
       roi_valid_mask: [B, NUM_ROIS]
    '''
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt_class_ids, logits=rpointnet_class_logits)
    loss = tf.multiply(loss, roi_valid_mask)
    loss = tf.divide(tf.reduce_sum(loss), tf.reduce_sum(roi_valid_mask)+1e-8)
    
    acc = tf.reduce_mean(tf.reduce_sum(tf.cast(tf.equal(tf.argmax(rpointnet_class_logits,axis=2,output_type = tf.int32),gt_class_ids),tf.float32)*roi_valid_mask,axis = 1)/tf.reduce_sum(roi_valid_mask,axis=1))
                          
    return loss, acc

######################################################################################
######################################################################################
def get_rpointnet_bbox_loss(gt_bbox, gt_class_ids, pred_bbox, roi_valid_mask, num_category):
    '''
    Inputs:
       gt_bbox: [B, NUM_ROIS, 6]
       gt_class_ids: [B, NUM_ROIS], zero padded        # for remove negative sample
       pred_bbox: [B, NUM_ROIS, NUM_CATEGORY, 6]
       roi_valid_mask: [B, NUM_ROIS]                   # for remove padding;
       num_category: scalar
    '''
    # Only foreground box contribute to the loss
    gt_bbox = tf.reshape(gt_bbox, [-1, 6])
    gt_class_ids = tf.reshape(gt_class_ids, [-1])
    pred_bbox = tf.reshape(pred_bbox, [-1, num_category, 6])
    roi_valid_mask = tf.reshape(roi_valid_mask, [-1])

    gt_selected_indices = tf.where(tf.logical_and(tf.greater(roi_valid_mask, 0), tf.greater(gt_class_ids, 0)))[:,0]
    gt_selected_indices = tf.cast(gt_selected_indices, tf.int32)
    pred_selected_indices = tf.concat((tf.reshape(gt_selected_indices, [-1,1]),
        tf.reshape(tf.gather(gt_class_ids, gt_selected_indices), [-1,1])), axis=1)

    gt_bbox = tf.gather(gt_bbox, gt_selected_indices, axis=0)
    pred_bbox = tf.gather_nd(pred_bbox, pred_selected_indices)

    loss = tf.cond(tf.size(gt_bbox)>0,
        lambda: tf.reduce_mean(tf.reduce_sum(smooth_l1_loss(y_true=gt_bbox, y_pred=pred_bbox),1),0),
        lambda: tf.constant(0.0))

    return loss

######################################################################################
######################################################################################
def get_rpointnet_mask_loss(gt_masks, gt_class_ids, pred_masks, roi_valid_mask, num_category):
    '''
    Inputs:
       gt_masks: [B, NUM_ROIS, NUM_POINT_PER_ROI]
       gt_class_ids: [B, NUM_ROIS], zero padded
       pred_masks: [B, NUM_ROIS, NUM_POINT_PER_ROI, NUM_CATEGORY]
       roi_valid_mask: [B, NUM_ROIS]
       num_category: scalar
       num_point_per_roi: scalar
    '''
    # Only foreground box contribute to the loss
    num_point_per_roi = gt_masks.get_shape()[2].value

    gt_masks = tf.reshape(gt_masks, [-1, num_point_per_roi])
    gt_class_ids = tf.reshape(gt_class_ids, [-1])
    pred_masks = tf.reshape(pred_masks, [-1, num_point_per_roi, num_category])
    pred_masks = tf.transpose(pred_masks, perm=[0,2,1])
    roi_valid_mask = tf.reshape(roi_valid_mask, [-1])

    gt_selected_indices = tf.where(tf.logical_and(tf.greater(roi_valid_mask, 0), tf.greater(gt_class_ids, 0)))[:,0]
    gt_selected_indices = tf.cast(gt_selected_indices, tf.int32)
    pred_selected_indices = tf.concat((tf.reshape(gt_selected_indices, [-1,1]),
        tf.reshape(tf.gather(gt_class_ids, gt_selected_indices), [-1,1])), axis=1)

    gt_masks = tf.gather(gt_masks, gt_selected_indices, axis=0) # [N, NUM_POINT_PER_ROI]
    gt_masks_int = gt_masks
    gt_masks = tf.cast(gt_masks, tf.float32)
    pred_masks = tf.gather_nd(pred_masks, pred_selected_indices) # [N, NUM_POINT_PER_ROI]

    loss = tf.cond(tf.size(gt_masks)>0,
        lambda: tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_masks, logits=pred_masks)),
        lambda: tf.constant(0.0))

    acc = tf.cond(tf.size(gt_masks)>0,
        lambda: tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.greater(pred_masks,0),tf.int32),gt_masks_int),tf.float32))/tf.cast(tf.size(gt_masks_int),tf.float32),
        lambda: tf.constant(1.0))

    return loss, acc   
   
#############################################################################################
#############################################################################################
# open curve reconstruction loss
# open_restrcut_loss = get_rpointnet_reconstruction_loss(open_gt_sample_points, open_gt_type, open_pre_sample_points, open_gt_valid_mask)
# closed_restrcut_loss = get_rpointnet_reconstruction_loss(closed_gt_sample_points, closed_gt_type, closed_pre_sample_points, closed_gt_valid_mask)
def get_rpointnet_reconstruction_loss(gt_sample_points, gt_class_ids, pre_sample_points, roi_valid_mask):   
    '''
    Inputs:
       gt_sample_points: [B, NUM_ROIS, NUM_POINT_PER_ROI_1,3]
       gt_class_ids: [B, NUM_ROIS], zero padded
       roi_valid_mask: [B, NUM_ROIS]
       pre_sample_points: [B, NUM_ROIS, NUM_CATEGORY, NUM_POINT_PER_ROI_2,3]
    '''
    # Only foreground box contribute to the loss
    num_category =  pre_sample_points.get_shape()[2].value
    num_point_per_roi_1 = gt_sample_points.get_shape()[2].value
    num_point_per_roi_2 = pre_sample_points.get_shape()[3].value
    
    gt_class_ids_gt = gt_class_ids
    gt_class_ids = tf.reshape(gt_class_ids, [-1])
    roi_valid_mask = tf.reshape(roi_valid_mask, [-1])

    # Positive sample index
    gt_selected_indices = tf.where(tf.logical_and(tf.greater(roi_valid_mask, 0), tf.greater(gt_class_ids, 0)))[:,0]
    gt_selected_indices = tf.cast(gt_selected_indices, tf.int32)
    pred_selected_indices = tf.concat((tf.reshape(gt_selected_indices, [-1,1]),
        tf.reshape(tf.gather(gt_class_ids, gt_selected_indices), [-1,1])), axis=1)

    # selection [B*N, num_point_per_roi_1, 3]  to  [X, num_point_per_roi_1, 3]
    gt_sample_points = tf.reshape(gt_sample_points, [-1, num_point_per_roi_1, 3]) # [B*nsmp, nsmp_ins, 3]
    gt_sample_points = tf.gather(gt_sample_points, gt_selected_indices, axis = 0) # [N, NUM_POINT_PER_ROI,3]
	
    # selection [B*N, num_category, num_point_per_roi_2, 3]  to  [X, num_point_per_roi_2, 3] 
    pre_sample_points = tf.reshape(pre_sample_points, [-1, num_category, num_point_per_roi_2, 3]) # [B*nsmp, nsmp_ins, 3]   note that : maybe they (num_point_per_roi_1 num_point_per_roi_2) are different size;
    pre_sample_points = tf.gather_nd(pre_sample_points, pred_selected_indices)

    # Chamfer distance: from gt_sample_points to pre_sample_points: dists_forward; 
    dists_forward,_,_,_ = tf_nndistance.nn_distance(gt_sample_points, pre_sample_points) # from GSPN tf_nndistance: [N, nsmp_ins(64)]

    recons_loss = tf.cond(tf.size(gt_sample_points)>0,
        lambda: tf.reduce_mean(tf.reduce_mean(dists_forward,1),0),
        lambda: tf.constant(0.0)) 
  
    return recons_loss     

	
######################################################################################
###################################################################################### 
# open curve reconstruction loss
# open_restrcut_loss = get_rpointnet_reconstruction_loss(open_gt_sample_points, open_gt_type, open_pre_sample_points, open_gt_valid_mask)
# closed_restrcut_loss = get_rpointnet_reconstruction_loss(closed_gt_sample_points, closed_gt_type, closed_pre_sample_points, closed_gt_valid_mask)
def get_rpointnet_reconstruction_loss_1(gt_sample_points, gt_class_ids, pre_sample_points, roi_valid_mask):   
    '''
    Inputs:
       gt_sample_points: [B, NUM_ROIS, NUM_POINT_PER_ROI,3]
       gt_class_ids: [B, NUM_ROIS], zero padded
       roi_valid_mask: [B, NUM_ROIS]
       pre_sample_points: [B, NUM_ROIS, NUM_CATEGORY, NUM_POINT_PER_ROI,3]
    '''
    # Only foreground box contribute to the loss
    num_category =  pre_sample_points.get_shape()[2].value
    num_point_per_roi = pre_sample_points.get_shape()[3].value
    
    gt_class_ids_gt = gt_class_ids
    gt_class_ids = tf.reshape(gt_class_ids, [-1])
    roi_valid_mask = tf.reshape(roi_valid_mask, [-1])

    gt_selected_indices = tf.where(tf.logical_and(tf.greater(roi_valid_mask, 0), tf.greater(gt_class_ids, 0)))[:,0]
    gt_selected_indices = tf.cast(gt_selected_indices, tf.int32)
    pred_selected_indices = tf.concat((tf.reshape(gt_selected_indices, [-1,1]),
        tf.reshape(tf.gather(gt_class_ids, gt_selected_indices), [-1,1])), axis=1)
    
    # normalize parameters; note that we just do the normalize, but we don't do centerize.  this is very important; if centering ,will be reduce the loss, the performace is so bad!!!!    
    bbox_size = tf.reduce_max(gt_sample_points, axis=2, keep_dims=True)-tf.reduce_min(gt_sample_points, axis=2, keep_dims=True)
    #radius = tf.sqrt(tf.reduce_sum(tf.square(bbox_size/2), axis=-1, keep_dims=True)) # [B, nsmp, 1, 1]  1e-8 + 
    
    # test radius
    radius = gt_class_ids_gt
    radius = tf.expand_dims(radius, 2)
    radius = tf.expand_dims(radius ,3)
    radius = tf.cast(radius, tf.float32)
    radius = tf.tile(radius, [1,1,num_point_per_roi,3])

    #pc_ins_gt_normalized = tf.reshape(tf.div(gt_sample_points, radius), [-1, num_point_per_roi, 3]) # [B*nsmp, nsmp_ins, 3]
    pc_ins_gt_normalized = tf.reshape(gt_sample_points, [-1, num_point_per_roi, 3]) # [B*nsmp, nsmp_ins, 3]
    #pc_ins_gt_normalized = tf.gather(pc_ins_gt_normalized, gt_selected_indices, axis = 0) # [N, NUM_POINT_PER_ROI,3]  
    pc_ins_gt_normalized = tf.gather(pc_ins_gt_normalized, gt_selected_indices) # [N, NUM_POINT_PER_ROI,3]
    

    # radius add 1 dim
    radius_for_pre =  tf.expand_dims(radius,2)
    #radius_for_pre =  tf.tile(tf.expand_dims(radius,2), [1,1,num_category, 1,1])

    #pc_ins_pred_normalized_all = tf.reshape(tf.div(pre_sample_points, radius_for_pre), [-1, num_category, num_point_per_roi, 3]) # [B*nsmp, num_category, nsmp_ins, 3]
    #pc_ins_pred_normalized_all = tf.reshape(pre_sample_points, [-1, num_category, num_point_per_roi, 3])

    # selection
    #pc_ins_pred_normalized_all = tf.div(pre_sample_points, radius_for_pre) # [B, nsmp, num_category, nsmp_ins, 3]
    pc_ins_pred_normalized_all = tf.gather(pre_sample_points, 3, axis = 2)
    Batch_num =  pre_sample_points.get_shape()[0].value
    Roi_num = pre_sample_points.get_shape()[1].value
    #pc_ins_pred_normalized_all = tf.zeros([Batch_num,Roi_num,num_point_per_roi,3])
    pc_ins_pred_normalized_all = tf.div(pc_ins_pred_normalized_all, radius)
    pc_ins_pred_normalized_all = tf.reshape(pc_ins_pred_normalized_all, [-1, num_point_per_roi, 3]) # [B*nsmp, nsmp_ins, 3]

    pc_ins_pred_normalized_all = tf.gather(pc_ins_pred_normalized_all, gt_selected_indices)
    #pc_ins_pred_normalized_all = tf.gather_nd(pc_ins_pred_normalized_all, pred_selected_indices) # [N, nsmp_ins(64),3]
    
    
    # category: 0 non   # here, we need put it to prediction module will be better.
    # recons_loss_0 = tf.constant()
    #pre_sample_points_0 = tf.gather(pre_sample_points, 0, axis = 2)
	#pc_ins_pred_normalized_0 = tf.reshape(tf.div(pre_sample_points_0, radius), [-1, nsmp_ins, 3]) # [B*nsmp, nsmp_ins, 3]
    
    # category: 1 cycle
    #pre_sample_points_1 = tf.gather(pre_sample_points, 1, axis = 2)
	#pc_ins_pred_normalized_1 = tf.reshape(tf.div(pre_sample_points_1, radius), [-1, nsmp_ins, 3]) # [B*nsmp, nsmp_ins, 3]

    # category: 2 line
    #pre_sample_points_2 = tf.gather(pre_sample_points, 2, axis = 2)
	#pc_ins_pred_normalized_2 = tf.reshape(tf.div(pre_sample_points_2, radius), [-1, nsmp_ins, 3]) # [B*nsmp, nsmp_ins, 3]  
  
    # category: 3 B_Spline
    #pre_sample_points_3 = tf.gather(pre_sample_points, 3, axis = 2)
	#pc_ins_pred_normalized_3 = tf.reshape(tf.div(pre_sample_points_3, radius), [-1, nsmp_ins, 3]) # [B*nsmp, nsmp_ins, 3]    
  
    # all category
    #pc_ins_pred_normalized_all = tf.stack([pc_ins_pred_normalized_0, pc_ins_pred_normalized_1, pc_ins_pred_normalized_2, pc_ins_pred_normalized_3], axis = 1)   # # [B*nsmp(1*256) = N, num_category(4), nsmp_ins(64), 3]
         

    dists_forward,_,_,_ = tf_nndistance.nn_distance(pc_ins_gt_normalized, pc_ins_pred_normalized_all) # from GSPN tf_nndistance: [N, nsmp_ins(64)]
    #recons_loss = tf.reduce_mean(dists_forward, axis=-1) # B*nsmp   tf.reduce_mean(dists_forward+dists_backward, axis=-1) # [N,1]
    #recons_loss = tf.reduce_mean(recons_loss, axis=-1) # [0]

    recons_loss = tf.cond(tf.size(pc_ins_gt_normalized)>0,
        lambda: tf.reduce_mean(pc_ins_pred_normalized_all),  #tf.reduce_mean(tf.reduce_mean(dists_forward, axis=-1)),
        lambda: tf.constant(0.0)) 
  
    return recons_loss 


    
           

######################################################################################
######################################################################################            
def get_stage_1(pc,  dof_feat,simmat_feat, open_gt_256_64_idx, closed_gt_256_64_idx, closed_gt_pair_idx, open_gt_mask, closed_gt_mask, is_training,bn_decay=None):
    
    #######################################################################################################################################################
    #task1: key_point
    feat1 = tf_util.conv1d(dof_feat,128,1,padding='VALID',activation_fn = None,scope = 'stage1/task1/fc1', bn_decay=bn_decay)
    pred_labels_key_p = tf_util.conv1d(feat1, 2, 1, padding='VALID', activation_fn=None, scope='stage1/task1/fc2', bn_decay=bn_decay)
    print("--- Get track 1")
    #######################################################################################################################################################
    #task1_2: corner_point
    feat1_1 = tf_util.conv1d(dof_feat,128,1,padding='VALID',activation_fn = None,scope = 'stage1/task1_1/fc1', bn_decay=bn_decay)
    pred_labels_corner_p = tf_util.conv1d(feat1_1, 2, 1, padding='VALID', activation_fn=None, scope='stage1/task1_1/fc2', bn_decay=bn_decay)
    
    print("--- Get track 2")
    #######################################################################################################################################################
    #task10: open_curve_feature
    feat10 = tf_util.conv1d(dof_feat,128,1,padding='VALID',activation_fn = None,scope = 'stage1/task10/fc1', bn_decay=bn_decay)
    pc_fea10_cropped, pc_coord10_cropped, _ , open_ball_radius, open_ball_center = points_cropping(pc, feat10, open_gt_256_64_idx, normalize_crop_region=True)
    #feat10_open_points, open_grouped_xyz = pointnet_sa_module_open(point_cloud, feat10, radius, center, mlp=[256,256,512], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='stage1/task10/fc2')
    print("--- Get track 3")	
    # Mask open corner ball prediction head

    open_Batch = open_gt_256_64_idx.get_shape()[0].value
    open_Roi_num = open_gt_256_64_idx.get_shape()[1].value
    open_num_per_Roi = open_gt_256_64_idx.get_shape()[2].value
    open_gt_mask_corner = tf.concat((tf.ones([open_Roi_num,2],tf.float32), 0.5*tf.ones([open_Roi_num,open_num_per_Roi-2],tf.float32)), -1)
    open_gt_mask_corner = tf.tile(tf.expand_dims(open_gt_mask_corner,0),[open_Batch,1,1])
    pc_fea10_cropped = pc_fea10_cropped*tf.expand_dims(open_gt_mask_corner,3)

    open_pre_mask = segmentation_head(pc_coord10_cropped,pc_fea10_cropped, 2,[64, 64], [64, 128, 512], [256, 256], is_training, bn_decay, 'open_segmentation_head')
    open_pre_sem_prob = tf.nn.softmax(open_pre_mask, -1) #[B, NUM_POINT(64), NUM_CATEGORY(0/1)]
    print("--- Get track 4")
    # Classification and bbox refinement head
    # mask-aware pc and pc_feat;
    # pc_mask_coord10_cropped = pc_coord10_cropped.*open_sem_prob
    # pc_mask_fea10_cropped = pc_fea10_cropped.*open_sem_prob
    open_gt_mask = tf.cast(tf.expand_dims(open_gt_mask,3),tf.float32)
    pc_fea10_cropped = pc_fea10_cropped*open_gt_mask
    pc_coord10_cropped = pc_coord10_cropped*open_gt_mask
    open_pre_class_logits, open_pre_class, open_pre_res, open_pre_sample_points, \
                                                    open_cycle_curve_pre, open_b_spline_curve_pre, open_line_curve_pre = classification_head_open(pc_coord10_cropped, pc_fea10_cropped, 3+1, \
                                                    64, [128, 256, 512], [256, 256], is_training, bn_decay, scope = 'classification_head_open')
    
    print("--- Get track 5")
    #######################################################################################################################################################
    #######################################################################################################################################################
    #task11: closed_curve_feature
    feat11 = tf_util.conv1d(simmat_feat,128,1,padding='VALID',activation_fn = None,scope = 'stage1/task11/fc1', bn_decay=bn_decay)
    pc_fea11_cropped, pc_coord11_cropped, _ , closed_ball_radius, closed_ball_center = points_cropping(pc, feat11, closed_gt_256_64_idx, normalize_crop_region=True)
    #pc_fea_cropped, pc_coord_cropped, pc_coord_corpped_knn_idx = points_cropping_3_and_knn(pc, labels_key_p, feat11, closed_points_cand_idx, num_closed = 256, num_sample_knn = 64) # For simple idx to extract pc and pc_feat
    print("--- Get track 6")
    #task11_1 Semantic prediction head
    pc_fea11_all_512_cropped, pc_coord11_all_512_cropped = points_cropping_low(pc, feat11, closed_gt_pair_idx)
    end_points = {}
    end_points = sem_net(pc_coord11_all_512_cropped, tf.concat((pc_coord11_all_512_cropped, pc_fea11_all_512_cropped), -1),2, end_points, 'stage1/task11/fc_semantic', is_training, bn_decay=None, return_fullfea=False)
    closed_pre_type = end_points['sem_class_logits']
    
    print("--- Get track 7")
    # Mask closed prediction head 

    closed_Batch = closed_gt_256_64_idx.get_shape()[0].value
    closed_Roi_num = closed_gt_256_64_idx.get_shape()[1].value
    closed_num_per_Roi = closed_gt_256_64_idx.get_shape()[2].value
    

   	
    closed_pre_mask = segmentation_head(pc_coord11_cropped,pc_fea11_cropped, 2,[64, 64], [64, 128, 512], [256, 256], is_training, bn_decay, 'closed_segmentation_head')
    closed_pre_sem_prob = tf.nn.softmax(closed_pre_mask, -1) #[B, NUM_POINT(64), NUM_CATEGORY(0/1)]
    print("--- Get track 8")
    # Classification and bbox refinement head

    closed_gt_mask = tf.cast(tf.expand_dims(closed_gt_mask,3),tf.float32)
    pc_fea11_cropped = pc_fea11_cropped*closed_gt_mask
    pc_coord11_cropped = pc_coord11_cropped*closed_gt_mask
#    closed_gt_mask = tf.concat((tf.ones([closed_Roi_num,2],tf.float32), 0.5*tf.ones([closed_Roi_num,closed_num_per_Roi-2],tf.float32)), -1)
#    closed_gt_mask = tf.tile(tf.expand_dims(closed_gt_mask,0),[closed_Batch,1,1])
#    pc_fea11_cropped = pc_fea11_cropped*tf.expand_dims(closed_gt_mask,3)


    closed_pre_class_logits, closed_pre_class, closed_pre_res, closed_pre_sample_points, closed_para_cycle = classification_head_closed(pc_coord11_cropped,pc_fea11_cropped, \
                                                                    2, [128, 256, 512], [256, 256], is_training, bn_decay, scope = 'classification_head_closed', bn=True)
    #closed_pre_type = closed_pre_class_logits 
    print("--- Get track 5")


    return pred_labels_key_p,pred_labels_corner_p, open_pre_mask, open_pre_class_logits, open_pre_res, closed_pre_type, closed_pre_mask, closed_pre_class_logits, closed_pre_res, \
        open_pre_sample_points, closed_pre_sample_points, \
        open_ball_radius, open_ball_center, \
        closed_ball_radius, closed_ball_center, \
        open_cycle_curve_pre, open_b_spline_curve_pre, open_line_curve_pre, \
        closed_para_cycle
        
        
######################################################################################
######################################################################################
def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typically: [N, 6], but could be any shape.
    """
    diff = tf.abs(y_true - y_pred)
    less_than_one = tf.cast(tf.less(diff, 1.0), tf.float32)
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return loss

######################################################################################
######################################################################################
def get_stage_1_loss(pred_labels_key_p,pred_labels_corner_p, labels_key_p,labels_corner_p, \
                    open_pre_mask, open_pre_class_logits, open_pre_res, open_pre_sample_points, open_gt_mask, open_gt_type, open_gt_res, open_gt_sample_points, open_gt_valid_mask, \
                    closed_pre_type, closed_pre_mask, closed_pre_class_logits, closed_pre_res, closed_pre_sample_points, \
                    closed_gt_mask, closed_gt_type, closed_gt_res, closed_gt_sample_points, closed_gt_valid_mask, \
                    open_ball_radius, open_ball_center, \
                    closed_ball_radius, closed_ball_center):
    batch_size = pred_labels_key_p.get_shape()[0].value
    num_point = pred_labels_key_p.get_shape()[1].value
    mask = tf.cast(labels_key_p,tf.float32)
    neg_mask = tf.ones_like(mask)-mask
    Np = tf.expand_dims(tf.reduce_sum(mask,axis=1),1)     
    Ng = tf.expand_dims(tf.reduce_sum(neg_mask,axis=1),1)  
    all_mask = tf.ones_like(mask)
    #loss:task1
    task_1_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = pred_labels_key_p,labels = labels_key_p)*(mask*(Ng/Np)+1))
    task_1_recall = tf.reduce_mean(tf.reduce_sum(tf.cast(tf.equal(tf.argmax(pred_labels_key_p,axis=2,output_type = tf.int32),\
                          labels_key_p),tf.float32)*mask,axis = 1)/tf.reduce_sum(mask,axis=1))
    task_1_acc = tf.reduce_mean(tf.reduce_sum(tf.cast(tf.equal(tf.argmax(pred_labels_key_p,axis=2,output_type = tf.int32),\
                          labels_key_p),tf.float32),axis = 1)/num_point)
    
    
    
    #loss:task1_1
    mask_1_1 = tf.cast(labels_corner_p,tf.float32)
    neg_mask_1_1 = tf.ones_like(mask_1_1)-mask_1_1
    Np_1_1 = tf.expand_dims(tf.reduce_sum(mask_1_1,axis=1),1)     
    Ng_1_1 = tf.expand_dims(tf.reduce_sum(neg_mask_1_1,axis=1),1) 
    task_1_1_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = pred_labels_corner_p,labels = labels_corner_p)*(mask_1_1*(Ng_1_1/Np_1_1)+1))
    task_1_1_recall = tf.reduce_mean(tf.reduce_sum(tf.cast(tf.equal(tf.argmax(pred_labels_corner_p,axis=2,output_type = tf.int32),\
                          labels_corner_p),tf.float32)*mask_1_1,axis = 1)/tf.reduce_sum(mask_1_1,axis=1))
    task_1_1_acc = tf.reduce_mean(tf.reduce_sum(tf.cast(tf.equal(tf.argmax(pred_labels_corner_p,axis=2,output_type = tf.int32),\
                          labels_corner_p),tf.float32),axis = 1)/num_point)
    
    
    #loss:task10
    # open curve classification loss
    open_gt_valid_mask = tf.cast(open_gt_valid_mask, tf.float32)
    open_class_loss, open_class_acc = get_rpointnet_class_loss(open_pre_class_logits, open_gt_type, open_gt_valid_mask)

    # open curve bbox loss
    open_res_loss = get_rpointnet_bbox_loss(open_gt_res, open_gt_type, open_pre_res, open_gt_valid_mask, num_category = 4)

    # open curve mask loss
    open_mask_loss, open_mask_acc = get_rpointnet_mask_loss(open_gt_mask, open_gt_type, open_pre_mask, open_gt_valid_mask, num_category = 4)
    
    # open curve reconstruction loss
    # 1), centering and normlizing gt sampe points;
    open_gt_sample_points = open_gt_sample_points-open_ball_center
    open_gt_sample_points = tf.divide(open_gt_sample_points, open_ball_radius)
    open_restrcut_loss = get_rpointnet_reconstruction_loss(open_gt_sample_points, open_gt_type, open_pre_sample_points, open_gt_valid_mask)
    
    
    #loss:task11
    # closed rpointnet classification loss
    closed_gt_valid_mask = tf.cast(closed_gt_valid_mask, tf.float32)
    closed_class_loss, closed_class_acc = get_rpointnet_class_loss(closed_pre_type, closed_gt_type, closed_gt_valid_mask)

    # closed curve bbox loss
    closed_res_loss = get_rpointnet_bbox_loss(closed_gt_res, closed_gt_type, closed_pre_res, closed_gt_valid_mask, num_category = 2)

    # closed curve mask loss
    closed_mask_loss, closed_mask_acc = get_rpointnet_mask_loss(closed_gt_mask, closed_gt_type, closed_pre_mask, closed_gt_valid_mask, num_category = 2)
    
    # closed curve reconstruction loss
    # 1), centering and normlizing gt sampe points;
    closed_gt_sample_points = closed_gt_sample_points-closed_ball_center
    closed_gt_sample_points = tf.divide(closed_gt_sample_points, closed_ball_radius)   
    closed_restruct_loss = get_rpointnet_reconstruction_loss(closed_gt_sample_points, closed_gt_type, closed_pre_sample_points, closed_gt_valid_mask)
       
    w1 = 0
    w1_1 = 0
    
    w10_1 = 1
    w10_2 = 0
    w10_3 = 1
    w10_4 = 10
    
    
    w11_1 = 1
    w11_2 = 0
    w11_3 = 1
    w11_4 = 10
    
    
    loss = task_1_loss*w1 + task_1_1_loss*w1_1 + open_class_loss*w10_1 + open_res_loss*w10_2 + open_mask_loss*w10_3 + open_restrcut_loss*w10_4 + \
           closed_class_loss*w11_1 + closed_res_loss*w11_2 + closed_mask_loss*w11_3 + closed_restruct_loss*w11_4
    tf.summary.scalar('all loss', loss)
    tf.add_to_collection('losses', loss)
    return task_1_loss,task_1_recall,task_1_acc,task_1_1_loss,task_1_1_recall,task_1_1_acc, \
            open_class_loss,open_res_loss,open_mask_loss, open_restrcut_loss, \
            closed_class_loss,closed_res_loss,closed_mask_loss, closed_restruct_loss, loss, \
            open_class_acc, open_mask_acc, closed_class_acc, closed_mask_acc