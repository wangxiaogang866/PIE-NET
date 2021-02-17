import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_fp_module

def smooth_l1_dist(deltas, sigma2=2.0, name='smooth_l1_dist'):
    with tf.name_scope(name=name) as scope:
        deltas_abs = tf.abs(deltas)
        smoothL1_sign = tf.cast(tf.less(deltas_abs, 1.0/sigma2), tf.float32)
        return tf.square(deltas) * 0.5 * sigma2 * smoothL1_sign + \
                   (deltas_abs - 0.5 / sigma2) * tf.abs(smoothL1_sign - 1)




def placeholder_inputs_stage_1(batch_size,num_point,num_roi):
    pointclouds_pl = tf.placeholder(tf.float32,shape=(batch_size,num_point,5))  
    corner_pair = tf.placeholder(tf.int32,shape=(batch_size,num_roi,2))
    label_1 = tf.placeholder(tf.int32,shape=(batch_size,num_roi))  
    label_2 = tf.placeholder(tf.int32,shape=(batch_size,num_roi)) 
    #label_3 = tf.placeholder(tf.int32,shape=(batch_size,num_roi))  
    #label_4 = tf.placeholder(tf.int32,shape=(batch_size,num_roi)) 
    label_5 = tf.placeholder(tf.int32,shape=(batch_size,num_point)) 
    label_6 = tf.placeholder(tf.int32,shape=(batch_size,num_point))
    label_7 = tf.placeholder(tf.int32,shape=(batch_size,num_point)) 
    label_8 = tf.placeholder(tf.int32,shape=(batch_size,num_point))  
    #labels_direction = tf.placeholder(tf.int32,shape=(batch_size,num_point))
#    regression_direction = tf.placeholder(tf.float32,shape=(batch_size,num_point,3))
#    regression_position = tf.placeholder(tf.float32,shape=(batch_size,num_point,3))
#    labels_type = tf.placeholder(tf.int32,shape=(batch_size,num_point))
#    simmat_pl = tf.placeholder(tf.float32,shape=(batch_size,num_point,num_point))
#    neg_simmat_pl = tf.placeholder(tf.float32,shape=(batch_size,num_point,num_point))
#    return pointclouds_pl,labels_key_p,labels_direction,regression_direction,regression_position,labels_type,simmat_pl,neg_simmat_pl
    return pointclouds_pl,corner_pair,label_1,label_2,label_5,label_6,label_7,label_8		

	

	

######################################################################################
######################################################################################
def points_cropping(pc, pc_fea, masks_selection_idx, num_rois, num_point_per_roi):
    ''' Crop points for network heads, in analogy to ROIAlign
    Inputs:
        pc: [B, NUM_POINT, 3]
        pc_fea: [B, NUM_POINT, NFEA]
        masks_selection_idx: [B, NUM_ROIS, NUM_POINT_PER_ROI]
    Returns:
        pc_fea_cropped: [B, NUM_ROIS, NUM_POINT_PER_ROI, NFEA]
        pc_coord_cropped: [B, NUM_ROIS, NUM_POINT_PER_ROI, 3]
    '''
    batch_size = pc.get_shape()[0].value
    smp_idx = tf.reshape(tf.tile(tf.reshape(tf.range(batch_size),[-1,1]),[1, num_rois*num_point_per_roi]),[-1,1])
    smp_idx = tf.concat((smp_idx, tf.reshape(masks_selection_idx,[-1,1])),1)
    pc_fea_cropped = tf.reshape(tf.gather_nd(pc_fea, smp_idx),[batch_size, num_rois, num_point_per_roi, -1])
    pc_coord_cropped_unnormalized = tf.reshape(tf.gather_nd(pc, smp_idx),[batch_size, num_rois, num_point_per_roi, -1])
    pc_coord_cropped = pc_coord_cropped_unnormalized

    return pc_fea_cropped, pc_coord_cropped
	
def get_feature(point_cloud, pair_mask, is_training,stage,bn_decay=None):
    """ Part segmentation PointNet, input is BxNx6 (XYZ NormalX NormalY NormalZ), output Bx50 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = tf.slice(point_cloud, [0,0,0], [-1,-1,3])
    l0_points = tf.slice(point_cloud, [0,0,3], [-1,-1,2])

#    # Set Abstraction layers#
#    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=512, radius=0.2, nsample=64, mlp=[64,64,128], mlp2=None, group_all=False, #is_training=is_training, bn_decay=bn_decay, scope='pointnet/layer1')
#    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=128, radius=0.4, nsample=64, mlp=[128,128,256], mlp2=None, group_all=False, #is_training=is_training, bn_decay=bn_decay, scope='pointnet/layer2')
#    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, #is_training=is_training, bn_decay=bn_decay, scope='pointnet/layer3')

#    # Feature Propagation layers
#    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256,256], is_training, bn_decay, scope='pointnet/fa_layer1')
#    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,128], is_training, bn_decay, scope='pointnet/fa_layer2')
#    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, tf.concat([l0_xyz,l0_points],axis=-1), l1_points, [128,128,128], is_training, bn_decay, scope='pointnet/fa_layer3')

    # Layer 1
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=4096, radius=0.05, nsample=32, mlp=[32,32,64], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=2048, radius=0.1, nsample=32, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=1024, radius=0.2, nsample=32, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer3')
    l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=512, radius=0.4, nsample=32, mlp=[256,256,512], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer4')

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
	
	#task11_1 Semantic prediction head
    #pc_fea11_cropped, pc_coord11_cropped = points_cropping(l0_xyz, dof_feat, pair_mask, num_rois = 128, num_point_per_roi = 2)
    #tf.concat((pc_coord11_cropped, pc_fea11_cropped), -1)

    # Points cropping - pc_fea_cropped: [B, NUM_ROIS, NUM_POINT_PER_ROI, NFEA]
    # pc_coord_cropped: [B, NUM_ROIS, NUM_POINT_PER_ROI, 3]
    ##### sem fpn fea
    #sem_fea_full_l1 = tf_util.conv1d(end_points['sem_fea_full_l1'], 64, 1, padding='VALID', bn=True, is_training=is_training, scope='fpn1', bn_decay=bn_decay)
    #sem_fea_full_l2 = tf_util.conv1d(end_points['sem_fea_full_l2'], 64, 1, padding='VALID', bn=True, is_training=is_training, scope='fpn2', bn_decay=bn_decay)
    #sem_fea_full_l3 = tf_util.conv1d(end_points['sem_fea_full_l3'], 64, 1, padding='VALID', bn=True, is_training=is_training, scope='fpn3', bn_decay=bn_decay)
    #sem_fea_full_l4 = tf_util.conv1d(end_points['sem_fea_full_l4'], 64, 1, padding='VALID', bn=True, is_training=is_training, scope='fpn4', bn_decay=bn_decay)
    #pc_fea_cropped, pc_coord_cropped = points_cropping(pc, tf.concat((end_points['entity_fea'], sem_fea_full_l1, sem_fea_full_l2, sem_fea_full_l3, sem_fea_full_l4), -1), 
    #    end_points['center_pos'], rois, target_mask_selection_idx, config.TRAIN_ROIS_PER_IMAGE, config.NUM_POINT_INS_MASK, config.NORMALIZE_CROP_REGION)
    pc_fea_cropped, pc_coord_cropped = points_cropping(l0_xyz, dof_feat, pair_mask, num_rois = 8128, num_point_per_roi = 2)

	
    return end_points,pc_fea_cropped,pc_coord_cropped,point_cloud,dof_feat 	
	
	
		
######################################################################################
######################################################################################		
def task_head(pc, pc_fea, num_category, mlp_list, mlp_list2, is_training, bn_decay, scope, bn=True):
    '''
    Inputs:
        pc: [B, NUM_ROIS, NUM_POINT_PER_ROI, 3]
        pc_fea: [B, NUM_ROIS, NUM_POINT_PER_ROI, NFEA]
        num_category: scalar
    Returns:
        logits: [B, NUM_ROIS, NUM_CATEGORY]
        probs: [B, NUM_ROIS, NUM_CATEGORY]
        #bbox_deltas: [B, NUM_ROIS, NUM_CATEGORY, (dz, dy, dx, log(dh), log(dw), log(dl))]
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
#        task_deltas = tf_util.conv1d(new_points, num_category*6, 1, padding='VALID',
#                                     stride=1, scope='conv_bbox_regress', activation_fn=None)
#        task_deltas = tf.reshape(bbox_deltas, [-1, num_rois, num_category, 6])
        return logits, probs   #, task_deltas
	
	
def get_stage_1(pc_coord_cropped , pc_fea_cropped, dof_feat, is_training,bn_decay=None):

    # task 1
    class_logits_1, class_pro_1 = task_head(pc_coord_cropped, 
                pc_fea_cropped, 2,[128, 256, 512], [256, 256], is_training, bn_decay, 'cls1')	

    # task 2
    class_logits_2, class_pro_2 = task_head(pc_coord_cropped, 
                pc_fea_cropped, 2,[128, 256, 512], [256, 256], is_training, bn_decay, 'cls2')
				
    # task 3
    #class_logits_3, class_pro_3 = task_head(pc_coord_cropped, 
    #            pc_fea_cropped, 2,[128, 256, 512], [256, 256], is_training, bn_decay, 'cls3')	

    # task 4
    #class_logits_4, class_pro_4 = task_head(pc_coord_cropped, 
    #            pc_fea_cropped, 2,[128, 256, 512], [256, 256], is_training, bn_decay, 'cls4')	

    #task5
    feat1 = tf_util.conv1d(dof_feat,128,1,padding='VALID',activation_fn = None,scope = 'stage1/task1/fc1', bn_decay=bn_decay)
    class_logits_5 = tf_util.conv1d(feat1, 2, 1, padding='VALID', activation_fn=None, scope='stage1/task1/fc2', bn_decay=bn_decay)
    
    #task6
    feat2 = tf_util.conv1d(dof_feat,128,1,padding='VALID',activation_fn = None,scope = 'stage1/task1_1/fc1', bn_decay=bn_decay)
    class_logits_6 = tf_util.conv1d(feat2, 2, 1, padding='VALID', activation_fn=None, scope='stage1/task2/fc2', bn_decay=bn_decay)
    
    #task7
    feat3 = tf_util.conv1d(dof_feat,128,1,padding='VALID',activation_fn = None,scope = 'stage1/task3/fc1', bn_decay=bn_decay)
    class_logits_7 = tf_util.conv1d(feat3, 2, 1, padding='VALID', activation_fn=None, scope='stage1/task3/fc2', bn_decay=bn_decay)
    
    #task8
    feat4 = tf_util.conv1d(dof_feat,128,1,padding='VALID',activation_fn = None,scope = 'stage1/task4/fc1', bn_decay=bn_decay)
    class_logits_8 = tf_util.conv1d(feat4, 2, 1, padding='VALID', activation_fn=None, scope='stage1/task4/fc2', bn_decay=bn_decay)
    return class_logits_1, class_logits_2, class_logits_5, class_logits_6, class_logits_7, class_logits_8

	
	
######################################################################################
######################################################################################	
# loss
def get_rpointnet_loss(class_logits, gt_class_ids):
    '''
    Inputs:
       class_logits: [B, NUM_ROIS, NUM_CATEGORY]
       gt_class_ids: [B, NUM_ROIS], zero padded
       #roi_valid_mask: [B, NUM_ROIS]
    '''
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt_class_ids, logits=class_logits)
    #loss = tf.multiply(loss, roi_valid_mask)
    num_batch = gt_class_ids.get_shape()[0].value
    num_roi = gt_class_ids.get_shape()[1].value
    loss = tf.divide(tf.reduce_sum(loss), num_batch*num_roi)  #tf.reduce_sum(roi_valid_mask)+1e-8)
	
    mask_pos = tf.cast(gt_class_ids,tf.float32)
    recall = tf.reduce_mean(tf.reduce_sum(tf.cast(tf.equal(tf.argmax(class_logits,axis=2,output_type = tf.int32),\
                          gt_class_ids),tf.float32)*mask_pos,axis = 1)/tf.reduce_sum(mask_pos,axis=1))
						  
    acc = tf.reduce_mean(tf.reduce_sum(tf.cast(tf.equal(tf.argmax(class_logits,axis=2,output_type = tf.int32),\
                          gt_class_ids),tf.float32),axis = 1)/num_roi)					  

    return loss, recall, acc

def get_rpointnet_loss_1(class_logits, gt_class_ids):
    '''
    Inputs:
       class_logits: [B, NUM_ROIS, NUM_CATEGORY]
       gt_class_ids: [B, NUM_ROIS], zero padded
       #roi_valid_mask: [B, NUM_ROIS]
    '''
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt_class_ids, logits=class_logits)
    num_batch = gt_class_ids.get_shape()[0].value
    num_roi = gt_class_ids.get_shape()[1].value
    loss = tf.divide(tf.reduce_sum(loss), num_batch*num_roi)  #tf.reduce_sum(roi_valid_mask)+1e-8)

    batch_size = gt_class_ids.get_shape()[0].value
    num_point = gt_class_ids.get_shape()[1].value
    mask = tf.cast(gt_class_ids,tf.float32)
    neg_mask = tf.ones_like(mask)-mask
    Np = tf.expand_dims(tf.reduce_sum(mask,axis=1),1)     
    Ng = tf.expand_dims(tf.reduce_sum(neg_mask,axis=1),1)  
    all_mask = tf.ones_like(mask)
    #loss:task5
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = class_logits,labels = gt_class_ids)*(mask*(Ng/Np)+1))	
    mask_pos = tf.cast(gt_class_ids,tf.float32)
    recall = tf.reduce_mean(tf.reduce_sum(tf.cast(tf.equal(tf.argmax(class_logits,axis=2,output_type = tf.int32),\
                          gt_class_ids),tf.float32)*mask_pos,axis = 1)/tf.reduce_sum(mask_pos,axis=1))
						  
    acc = tf.reduce_mean(tf.reduce_sum(tf.cast(tf.equal(tf.argmax(class_logits,axis=2,output_type = tf.int32),\
                          gt_class_ids),tf.float32),axis = 1)/num_roi)					  

    return loss, recall, acc


	
def get_stage_1_loss(class_logits_1, gt_class_ids_1, class_logits_2, gt_class_ids_2, class_logits_5, gt_class_ids_5, class_logits_6, gt_class_ids_6, class_logits_7, gt_class_ids_7, class_logits_8, gt_class_ids_8):

    #loss:task1
    loss_1, recall_1, acc_1 = get_rpointnet_loss(class_logits_1, gt_class_ids_1)

    #loss:task2
    loss_2, recall_2, acc_2 = get_rpointnet_loss(class_logits_2, gt_class_ids_2)

    #loss:task3
    #loss_3, recall_3, acc_3 = get_rpointnet_loss(class_logits_3, gt_class_ids_3)

    #loss:task4
    #loss_4, recall_4, acc_4 = get_rpointnet_loss(class_logits_4, gt_class_ids_4)

    #loss:task5
    loss_5, recall_5, acc_5 = get_rpointnet_loss(class_logits_5, gt_class_ids_5)

    #loss:task4
    loss_6, recall_6, acc_6 = get_rpointnet_loss(class_logits_6, gt_class_ids_6)
    
    #loss:task7
    loss_7, recall_7, acc_7 = get_rpointnet_loss(class_logits_7, gt_class_ids_7)

    #loss:task8
    loss_8, recall_8, acc_8 = get_rpointnet_loss(class_logits_8, gt_class_ids_8)

    #loss
    w1 = 1
    w2 = 1
    w3 = 1
    w4 = 1
    w5 = 1
    w6 = 2
    w7 = 2
    w8 = 2
    loss = w1*loss_1 + w2*loss_2 + w5*loss_5 + w6*loss_6 + w7*loss_7 + w8*loss_8	
	
    return loss, loss_1, recall_1, acc_1, loss_2, recall_2, acc_2, loss_5, recall_5, acc_5, loss_6, recall_6, acc_6, loss_7, recall_7, acc_7, loss_8, recall_8, acc_8
