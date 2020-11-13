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

def get_feature(point_cloud, is_training,stage,bn_decay=None):
    """ Part segmentation PointNet, input is BxNx6 (XYZ NormalX NormalY NormalZ), output Bx50 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = tf.slice(point_cloud, [0,0,0], [-1,-1,3])
    l0_points = tf.slice(point_cloud, [0,0,3], [-1,-1,0])

#    # Set Abstraction layers#
#    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=512, radius=0.2, nsample=64, mlp=[64,64,128], mlp2=None, group_all=False, #is_training=is_training, bn_decay=bn_decay, scope='pointnet/layer1')
#    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=128, radius=0.4, nsample=64, mlp=[128,128,256], mlp2=None, group_all=False, #is_training=is_training, bn_decay=bn_decay, scope='pointnet/layer2')
#    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, #is_training=is_training, bn_decay=bn_decay, scope='pointnet/layer3')

#    # Feature Propagation layers
#    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256,256], is_training, bn_decay, scope='pointnet/fa_layer1')
#    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,128], is_training, bn_decay, scope='pointnet/fa_layer2')
#    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, tf.concat([l0_xyz,l0_points],axis=-1), l1_points, [128,128,128], is_training, bn_decay, scope='pointnet/fa_layer3')

    # Layer 1
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=1024, radius=0.05, nsample=32, mlp=[32,32,64], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=512, radius=0.1, nsample=32, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=256, radius=0.2, nsample=32, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer3')
    l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=128, radius=0.4, nsample=32, mlp=[256,256,512], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer4')

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

def placeholder_inputs_stage_1(batch_size,num_point):
    pointclouds_pl = tf.placeholder(tf.float32,shape=(batch_size,num_point,3))  # input
    labels_key_p = tf.placeholder(tf.int32,shape=(batch_size,num_point))  # edge points label 0/1
    labels_corner_p = tf.placeholder(tf.int32,shape=(batch_size,num_point)) 
    #labels_direction = tf.placeholder(tf.int32,shape=(batch_size,num_point))
#    regression_direction = tf.placeholder(tf.float32,shape=(batch_size,num_point,3))
#    regression_position = tf.placeholder(tf.float32,shape=(batch_size,num_point,3))
#    labels_type = tf.placeholder(tf.int32,shape=(batch_size,num_point))
#    simmat_pl = tf.placeholder(tf.float32,shape=(batch_size,num_point,num_point))
#    neg_simmat_pl = tf.placeholder(tf.float32,shape=(batch_size,num_point,num_point))
#    return pointclouds_pl,labels_key_p,labels_direction,regression_direction,regression_position,labels_type,simmat_pl,neg_simmat_pl
    return pointclouds_pl,labels_key_p,labels_corner_p

def get_stage_1(dof_feat,simmat_feat,is_training,bn_decay=None):
    batch_size = dof_feat.get_shape()[0].value

    #task1: key_point
    feat1 = tf_util.conv1d(dof_feat,128,1,padding='VALID',activation_fn = None,scope = 'stage1/task1/fc1', bn_decay=bn_decay)
    pred_labels_key_p = tf_util.conv1d(feat1, 2, 1, padding='VALID', activation_fn=None, scope='stage1/task1/fc2', bn_decay=bn_decay)
    
    #task1_2: corner_point
    feat1_1 = tf_util.conv1d(dof_feat,128,1,padding='VALID',activation_fn = None,scope = 'stage1/task1_1/fc1', bn_decay=bn_decay)
    pred_labels_corner_p = tf_util.conv1d(feat1_1, 2, 1, padding='VALID', activation_fn=None, scope='stage1/task1_1/fc2', bn_decay=bn_decay)

#    #task2_1: labels_direction
#    feat2_1 = tf_util.conv1d(dof_feat,128,1,padding='VALID',activation_fn = None,scope = 'stage1/task2_1/fc1', bn_decay=bn_decay)
#    pred_labels_direction = tf_util.conv1d(feat2_1, 15, 1, padding='VALID', activation_fn=None, scope='stage1/task2_1/fc2', bn_decay=bn_decay)

#    #task2_2: regression_direction
#    feat2_2 = tf_util.conv1d(dof_feat,128,1,padding='VALID',activation_fn = None,scope = 'stage1/task2_2/fc1', bn_decay=bn_decay)
#    pred_regression_direction = tf_util.conv1d(feat2_2, 3, 1, padding='VALID', activation_fn=None, scope='stage1/task2_2/fc2', bn_decay=bn_decay)

#    #task_3: position
#    feat3 = tf_util.conv1d(dof_feat,128,1,padding='VALID',activation_fn = None,scope = 'stage1/task3/fc1', bn_decay=bn_decay)
#    pred_regression_position = tf_util.conv1d(feat3, 3, 1, padding='VALID', activation_fn=None, scope='stage1/task3/fc2', bn_decay=bn_decay)
#
#    #task_4: dof_type
#    feat4 = tf_util.conv1d(dof_feat,128,1,padding='VALID',activation_fn = None,scope = 'stage1/task4/fc1', bn_decay=bn_decay)
#    pred_labels_type = tf_util.conv1d(feat4, 4, 1, padding='VALID', activation_fn=None, scope='stage1/task4/fc2', bn_decay=bn_decay)
#
#    #task_5: similar matrix
#    feat5 = tf_util.conv1d(simmat_feat,128,1,padding='VALID',activation_fn = None,scope = 'stage1/task_5/fc1', bn_decay=bn_decay)
#    r = tf.reduce_sum(feat5*feat5,2)
#    r = tf.reshape(r, [batch_size, -1, 1])
#    D = r-2*tf.matmul(feat5,tf.transpose(feat5,perm=[0,2,1]))+tf.transpose(r, perm=[0,2,1])
#    pred_simmat = tf.maximum(10*D,0.)
#
#    #task_6: confidence map
#    feat6 = tf_util.conv1d(simmat_feat,128,1,padding='VALID',activation_fn = None,scope = 'stage1/task6/fc1', bn_decay=bn_decay)
#    conf_logits = tf_util.conv1d(feat6,1,1,padding='VALID',activation_fn = None,scope = 'stage1/task_6/fc2', bn_decay=bn_decay)
#    pred_conf_logits = tf.nn.sigmoid(conf_logits, name='stage1/task_6/confidence')

#    return pred_labels_key_p,pred_labels_direction,pred_regression_direction,pred_regression_position, \
#                                             pred_labels_type,pred_simmat,pred_conf_logits
    return pred_labels_key_p,pred_labels_corner_p

def get_stage_1_loss(pred_labels_key_p,pred_labels_corner_p, \
                       labels_key_p,labels_corner_p):
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
    
#    #loss:task2_1
#    task_2_1_loss =  tf.reduce_mean(tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = pred_labels_direction,\
#                               labels = labels_direction)*mask,axis = 1)/tf.reduce_sum(mask,axis=1))
#    task_2_1_acc = tf.reduce_mean(tf.reduce_sum(tf.cast(tf.equal(tf.argmax(pred_labels_direction,axis=2,output_type=tf.int32), \
#                               labels_direction),tf.float32)*mask,axis=1)/tf.reduce_sum(mask,axis=1))
#    #loss:task2_2
#    task_2_2_loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_mean(smooth_l1_dist(pred_regression_direction-regression_direction),axis=2)*mask, \
#                               axis = 1)/tf.reduce_sum(mask,axis=1))
#    #loss:task3
#    task_3_loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_mean(smooth_l1_dist(pred_regression_position-regression_position),axis=2)*mask, \
#                               axis = 1)/tf.reduce_sum(mask,axis=1))
#    #loss:task4
#    task_4_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = pred_labels_type,labels = labels_type)*mask,axis = 1)/tf.reduce_sum(mask,axis=1))
#    task_4_acc = tf.reduce_mean(tf.reduce_sum(tf.cast(tf.equal(tf.argmax(pred_labels_type,axis=2,output_type = tf.int32),\
#                          labels_type),tf.float32)*mask,axis = 1)/tf.reduce_sum(mask,axis=1))
#
#    #loss: task_5
#    pos = pred_simmat*simmat_pl
#    neg = tf.maximum(80-pred_simmat,0) * neg_simmat_pl
#    task_5_loss = tf.reduce_mean(pos+neg)
#    #loss: task_6
#    ng_label = tf.greater(simmat_pl,0.5)
#    ng = tf.less(pred_simmat,80)
#    epsilon = tf.constant(np.ones(ng_label.get_shape()[:2]).astype(np.float32) *1e-6)
#    pts_iou = tf.reduce_sum(tf.cast(tf.logical_and(ng, ng_label), tf.float32), axis=2) / \
#                      (tf.reduce_sum(tf.cast(tf.logical_or(ng, ng_label), tf.float32), axis=2) + epsilon)
#    task_6_loss = tf.reduce_mean(tf.squared_difference(pts_iou,tf.squeeze(pred_conf_logits,[2])))
    w1 = 1
    w1_1 = 1
    
    w2_1 = 1
    w2_2 = 1
    w3 = 100
    w4 = 1
    w5 = 1
    w6 = 100

#    loss = task_1_loss*w1 + task_2_1_loss*w2_1 + task_2_2_loss*w2_2 + task_3_loss*w3 + task_4_loss*w4 + task_5_loss*w5 + task_6_loss*w6
    loss = task_1_loss*w1 + task_1_1_loss*w1_1

    tf.summary.scalar('all loss', loss)
    tf.add_to_collection('losses', loss)
#    return task_1_loss,task_1_recall,task_1_acc,task_2_1_loss,task_2_1_acc,task_2_2_loss,task_3_loss,task_4_loss,task_4_acc,task_5_loss,task_6_loss,loss
    return task_1_loss,task_1_recall,task_1_acc,task_1_1_loss,task_1_1_recall,task_1_1_acc,loss

def placeholder_inputs_stage_2(batch_size,num_point):
    pointclouds_pl = tf.placeholder(tf.float32,shape=(batch_size,num_point,6))
    proposal_nx_pl = tf.placeholder(tf.int32,shape=(batch_size,num_point))
    dof_mask_pl = tf.placeholder(tf.int32,shape=(batch_size,num_point))
    dof_score_pl = tf.placeholder(tf.float32,shape=(batch_size,num_point))
    return pointclouds_pl,proposal_nx_pl,dof_mask_pl,dof_score_pl

def get_stage_2(dof_feat,simmat_feat,dof_mask_pl,proposal_nx_pl,is_training,bn_decay=None):

    dof_feat = tf_util.conv1d(dof_feat,512,1,padding='VALID',activation_fn = None,scope = 'stage2/task1/smat_fc1')
    simmat_feat = tf_util.conv1d(simmat_feat,512,1,padding='VALID',activation_fn = None,scope = 'stage2/task1/dof_fc1')
    proposal_nx_pl = tf.expand_dims(proposal_nx_pl,axis = -1)
    proposal_nx_pl = tf.cast(tf.tile(proposal_nx_pl,[1,1,512]),tf.float32)
    simmat_feat_mul = simmat_feat * proposal_nx_pl
    simmat_feat_reduce = tf.reduce_max(simmat_feat_mul,axis=1)
    simmat_feat_expand = tf.tile(tf.expand_dims(simmat_feat_reduce,axis=1),[1,4096,1])
    simmat_feat_all = tf.reduce_max(simmat_feat,axis=1)
    simmat_feat_all = tf.tile(tf.expand_dims(simmat_feat_all,axis=1),[1,4096,1])
    all_feat = tf.concat([dof_feat,simmat_feat_expand],axis = 2)
    dof_mask_pl = tf.expand_dims(dof_mask_pl,axis =-1)
    dof_mask_pl = tf.cast(tf.tile(dof_mask_pl,[1,1,1024]),tf.float32)
    all_feat = all_feat * dof_mask_pl
    feat1 = tf_util.conv1d(all_feat,1024,1,padding='VALID',activation_fn = None,scope = 'stage2/task1/fc1')
    feat2 = tf_util.conv1d(feat1,512,1,padding='VALID',activation_fn = None,scope = 'stage2/task1/fc2')
    feat3 = tf_util.conv1d(feat2,256,1,padding='VALID',activation_fn = None,scope = 'stage2/task1/fc3')
    pred_dof_score = tf_util.conv1d(feat3, 1,1, padding='VALID', activation_fn=None, scope='stage2/task1/fc4')
    pred_dof_score = tf.nn.sigmoid(pred_dof_score, name='stage2/task_1/score')
    pred_dof_score = tf.squeeze(pred_dof_score,axis = -1)
    return pred_dof_score


def get_stage_2_loss(pred_dof_score,dof_score_pl,dof_mask_pl):
    dof_mask_pl = tf.cast(dof_mask_pl,tf.float32)
    dof_score_pl = tf.expand_dims(dof_score_pl,-1)
    pred_dof_score = tf.expand_dims(pred_dof_score,axis = -1)
    loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_mean(smooth_l1_dist(pred_dof_score-dof_score_pl),axis=2)*dof_mask_pl, \
                               axis = 1)/tf.reduce_sum(dof_mask_pl,axis=1))
    return loss
