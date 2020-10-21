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
    # simmat_feature
    simmat_feat = tf_util.conv1d(net, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='pointnet/fc_simmat', bn_decay=bn_decay)
    return end_points,dof_feat,simmat_feat

def placeholder_inputs_stage_1(batch_size,num_point):
    pointclouds_pl = tf.placeholder(tf.float32,shape=(batch_size,num_point,3))  # input
    labels_key_p = tf.placeholder(tf.int32,shape=(batch_size,num_point))  # edge points label 0/1
    labels_corner_p = tf.placeholder(tf.int32,shape=(batch_size,num_point)) 
    regression_position = tf.placeholder(tf.float32,shape=(batch_size,num_point,3))

    return pointclouds_pl,labels_key_p,labels_corner_p,regression_position

def get_stage_1(dof_feat,simmat_feat,is_training,bn_decay=None):
    batch_size = dof_feat.get_shape()[0].value

    #task1: key_point
    feat1 = tf_util.conv1d(dof_feat,128,1,padding='VALID',activation_fn = None,scope = 'stage1/task1/fc1', bn_decay=bn_decay)
    pred_labels_key_p = tf_util.conv1d(feat1, 2, 1, padding='VALID', activation_fn=None, scope='stage1/task1/fc2', bn_decay=bn_decay)
    
    #task1_2: corner_point
    feat1_1 = tf_util.conv1d(dof_feat,128,1,padding='VALID',activation_fn = None,scope = 'stage1/task1_1/fc1', bn_decay=bn_decay)
    pred_labels_corner_p = tf_util.conv1d(feat1_1, 2, 1, padding='VALID', activation_fn=None, scope='stage1/task1_1/fc2', bn_decay=bn_decay)

    #task_3: position
    feat3 = tf_util.conv1d(dof_feat,128,1,padding='VALID',activation_fn = None,scope = 'stage1/task3/fc1', bn_decay=bn_decay)
    pred_regression_position = tf_util.conv1d(feat3, 3, 1, padding='VALID', activation_fn=None, scope='stage1/task3/fc2', bn_decay=bn_decay)

    return pred_labels_key_p,pred_labels_corner_p,pred_regression_position

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

    #loss:task3
    task_3_loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_mean(smooth_l1_dist(pred_regression_position-regression_position),axis=2)*mask, \
                               axis = 1)/tf.reduce_sum(mask,axis=1))

    w1 = 1
    w1_1 = 1
    w3 = 100

    loss = task_1_loss*w1 + task_1_1_loss*w1_1 + task_3_loss*w3

    tf.summary.scalar('all loss', loss)
    tf.add_to_collection('losses', loss)

    return task_1_loss,task_1_recall,task_1_acc,task_1_1_loss,task_1_1_recall,task_1_1_acc,task_3_loss,loss
