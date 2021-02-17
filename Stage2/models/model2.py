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

def get_feature(point_cloud, is_training,bn_decay=None):
    """ Part segmentation PointNet, input is BxNx6 (XYZ NormalX NormalY NormalZ), output Bx50 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = tf.slice(point_cloud, [0,0,0], [-1,-1,3])
    l0_points = tf.slice(point_cloud, [0,0,3], [-1,-1,3])

    # Set Abstraction layers
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=512, radius=0.2, nsample=64, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='pointnet/layer1')
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=128, radius=0.4, nsample=64, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='pointnet/layer2')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='pointnet/layer3')

    # Feature Propagation layers
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256,256], is_training, bn_decay, scope='pointnet/fa_layer1')
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,128], is_training, bn_decay, scope='pointnet/fa_layer2')
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, tf.concat([l0_xyz,l0_points],axis=-1), l1_points, [128,128,128], is_training, bn_decay, scope='pointnet/fa_layer3')

    # FC layers
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='pointnet/fc1', bn_decay=bn_decay)

    end_points['feats'] = net 
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='pointnet/dp1')

    feat = tf_util.conv1d(net, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='pointnet/fc2', bn_decay=bn_decay)
    return end_points,feat

def placeholder_inputs_stage_3(batch_size,num_point):
    pointclouds_pl = tf.placeholder(tf.float32,shape=(batch_size,num_point,6))
    proposal_pl = tf.placeholder(tf.int32,shape=(batch_size,num_point))
    dof_regression_pl = tf.placeholder(tf.float32,shape=(batch_size,1,6))
    field_pl = tf.placeholder(tf.float32,shape=(batch_size,3,num_point,6))
    return pointclouds_pl,proposal_pl,dof_regression_pl,field_pl

def get_stage_3(pointclouds_pl,field_pl,is_training,bn_decay=None):
    batch_size = pointclouds_pl.get_shape()[0].value
    scope = 'point'
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as sc:
        end_points1,feat1 = get_feature(pointclouds_pl,is_training,bn_decay)
    field1 = field_pl[:,0,:,:]
    field2 = field_pl[:,1,:,:]
    field3 = field_pl[:,2,:,:]
    scope = 'field'
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as sc:
        end_points2_1,feat2_1 = get_feature(field1,is_training,bn_decay)

    scope = 'field'
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as sc:
        end_points2_2,feat2_2 = get_feature(field2,is_training,bn_decay)

    scope = 'field'
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as sc:
        end_points2_3,feat2_3 = get_feature(field3,is_training,bn_decay)
    
    feat2_1 = tf.expand_dims(feat2_1,axis = 1)
    feat2_2 = tf.expand_dims(feat2_2,axis = 1)
    feat2_3 = tf.expand_dims(feat2_3,axis = 1)
    feat2 = tf.concat([feat2_1,feat2_2,feat2_3],axis=1)
    feat1 = tf.expand_dims(feat1,axis=1)
    feat = tf.concat([feat1,feat2],axis=1)
    feat = tf.reduce_max(feat,axis = 1)
    feat_proposal = tf_util.conv1d(feat, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='stage3/fc1_1', bn_decay=bn_decay)
    pred_proposal = tf_util.conv1d(feat_proposal, 2, 1, padding='VALID', bn=True, is_training=is_training, scope='stage3/fc1_2', bn_decay=bn_decay)
    feat_dof = tf_util.conv1d(feat, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='stage3/fc2_1', bn_decay=bn_decay)
    pred_dof_regression = tf_util.conv1d(feat_dof, 6, 4096, padding='VALID', bn=True, is_training=is_training, scope='stage3/fc2_2', bn_decay=bn_decay) 
    return pred_proposal,pred_dof_regression

def get_stage_3_loss(pred_proposal,pred_dof_regression,proposal_pl,dof_regression_pl):
    num_point = pred_proposal.get_shape()[1].value
    loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = pred_proposal,labels = proposal_pl))
    loss2 = tf.reduce_mean(tf.reduce_sum(smooth_l1_dist(pred_dof_regression-dof_regression_pl),axis=2))
    task_1_acc = tf.reduce_mean(tf.reduce_sum(tf.cast(tf.equal(tf.argmax(pred_proposal,axis=2,output_type = tf.int32),\
                          proposal_pl),tf.float32),axis = 1)/num_point)
    proposal = tf.argmax(pred_proposal,axis=2,output_type = tf.int32)
    proposal = tf.greater(tf.cast(proposal,tf.float32),0.5)
    proposal_pl = tf.greater(tf.cast(proposal_pl,tf.float32),0.5)
    epsilon = tf.constant(np.ones(proposal_pl.get_shape()[:1]).astype(np.float32) *1e-6)
    iou = tf.reduce_sum(tf.cast(tf.logical_and(proposal, proposal_pl), tf.float32), axis=-1) / \
                      (tf.reduce_sum(tf.cast(tf.logical_or(proposal, proposal_pl), tf.float32), axis=-1) + epsilon)
    iou = tf.reduce_mean(iou)
    loss = loss1+loss2
    return task_1_acc,iou,loss1,loss2,loss

