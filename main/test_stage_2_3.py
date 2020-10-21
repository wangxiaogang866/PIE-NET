import argparse
import math
from datetime import datetime
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
import time
from datetime import datetime
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import tf_util
import scipy.io as sio
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1, help='GPU to use [default: GPU 0]')
parser.add_argument('--model_2', default='model', help='Model name [default: model]')
parser.add_argument('--stage_1_log_dir', default='stage_1_log', help='Log dir [default: log]')
parser.add_argument('--stage_2_log_dir', default='stage_2_log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=8096, help='Point Number [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=1, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--stage',type=int,default=2,help='network stage')

parser.add_argument('--num_open_pair', type=int, default=256, help='Point Number [default: 256]')
parser.add_argument('--num_point_per_open_pair', type=int, default=64, help='Point Number [default: 64]')
parser.add_argument('--num_open_gt_sample', type=int, default=64, help='Point Number [default: 64]')

parser.add_argument('--num_closed_point', type=int, default=256, help='Point Number [default: 256]')
parser.add_argument('--num_point_per_closed_point', type=int, default=64, help='Point Number [default: 64]')
parser.add_argument('--num_closed_gt_sample', type=int, default=64, help='Point Number [default: 64]')

FLAGS = parser.parse_args()

EPOCH_CNT = 0
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
STAGE = FLAGS.stage

NUM_OPEN_PAIR = FLAGS.num_open_pair
NUM_POINT_PER_OPEN_PAIR = FLAGS.num_point_per_open_pair
NUM_OPEN_GT_SAMPLE = FLAGS.num_open_gt_sample

NUM_CLOSED_POINT = FLAGS.num_closed_point
NUM_POINT_PER_CLOSED_POINT = FLAGS.num_point_per_closed_point
NUM_CLOSED_GT_SAMPLE = FLAGS.num_closed_gt_sample

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')
if STAGE == 1:
    LOG_DIR = FLAGS.stage_1_log_dir
else:
    LOG_DIR = FLAGS.stage_2_log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_learning_rate_stage_2(batch,base_learning_rate):
    learning_rate = tf.train.exponential_decay(
                        base_learning_rate,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            if STAGE==1:
                print('stage_1')
                pointclouds_pl,labels_key_p,labels_corner_p, \
                   open_gt_256_64_idx, open_gt_mask, open_gt_type, open_gt_res, open_gt_sample_points, open_gt_valid_mask, open_gt_pair_idx, \
                   closed_gt_256_64_idx, closed_gt_mask, closed_gt_type, closed_gt_res, closed_gt_sample_points, \
                   closed_gt_valid_mask, closed_gt_pair_idx = MODEL.placeholder_inputs_stage_1(BATCH_SIZE,NUM_POINT, \
                                                                        NUM_OPEN_PAIR,NUM_POINT_PER_OPEN_PAIR,NUM_OPEN_GT_SAMPLE, \
                                                                        NUM_CLOSED_POINT, NUM_POINT_PER_CLOSED_POINT,NUM_CLOSED_GT_SAMPLE)
                is_training_pl = tf.placeholder(tf.bool, shape=())
                batch_stage_1 = tf.Variable(0,name='stage1/batch')
                bn_decay = get_bn_decay(batch_stage_1)
                tf.summary.scalar('bn_decay', bn_decay)
                print("--- Get model and loss")
                # Get model and loss 
                end_points,dof_feat,simmat_feat = MODEL.get_feature(pointclouds_pl, is_training_pl,STAGE,bn_decay=bn_decay)
                print("--- Get model and loss1")
                pred_labels_key_p,pred_labels_corner_p, \
                    open_pre_mask, open_pre_class_logits, open_pre_res, \
                    closed_pre_type, closed_pre_mask, closed_pre_class_logits,closed_pre_res, \
                    open_pre_sample_points, closed_pre_sample_points, \
                    open_ball_radius, open_ball_center, \
                    closed_ball_radius, closed_ball_center, \
                    open_cycle_curve_pre, open_b_spline_curve_pre, open_line_curve_pre, \
                    closed_para_cycle = MODEL.get_stage_1(pointclouds_pl, dof_feat,simmat_feat, open_gt_256_64_idx, closed_gt_256_64_idx, closed_gt_pair_idx, \
                    open_gt_mask, closed_gt_mask, is_training_pl,bn_decay=bn_decay)
                print("--- Get model and loss2")
                task_1_loss,task_1_recall,task_1_acc,task_1_1_loss,task_1_1_recall,task_1_1_acc, \
                    open_class_loss,open_res_loss,open_mask_loss, open_restrcut_loss, \
                    closed_class_loss,closed_res_loss,closed_mask_loss,closed_restruct_loss, loss, \
                    open_class_acc, open_mask_acc, closed_class_acc, closed_mask_acc = MODEL.get_stage_1_loss(pred_labels_key_p,pred_labels_corner_p, labels_key_p,labels_corner_p,
                                        open_pre_mask, open_pre_class_logits, open_pre_res, open_pre_sample_points, open_gt_mask, open_gt_type, open_gt_res, open_gt_sample_points, open_gt_valid_mask, \
                                        closed_pre_type, closed_pre_mask, closed_pre_class_logits, closed_pre_res, closed_pre_sample_points, \
                                        closed_gt_mask, closed_gt_type, closed_gt_res, closed_gt_sample_points, closed_gt_valid_mask, \
                                        open_ball_radius, open_ball_center, \
                                        closed_ball_radius, closed_ball_center)
                tf.summary.scalar('labels_key_p_loss', task_1_loss)
                tf.summary.scalar('labels_key_p_recall', task_1_recall)
                tf.summary.scalar('labels_key_p_acc', task_1_acc)                
                tf.summary.scalar('labels_corner_p_loss', task_1_1_loss)
                tf.summary.scalar('labels_corner_p_recall', task_1_1_recall)
                tf.summary.scalar('labels_corner_p_acc', task_1_1_acc)
                
                tf.summary.scalar('open_class_loss', open_class_loss)
                tf.summary.scalar('open_res_loss', open_res_loss)
                tf.summary.scalar('open_mask_loss', open_mask_loss)
                
                tf.summary.scalar('closed_class_loss', closed_class_loss)
                tf.summary.scalar('closed_res_loss', closed_res_loss)                
                tf.summary.scalar('closed_mask_loss', closed_mask_loss)
                
                tf.summary.scalar('open_class_acc', open_class_acc)
                tf.summary.scalar('open_mask_acc', open_mask_acc)                
                tf.summary.scalar('closed_class_acc', closed_class_acc)
                tf.summary.scalar('closed_mask_acc', closed_mask_acc)
                
                tf.summary.scalar('open_restrcut_loss', open_restrcut_loss)
                tf.summary.scalar('closed_restruct_loss', closed_restruct_loss)
                
                tf.summary.scalar('loss', loss)

                print("--- Get training operator")
                # Get training operator
                learning_rate = get_learning_rate(batch_stage_1)
                tf.summary.scalar('learning_rate', learning_rate)
                if OPTIMIZER == 'momentum':
                    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
                elif OPTIMIZER == 'adam':
                    optimizer = tf.train.AdamOptimizer(learning_rate)
                train_op = optimizer.minimize(loss, global_step=batch_stage_1)
            
                # Add ops to save and restore all the variables.
                saver = tf.train.Saver(max_to_keep=10)
            
                
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)
        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        model_path = './'+LOG_DIR + '/model100.ckpt'
        if STAGE == 1:
            #init = tf.global_variables_initializer()
            #sess.run(init)
            saver.restore(sess,model_path)
        else:
            #init = tf.global_variables_initializer()
            #sess.run(init)
            saver.restore(sess,model_path)
    if STAGE==1:
            ops = {'pointclouds_pl': pointclouds_pl,
               'labels_key_p': labels_key_p,
               'labels_corner_p': labels_corner_p,               
               'open_gt_256_64_idx': open_gt_256_64_idx,
               'open_gt_mask': open_gt_mask,
               'open_gt_type': open_gt_type,
               'open_gt_res': open_gt_res,
               'open_gt_sample_points': open_gt_sample_points,
               'open_gt_valid_mask': open_gt_valid_mask,
               'open_gt_pair_idx': open_gt_pair_idx,               
               'closed_gt_256_64_idx': closed_gt_256_64_idx,
               'closed_gt_mask': closed_gt_mask,
               'closed_gt_type': closed_gt_type,
               'closed_gt_res': closed_gt_res,
               'closed_gt_sample_points': closed_gt_sample_points,
               'closed_gt_valid_mask': closed_gt_valid_mask,
               'closed_gt_pair_idx': closed_gt_pair_idx,               
               'is_training_pl': is_training_pl,
               'pred_labels_key_p': pred_labels_key_p,
               'pred_labels_corner_p': pred_labels_corner_p,                
               'open_pre_mask': open_pre_mask,
               'open_pre_class_logits': open_pre_class_logits,
               'open_pre_res': open_pre_res,              
               'closed_pre_type': closed_pre_type,
               'closed_pre_mask': closed_pre_mask,
               'closed_pre_class_logits': closed_pre_class_logits,
               'closed_pre_res': closed_pre_res,                 
               'task_1_loss': task_1_loss,
               'task_1_recall':task_1_recall,
               'task_1_acc': task_1_acc,               
               'task_1_1_loss': task_1_1_loss,
               'task_1_1_recall':task_1_1_recall,
               'task_1_1_acc': task_1_1_acc,               
               'open_class_loss': open_class_loss,
               'open_res_loss': open_res_loss,
               'open_mask_loss': open_mask_loss,
               'closed_class_loss': closed_class_loss,
               'closed_res_loss': closed_res_loss,
               'closed_mask_loss': closed_mask_loss,
               'open_class_acc': open_class_acc,
               'open_mask_acc': open_mask_acc,
               'closed_class_acc': closed_class_acc,
               'closed_mask_acc': closed_mask_acc,
               # radius, center
               'open_ball_center': open_ball_center,
               'open_ball_radius': open_ball_radius,
               'closed_ball_center': closed_ball_center,
               'closed_ball_radius': closed_ball_radius,
               # pre_points
               'open_pre_sample_points': open_pre_sample_points,
               'closed_pre_sample_points': closed_pre_sample_points,
               # pre_para
               'open_cycle_curve_pre': open_cycle_curve_pre,
               'open_b_spline_curve_pre': open_b_spline_curve_pre,
               'open_line_curve_pre': open_line_curve_pre,
               'closed_para_cycle': closed_para_cycle,
               # restruct loss
               'open_restrcut_loss': open_restrcut_loss,
               'closed_restruct_loss': closed_restruct_loss,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch_stage_1,
               'end_points': end_points}
            for epoch in range(MAX_EPOCH):
                log_string('**** TEST EPOCH %03d ****' % (epoch))
                sys.stdout.flush()
                eval_one_epoch_stage_1(sess,ops,test_writer)



def eval_one_epoch_stage_1(sess, ops, train_writer):
    is_training = True
    dataset = ['/media/user_c/disk1/PC2EDGE/TEST1_DATa/35.mat','/media/user_c/disk1/PC2EDGE/TEST1_DATa/36.mat',
               '/media/user_c/disk1/PC2EDGE/TEST1_DATa/37.mat','/media/user_c/disk1/PC2EDGE/TEST1_DATa/38.mat']
    for i in range(len(dataset)):
        load_data_start_time = time.time();
        train_data = sio.loadmat(dataset[i])['Training_data']
        load_data_duration = time.time() - load_data_start_time
        log_string('\t%s: %s load time: %f' % (datetime.now(),dataset[i],load_data_duration))
        num_data = train_data.shape[0]
        num_batch = num_data // BATCH_SIZE
        total_loss = 0.0
        total_task_1_loss = 0.0
        total_task_1_acc = 0.0
        total_task_1_recall = 0.0
        total_task_1_1_loss = 0.0
        total_task_1_1_acc = 0.0
        total_task_1_1_recall = 0.0
        total_open_class_loss = 0.0
        total_open_res_loss = 0.0
        total_open_mask_loss = 0.0
        total_closed_class_loss = 0.0
        total_closed_res_loss = 0.0
        total_closed_mask_loss = 0.0
        total_open_class_acc = 0.0
        total_open_mask_acc = 0.0
        total_closed_class_acc = 0.0
        total_closed_mask_acc = 0.0
        # reconstruct loss
        total_open_restrcut_loss = 0.0
        total_closed_restruct_loss = 0.0
        process_start_time = time.time()
        pred_labels_key_p_val = np.zeros((num_data, NUM_POINT, 2), np.float32)
        pred_labels_corner_p_val = np.zeros((num_data, NUM_POINT, 2), np.float32)
        # pre segmentation, classification, residual
        open_pre_mask_val = np.zeros((num_data, NUM_OPEN_PAIR, NUM_POINT_PER_OPEN_PAIR, 2), np.float32)
        open_pre_class_logits_val = np.zeros((num_data, NUM_OPEN_PAIR, 4), np.float32)
        open_pre_res_val = np.zeros((num_data, NUM_OPEN_PAIR,4, 3*2), np.float32)
        closed_pre_type_val = np.zeros((num_data, NUM_CLOSED_POINT, 2), np.float32)
        closed_pre_mask_val = np.zeros((num_data, NUM_CLOSED_POINT, NUM_POINT_PER_CLOSED_POINT, 2), np.float32)
        closed_pre_class_logits_val = np.zeros((num_data, NUM_CLOSED_POINT, 2), np.float32)
        closed_pre_res_val = np.zeros((num_data, NUM_CLOSED_POINT,2, 3*2), np.float32)
        # radius, center 
        open_ball_center_val = np.zeros((num_data, NUM_OPEN_PAIR, 1, 3), np.float32)
        open_ball_radius_val =np.zeros((num_data, NUM_OPEN_PAIR, 1, 1), np.float32)
        closed_ball_center_val = np.zeros((num_data, NUM_CLOSED_POINT, 1, 3), np.float32)
        closed_ball_radius_val = np.zeros((num_data, NUM_CLOSED_POINT, 1, 1), np.float32)
        # pre_points
        open_pre_sample_points_val = np.zeros((num_data, NUM_OPEN_PAIR, 4, 64, 3), np.float32)
        closed_pre_sample_points_val = np.zeros((num_data, NUM_CLOSED_POINT, 2, 64, 3), np.float32)
        # pre_para
        open_cycle_curve_pre_val = np.zeros((num_data, NUM_OPEN_PAIR, 3+3+1), np.float32)
        open_b_spline_curve_pre_val = np.zeros((num_data, NUM_OPEN_PAIR, 4*3), np.float32)
        open_line_curve_pre_val= np.zeros((num_data, NUM_OPEN_PAIR, 3*2), np.float32)
        closed_para_cycle_val = np.zeros((num_data, NUM_CLOSED_POINT, 3+3+1), np.float32)

        input_data = np.zeros((num_data, NUM_POINT, 3), np.float32)
        input_labels_key_p = np.zeros((num_data,NUM_POINT),np.int32)
        input_labels_corner_p = np.zeros((num_data,NUM_POINT),np.int32)
        for x in range(num_data):
            tmp_data_1 = train_data[x,0]
            input_data[x,:,:] = tmp_data_1['down_sample_point'][0,0]
            input_labels_key_p[x,:] = np.squeeze(tmp_data_1['PC_8096_edge_points_label_bin'][0,0])
            input_labels_corner_p[x,:] = np.squeeze(tmp_data_1['corner_points_label'][0,0])
            
        for j in range(num_batch):
            begin_idx = j*BATCH_SIZE
            end_idx = (j+1)*BATCH_SIZE
            data_cells = train_data[begin_idx: end_idx,0]
            batch_inputs = np.zeros((BATCH_SIZE,NUM_POINT,3),np.float32)
            batch_labels_key_p = np.zeros((BATCH_SIZE,NUM_POINT),np.int32)
            batch_labels_corner_p = np.zeros((BATCH_SIZE,NUM_POINT),np.int32)
            
            batch_open_gt_256_64_idx = np.zeros((BATCH_SIZE,NUM_OPEN_PAIR,NUM_POINT_PER_OPEN_PAIR),np.int32)
            batch_open_gt_mask = np.zeros((BATCH_SIZE,NUM_OPEN_PAIR,NUM_POINT_PER_OPEN_PAIR),np.int32)
            batch_open_gt_type = np.zeros((BATCH_SIZE,NUM_OPEN_PAIR),np.int32)
            batch_open_gt_res = np.zeros((BATCH_SIZE,NUM_OPEN_PAIR, 3*2),np.float32)
            batch_open_gt_sample_points = np.zeros((BATCH_SIZE, NUM_OPEN_PAIR, NUM_OPEN_GT_SAMPLE,3), np.float32)
            batch_open_gt_valid_mask = np.zeros((BATCH_SIZE, NUM_OPEN_PAIR), np.int32)
            batch_open_gt_pair_idx = np.zeros((BATCH_SIZE, NUM_OPEN_PAIR, 2), np.int32)

            batch_closed_gt_256_64_idx = np.zeros((BATCH_SIZE,NUM_CLOSED_POINT,NUM_POINT_PER_CLOSED_POINT),np.int32)
            batch_closed_gt_mask = np.zeros((BATCH_SIZE,NUM_CLOSED_POINT,NUM_POINT_PER_CLOSED_POINT),np.int32)
            batch_closed_gt_type = np.zeros((BATCH_SIZE,NUM_CLOSED_POINT),np.int32)
            batch_closed_gt_res = np.zeros((BATCH_SIZE,NUM_CLOSED_POINT, 3),np.float32)
            batch_closed_gt_sample_points = np.zeros((BATCH_SIZE, NUM_CLOSED_POINT, NUM_CLOSED_GT_SAMPLE,3), np.float32)
            batch_closed_gt_valid_mask = np.zeros((BATCH_SIZE, NUM_CLOSED_POINT), np.int32)
            batch_closed_gt_pair_idx = np.zeros((BATCH_SIZE, NUM_CLOSED_POINT, 1), np.int32) 
            for cnt in range(BATCH_SIZE):
                tmp_data = data_cells[cnt]
                batch_inputs[cnt,:,:] = tmp_data['down_sample_point'][0, 0]
                batch_labels_key_p[cnt,:] = np.squeeze(tmp_data['PC_8096_edge_points_label_bin'][0,0])
                batch_labels_corner_p[cnt,:] = np.squeeze(tmp_data['corner_points_label'][0,0])
                
                batch_open_gt_256_64_idx[cnt,:,:] = tmp_data['open_gt_256_64_idx'][0,0]
                batch_open_gt_mask[cnt,:,:] = tmp_data['open_gt_mask'][0,0]
                batch_open_gt_type[cnt,:] = np.squeeze(tmp_data['open_gt_type'][0,0])
                batch_open_gt_res[cnt,:,:] = tmp_data['open_gt_res'][0,0]
                batch_open_gt_sample_points[cnt,:,:] = tmp_data['open_gt_sample_points'][0,0]
                batch_open_gt_valid_mask[cnt,:] = np.squeeze(tmp_data['open_gt_valid_mask'][0,0])
                batch_open_gt_pair_idx[cnt,:,:] = tmp_data['open_gt_pair_idx'][0,0]
                
                batch_closed_gt_256_64_idx[cnt,:,:] = tmp_data['closed_gt_256_64_idx'][0,0]
                batch_closed_gt_mask[cnt,:,:] = tmp_data['closed_gt_mask'][0,0]
                batch_closed_gt_type[cnt,:] = np.squeeze(tmp_data['closed_gt_type'][0,0])
                batch_closed_gt_res[cnt,:,:] = tmp_data['closed_gt_res'][0,0]
                batch_closed_gt_sample_points[cnt,:,:] = tmp_data['closed_gt_sample_points'][0,0]
                batch_closed_gt_valid_mask[cnt,:] = np.squeeze(tmp_data['closed_gt_valid_mask'][0,0])
                batch_closed_gt_pair_idx[cnt,:,:] = tmp_data['closed_gt_pair_idx'][0,0]

            feed_dict = {ops['pointclouds_pl']: batch_inputs,
                         ops['labels_key_p']: batch_labels_key_p,
                         ops['labels_corner_p']: batch_labels_corner_p,
                         ops['open_gt_256_64_idx']: batch_open_gt_256_64_idx,
                         ops['open_gt_mask']: batch_open_gt_mask,
                         ops['open_gt_type']: batch_open_gt_type,
                         ops['open_gt_res']: batch_open_gt_res,
                         ops['open_gt_sample_points']: batch_open_gt_sample_points,
                         ops['open_gt_valid_mask']: batch_open_gt_valid_mask,
                         ops['open_gt_pair_idx']: batch_open_gt_pair_idx,                         
                         ops['closed_gt_256_64_idx']: batch_closed_gt_256_64_idx,
                         ops['closed_gt_mask']: batch_closed_gt_mask,
                         ops['closed_gt_type']: batch_closed_gt_type,
                         ops['closed_gt_res']: batch_closed_gt_res,
                         ops['closed_gt_sample_points']: batch_closed_gt_sample_points,
                         ops['closed_gt_valid_mask']: batch_closed_gt_valid_mask,
                         ops['closed_gt_pair_idx']: batch_closed_gt_pair_idx,
                         ops['is_training_pl']: is_training}                 
    
    
            summary, step, _, task_1_loss_val, task_1_recall_val, task_1_acc_val, task_1_1_loss_val, task_1_1_recall_val, task_1_1_acc_val, \
                                 open_class_loss_val, open_res_loss_val, open_mask_loss_val, open_restrcut_loss_val, \
                                 closed_class_loss_val, closed_res_loss_val, closed_mask_loss_val, closed_restruct_loss_val, loss_val, \
                                 open_class_acc_val, open_mask_acc_val, closed_class_acc_val, closed_mask_acc_val, \
                                 pred_labels_key_p_val[begin_idx:end_idx,:,:], \
                                 pred_labels_corner_p_val[begin_idx:end_idx,:,:], \
                                 open_pre_mask_val[begin_idx:end_idx,:,:,:], \
                                 open_pre_class_logits_val[begin_idx:end_idx,:,:],  \
                                 open_pre_res_val[begin_idx:end_idx,:,:],  \
                                 closed_pre_type_val[begin_idx:end_idx,:,:],  \
                                 closed_pre_mask_val[begin_idx:end_idx,:,:,:],  \
                                 closed_pre_class_logits_val[begin_idx:end_idx,:,:],  \
                                 closed_pre_res_val[begin_idx:end_idx,:,:],  \
                                 open_ball_center_val[begin_idx:end_idx,:,:],  \
                                 open_ball_radius_val[begin_idx:end_idx,:],  \
                                 closed_ball_center_val[begin_idx:end_idx,:,:],  \
                                 closed_ball_radius_val[begin_idx:end_idx,:],  \
                                 open_pre_sample_points_val[begin_idx:end_idx,:,:,:],  \
                                 closed_pre_sample_points_val[begin_idx:end_idx,:,:,:],  \
                                 open_cycle_curve_pre_val[begin_idx:end_idx,:,:],  \
                                 open_b_spline_curve_pre_val[begin_idx:end_idx,:,:],  \
                                 open_line_curve_pre_val[begin_idx:end_idx,:,:], \
                                 closed_para_cycle_val[begin_idx:end_idx,:,:], \
                                 = sess.run( [ops['merged'], ops['step'], \
                                 ops['train_op'], ops['task_1_loss'], ops['task_1_recall'], ops['task_1_acc'], ops['task_1_1_loss'], ops['task_1_1_recall'], ops['task_1_1_acc'], \
                                 ops['open_class_loss'], ops['open_res_loss'], ops['open_mask_loss'], ops['open_restrcut_loss'], \
                                 ops['closed_class_loss'], ops['closed_res_loss'], ops['closed_mask_loss'], ops['closed_restruct_loss'], ops['loss'], \
                                 ops['open_class_acc'], ops['open_mask_acc'], ops['closed_class_acc'], ops['closed_mask_acc'], \
                                 ops['pred_labels_key_p'], \
                                 ops['pred_labels_corner_p'], \
                                 ops['open_pre_mask'], \
                                 ops['open_pre_class_logits'], \
                                 ops['open_pre_res'], \
                                 ops['closed_pre_type'], \
                                 ops['closed_pre_mask'], \
                                 ops['closed_pre_class_logits'], \
                                 ops['closed_pre_res'], \
                                 # radius, center
                                 ops['open_ball_center'], \
                                 ops['open_ball_radius'], \
                                 ops['closed_ball_center'], \
                                 ops['closed_ball_radius'], \
                                 # pre_points
                                 ops['open_pre_sample_points'], \
                                 ops['closed_pre_sample_points'], \
                                 # pre_para
                                 ops['open_cycle_curve_pre'], \
                                 ops['open_b_spline_curve_pre'], \
                                 ops['open_line_curve_pre'], \
                                 ops['closed_para_cycle']], \
                                 feed_dict=feed_dict) 
            
            train_writer.add_summary(summary, step)
            total_loss += loss_val
            total_task_1_loss += task_1_loss_val
            total_task_1_acc += task_1_acc_val
            total_task_1_recall += task_1_recall_val
            total_task_1_1_loss += task_1_1_loss_val
            total_task_1_1_acc += task_1_1_acc_val
            total_task_1_1_recall += task_1_1_recall_val             
            total_open_class_loss += open_class_loss_val
            total_open_res_loss += open_res_loss_val
            total_open_mask_loss += open_mask_loss_val
            total_closed_class_loss += closed_class_loss_val
            total_closed_res_loss += closed_res_loss_val
            total_closed_mask_loss += closed_mask_loss_val
            total_open_class_acc += open_class_acc_val
            total_open_mask_acc += open_mask_acc_val
            total_closed_class_acc += closed_class_acc_val
            total_closed_mask_acc += closed_mask_acc_val
            # reconstruct loss
            total_open_restrcut_loss += open_restrcut_loss_val
            total_closed_restruct_loss += closed_restruct_loss_val
            #print('loss: %f' % loss_val)
        total_loss = total_loss * 1.0 / num_batch
        total_task_1_loss = total_task_1_loss * 1.0 / num_batch
        total_task_1_acc = total_task_1_acc * 1.0 / num_batch
        total_task_1_recall = total_task_1_recall * 1.0 / num_batch
        total_task_1_1_loss = total_task_1_1_loss * 1.0 / num_batch
        total_task_1_1_acc = total_task_1_1_acc * 1.0 / num_batch
        total_task_1_1_recall = total_task_1_1_recall * 1.0 / num_batch          
        total_open_class_loss = total_open_class_loss * 1.0 / num_batch
        total_open_res_loss = total_open_res_loss * 1.0 / num_batch
        total_open_mask_loss = total_open_mask_loss * 1.0 / num_batch
        total_closed_class_loss = total_closed_class_loss * 1.0 / num_batch
        total_closed_res_loss = total_closed_res_loss * 1.0 / num_batch
        total_closed_mask_loss = total_closed_mask_loss * 1.0 / num_batch
        total_open_class_acc = total_open_class_acc * 1.0 / num_batch
        total_open_mask_acc = total_open_mask_acc * 1.0 / num_batch
        total_closed_class_acc = total_closed_class_acc * 1.0 / num_batch
        total_closed_mask_acc = total_closed_mask_acc * 1.0 / num_batch
        # reconstruct loss
        total_open_restrcut_loss = total_open_restrcut_loss * 1.0 / num_batch
        total_closed_restruct_loss = total_closed_restruct_loss * 1.0 / num_batch        
        process_duration = time.time() - process_start_time
        examples_per_sec = num_data/process_duration
        sec_per_batch = process_duration/num_batch
        log_string('\t%s: step: %f loss: %f duration time %.3f (%.1f examples/sec; %.3f sec/batch)' \
           % (datetime.now(),step,total_loss,process_duration,examples_per_sec,sec_per_batch))
        log_string('\t\tTraining TASK 1 Mean_loss: %f' % total_task_1_loss)
        log_string('\t\tTraining TASK 1 Accuracy: %f' % total_task_1_acc)
        log_string('\t\tTraining TASK 1 Recall: %f' % total_task_1_recall)
        log_string('\t\tTraining TASK 1 Mean_loss: %f' % total_task_1_1_loss)
        log_string('\t\tTraining TASK 1 Accuracy: %f' % total_task_1_1_acc)
        log_string('\t\tTraining TASK 1 Recall: %f' % total_task_1_1_recall)         
        log_string('\t\tTraining open_class Mean_loss: %f' % total_open_class_loss)
        log_string('\t\tTraining open_class Accuracy: %f' % total_open_class_acc)        
        log_string('\t\tTraining open_res Mean_loss: %f' % total_open_res_loss)        
        log_string('\t\tTraining open_mask Mean_loss: %f' % total_open_mask_loss)
        log_string('\t\tTraining open_mask Accuracy: %f' % total_open_mask_acc)        
        log_string('\t\tTraining closed_class Mean_loss: %f' % total_closed_class_loss)
        log_string('\t\tTraining closed_class Accuracy: %f' % total_closed_class_acc)        
        log_string('\t\tTraining closed_res Mean_loss: %f' % total_closed_res_loss)
        log_string('\t\tTraining closed_mask Mean_loss: %f' % total_closed_mask_loss)
        log_string('\t\tTraining closed_mask Accuracy: %f' % total_closed_mask_acc)
        # reconstruct loss
        log_string('\t\tTraining open_restrcut_loss: %f' % total_open_restrcut_loss)
        log_string('\t\tTraining closed_restruct_loss: %f' % total_closed_restruct_loss)
        temp_name = dataset[i]   # name = '1.mat'
        temp_name = dataset[i]
        temp_name = temp_name[39:]
        sio.savemat('./test_data/test_pred_'+temp_name, {'input_point_cloud': input_data, \
                                                    'input_labels_key_p': input_labels_key_p, \
                                                    'input_labels_corner_p': input_labels_corner_p, \
                                                    'pred_labels_key_p_val': pred_labels_key_p_val, \
                                                    'pred_labels_corner_p_val': pred_labels_corner_p_val, \
                                                    # pre segmentation, classification, residual
                                                    'open_pre_mask_val': open_pre_mask_val, \
                                                    'open_pre_class_logits_val': open_pre_class_logits_val, \
                                                    'open_pre_res_val': open_pre_res_val, \
                                                    'closed_pre_type_val': closed_pre_type_val, \
                                                    'closed_pre_mask_val': closed_pre_mask_val, \
                                                    'closed_pre_class_logits_val': closed_pre_class_logits_val, \
                                                    'closed_pre_res_val': closed_pre_res_val, \
                                                    # radius, center 
                                                    'open_ball_center_val': open_ball_center_val, \
                                                    'open_ball_radius_val': open_ball_radius_val, \
                                                    'closed_ball_center_val': closed_ball_center_val, \
                                                    'closed_ball_radius_val': closed_ball_radius_val, \
                                                    # pre_points
                                                    'open_pre_sample_points_val': open_pre_sample_points_val, \
                                                    'closed_pre_sample_points_val': closed_pre_sample_points_val, \
                                                    # pre_para
                                                    'open_cycle_curve_pre_val': open_cycle_curve_pre_val, \
                                                    'open_b_spline_curve_pre_val': open_b_spline_curve_pre_val, \
                                                    'open_line_curve_pre_val': open_line_curve_pre_val,  \
                                                    'closed_para_cycle_val': closed_para_cycle_val})


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
