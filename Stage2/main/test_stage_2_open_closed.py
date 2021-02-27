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
parser.add_argument('--gpu', type=int, default=2, help='GPU to use [default: GPU 0]')    # change gpu device number
parser.add_argument('--model', default='model', help='Model name [default: model]')
parser.add_argument('--stage_1_log_dir', default='stage_1_log', help='Log dir [default: log]')
parser.add_argument('--stage_2_log_dir', default='stage_2_log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=8096, help='Point Number [default: 2048]')
parser.add_argument('--num_roi', type=int, default=8128, help='Roi Number [default: 128]')
parser.add_argument('--max_epoch', type=int, default=2, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--stage',type=int,default=2,help='network stage')
FLAGS = parser.parse_args()

EPOCH_CNT = 0
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
NUM_ROI = FLAGS.num_roi
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
STAGE = FLAGS.stage

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
                pointclouds_pl, corner_pair, label_1, label_2, label_5, label_6,label_7,label_8 = MODEL.placeholder_inputs_stage_1(BATCH_SIZE,NUM_POINT,NUM_ROI)
                is_training_pl = tf.placeholder(tf.bool, shape=())
                # Note the global_step=batch parameter to minimize. 
                # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
                batch_stage_1 = tf.Variable(0,name='stage1/batch')
                bn_decay = get_bn_decay(batch_stage_1)
                tf.summary.scalar('bn_decay', bn_decay)
                print("--- Get model and loss")
                # Get model and loss 
                end_points,pc_fea_cropped,pc_coord_cropped,point_cloud,dof_feat  = MODEL.get_feature(pointclouds_pl, corner_pair, is_training_pl,STAGE,bn_decay=bn_decay)
                pre_label_1, pre_label_2, \
  		pre_label_5, pre_label_6, pre_label_7, pre_label_8 = MODEL.get_stage_1(pc_coord_cropped, pc_fea_cropped, dof_feat, is_training_pl,bn_decay=bn_decay)						

                print("--- Get training operator")
                # Get training operator
                learning_rate = get_learning_rate(batch_stage_1)
                tf.summary.scalar('learning_rate', learning_rate)
                if OPTIMIZER == 'momentum':
                    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
                elif OPTIMIZER == 'adam':
                    optimizer = tf.train.AdamOptimizer(learning_rate)
                #train_op = optimizer.minimize(loss, global_step=batch_stage_1)
            
                # Add ops to save and restore all the variables.
                saver = tf.train.Saver(max_to_keep=10)
            elif STAGE==2:
                print('stage_2')
                pointclouds_pl,proposal_nx_pl,dof_mask_pl,dof_score_pl= MODEL.placeholder_inputs_stage_2(BATCH_SIZE,NUM_POINT)
                is_training_feature= False
                is_training_pl = tf.placeholder(tf.bool, shape=())
                # Note the global_step=batch parameter to minimize. 
                # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
                batch_stage_2 = tf.Variable(0,name='stage2/batch_2')
                bn_decay = get_bn_decay(batch_stage_2)
                tf.summary.scalar('bn_decay', bn_decay)
                print("--- Get model and loss")
                # Get model and loss 
                end_points,dof_feat,simmat_feat = MODEL.get_feature(pointclouds_pl, is_training_feature,STAGE,bn_decay=bn_decay)
                pred_dof_score,all_feat = MODEL.get_stage_2(dof_feat,simmat_feat,dof_mask_pl,proposal_nx_pl,is_training_pl,bn_decay=bn_decay)
                loss = MODEL.get_stage_2_loss(pred_dof_score,dof_score_pl,dof_mask_pl)
                tf.summary.scalar('loss', loss)
                print("--- Get training operator")
                # Get training operator
                learning_rate = get_learning_rate(batch_stage_2)
                tf.summary.scalar('learning_rate', learning_rate)
                variables = tf.contrib.framework.get_variables_to_restore()
                variables_to_resotre = [v for v in variables if v.name.split('/')[0]=='pointnet']
                variables_to_train = [v for v in variables if v.name.split('/')[0]=='stage2']
                if OPTIMIZER == 'momentum':
                    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
                elif OPTIMIZER == 'adam':
                    optimizer = tf.train.AdamOptimizer(learning_rate)
                train_op = optimizer.minimize(loss, global_step=batch_stage_2,var_list = variables_to_train)
                # Add ops to save and restore all the variables.
                saver = tf.train.Saver(var_list = variables_to_resotre)
                saver2 = tf.train.Saver(max_to_keep=100)
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
               'corner_pair': corner_pair,
               'label_1': label_1,
               'label_2': label_2,
               'label_5': label_5,
	       'label_6': label_6,
	       'label_7': label_7,
	       'label_8': label_8,
               'is_training_pl': is_training_pl,
               'pre_label_1': pre_label_1,
               'pre_label_2': pre_label_2,
               'pre_label_5': pre_label_5,
               'pre_label_6': pre_label_6,
               'pre_label_7': pre_label_7,
               'pre_label_8': pre_label_8,			   
               'merged': merged,
               'step': batch_stage_1,
               'end_points': end_points}
            for epoch in range(MAX_EPOCH):
                log_string('**** TEST EPOCH %03d ****' % (epoch))
                sys.stdout.flush()
                eval_one_epoch_stage_1(sess,ops,test_writer)
        elif STAGE==2:
            ops = {'pointclouds_pl': pointclouds_pl,
               'proposal_nx_pl': proposal_nx_pl,
               'dof_mask_pl': dof_mask_pl,
               'dof_score_pl': dof_score_pl,
               'pred_dof_score': pred_dof_score,
               'is_training_pl': is_training_pl,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch_stage_2,
               'all_feat':all_feat,
               'end_points': end_points}
            for epoch in range(MAX_EPOCH):
                log_string('**** TRAIN EPOCH %03d ****' % (epoch))
                sys.stdout.flush()
                train_one_epoch_stage_2(sess,ops,train_writer)
                # Save the variables to disk.
                if epoch % 2 == 0:
                    model_ccc_path = "model"+str(epoch)+".ckpt"
                    save_path = saver2.save(sess, os.path.join(LOG_DIR, model_ccc_path))
                    log_string("Model saved in file: %s" % save_path)


def eval_one_epoch_stage_1(sess, ops, train_writer):
    is_training = True
    dataset = ['./test_data_2_1/test_pred_135.mat']
    for i in range(len(dataset)):
        load_data_start_time = time.time();
        train_data = sio.loadmat(dataset[i])['Training_data_stage2']
        load_data_duration = time.time() - load_data_start_time
        log_string('\t%s: %s load time: %f' % (datetime.now(),dataset[i],load_data_duration))
        num_data = train_data.shape[0]
        num_batch = num_data // BATCH_SIZE
        process_start_time = time.time()
        pre_label_1 = np.zeros((num_data,NUM_ROI,2),np.float32)  
        pre_label_2 = np.zeros((num_data,NUM_ROI,2),np.float32)
        pre_label_5 = np.zeros((num_data,NUM_POINT,2),np.float32)
        pre_label_6 = np.zeros((num_data,NUM_POINT,2),np.float32)
        pre_label_7 = np.zeros((num_data,NUM_POINT,2),np.float32)
        pre_label_8 = np.zeros((num_data,NUM_POINT,2),np.float32)
        input_points_edge_corner = np.zeros((num_data,NUM_POINT,5),np.float32)  
	input_corner_pair = np.zeros((num_data,NUM_ROI,2),np.float32)


        for x in range(num_data):
            tmp_data = train_data[x,0]
            input_points_edge_corner[x,:,:] = tmp_data['input_point_cloud_edge_corner'][0, 0]
            input_corner_pair[x,:,:] = tmp_data['train_all_pair_rand'][0, 0]

            
        for j in range(num_batch):
            begin_idx = j*BATCH_SIZE
            end_idx = (j+1)*BATCH_SIZE
            data_cells = train_data[begin_idx: end_idx,0]
            batch_inputs = np.zeros((BATCH_SIZE,NUM_POINT,5),np.float32)  
	    batch_corner_pair = np.zeros((BATCH_SIZE,NUM_ROI,2),np.float32)
            for cnt in range(BATCH_SIZE):
                tmp_data = data_cells[cnt]
                batch_inputs[cnt,:,:] = tmp_data['input_point_cloud_edge_corner'][0, 0]
		batch_corner_pair[cnt,:,:] = tmp_data['train_all_pair_rand'][0, 0]
            feed_dict = {ops['pointclouds_pl']: batch_inputs,
                         ops['corner_pair']: batch_corner_pair,
                         ops['is_training_pl']: is_training}
                 
                    
            summary, step, pre_label_1[begin_idx:end_idx,:,:], \
                                 pre_label_2[begin_idx:end_idx,:,:], \
                                 pre_label_5[begin_idx:end_idx,:,:], \
                                 pre_label_6[begin_idx:end_idx,:,:], \
                                 pre_label_7[begin_idx:end_idx,:,:], \
                                 pre_label_8[begin_idx:end_idx,:,:] = sess.run([ops['merged'], ops['step'], \
                                                  ops['pre_label_1'],ops['pre_label_2'],ops['pre_label_5'], \
                                                  ops['pre_label_6'],ops['pre_label_7'],ops['pre_label_8']], feed_dict=feed_dict) 
          
            train_writer.add_summary(summary, step) 
			
			
        
        process_duration = time.time() - process_start_time
        examples_per_sec = num_data/process_duration
        sec_per_batch = process_duration/num_batch        
        temp_name = dataset[i]   
        temp_name = temp_name[16:]
        sio.savemat('./test_result_2_1/test_pred_'+temp_name, {'input_points_edge_corner': input_points_edge_corner, \
                                                    'input_corner_pair': input_corner_pair, \
                                                    'pre_label_1': pre_label_1, \
                                                    'pre_label_2': pre_label_2, \
                                                    'pre_label_5': pre_label_5, \
                                                    'pre_label_6': pre_label_6, \
                                                    'pre_label_7': pre_label_7, \
                                                    'pre_label_8': pre_label_8}) 

def train_one_epoch_stage_2(sess, ops, train_writer):
    is_training = True
    permutation = np.random.permutation(328)
    for i in range(len(permutation)/4):
        load_data_start_time = time.time();
        loadpath = './train_data_stage_2/train_stage_2_data_'+str(permutation[i*4]+1)+'.mat'
        train_data = sio.loadmat(loadpath)['Training_data']
        load_data_duration = time.time() - load_data_start_time
        log_string('\t%s: %s load time: %f' % (datetime.now(),loadpath,load_data_duration))
        for j in range(3):
            temp_load_data_start_time = time.time();
            temp_loadpath = './train_data_stage_2/train_stage_2_data_'+str(permutation[i*4+j+1]+1)+'.mat'
            temp_train_data = sio.loadmat(temp_loadpath)['Training_data']
            temp_load_data_duration = time.time() - temp_load_data_start_time
            log_string('\t%s: %s load time: %f' % (datetime.now(),temp_loadpath,temp_load_data_duration))
            train_data = np.concatenate((train_data,temp_train_data),axis = 0)
            print(train_data.shape)
        
        num_data = train_data.shape[0]
        num_batch = num_data // BATCH_SIZE
        total_loss = 0.0
        process_start_time = time.time()
        np.random.shuffle(train_data)
        
        for j in range(num_batch):
            begin_idx = j*BATCH_SIZE
            end_idx = (j+1)*BATCH_SIZE
            data_cells = train_data[begin_idx: end_idx,0]
            batch_inputs = np.zeros((BATCH_SIZE,NUM_POINT,6),np.float32)
            batch_dof_mask = np.zeros((BATCH_SIZE,NUM_POINT),np.int32)
            batch_proposal_nx = np.zeros((BATCH_SIZE,NUM_POINT),np.int32)
            batch_dof_score = np.zeros((BATCH_SIZE,NUM_POINT),np.float32)
            for cnt in range(BATCH_SIZE):
                tmp_data = data_cells[cnt]
                batch_inputs[cnt,:,:] = tmp_data['inputs_all'][0, 0]
                batch_dof_mask[cnt,:] = np.squeeze(tmp_data['dof_mask'][0,0])
                batch_proposal_nx[cnt,:] = np.squeeze(tmp_data['proposal_nx'][0,0])
                batch_dof_score[cnt,:] = np.squeeze(tmp_data['dof_score'][0,0])
            feed_dict = {ops['pointclouds_pl']: batch_inputs,
                         ops['proposal_nx_pl']: batch_proposal_nx,
                         ops['dof_mask_pl']: batch_dof_mask,
                         ops['dof_score_pl']: batch_dof_score,
                         ops['is_training_pl']: is_training}
                    
            summary, step, _, loss_val = sess.run([ops['merged'], ops['step'], \
                                 ops['train_op'],ops['loss']],feed_dict=feed_dict)
            train_writer.add_summary(summary, step)
            total_loss += loss_val
            #print('loss: %f' % loss_val)
        total_loss = total_loss * 1.0 / num_batch
        process_duration = time.time() - process_start_time
        examples_per_sec = num_data/process_duration
        sec_per_batch = process_duration/num_batch
        log_string('\t%s: step: %f loss: %f duration time %.3f (%.1f examples/sec; %.3f sec/batch)' \
           % (datetime.now(),step,total_loss,process_duration,examples_per_sec,sec_per_batch))


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
