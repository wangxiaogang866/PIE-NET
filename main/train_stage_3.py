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
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='model2', help='Model name [default: model]')
parser.add_argument('--stage_3_log_dir', default='stage_3_log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=4096, help='Point Number [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=101, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
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

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.stage_3_log_dir
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
            print('stage_1')
            pointclouds_pl,proposal_pl,dof_regression_pl,field_pl=MODEL.placeholder_inputs_stage_3(BATCH_SIZE,NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)
            print "--- Get model and loss"
            # Get model and loss 
            pred_proposal,pred_dof_regression = MODEL.get_stage_3(pointclouds_pl,field_pl, is_training_pl,bn_decay=bn_decay)
            task_1_acc,iou,loss1,loss2,loss = MODEL.get_stage_3_loss(pred_proposal,pred_dof_regression,proposal_pl,dof_regression_pl)
            tf.summary.scalar('loss', loss)

            print "--- Get training operator"
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            variables = tf.contrib.framework.get_variables_to_restore()
            print "variables"
            for v in variables:
                print v
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver(max_to_keep=100)
       
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
        init = tf.global_variables_initializer()
        sess.run(init)
        ops = {'pointclouds_pl': pointclouds_pl,
               'proposal_pl': proposal_pl,
               'dof_regression_pl': dof_regression_pl,
               'field_pl': field_pl,
               'pred_proposal': pred_proposal,
               'pred_dof_regression': pred_dof_regression,
               'is_training_pl': is_training_pl,
               'loss': loss,
               'loss1':loss1,
               'loss2':loss2,
               'task_1_acc':task_1_acc,
               'iou':iou,
               'train_op': train_op,
               'merged': merged,
               'step': batch}
        for epoch in range(MAX_EPOCH):
            log_string('**** TRAIN EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
            train_one_epoch_stage_3(sess,ops,train_writer)
            # Save the variables to disk.
            if epoch % 2 == 0:
                model_ccc_path = "model"+str(epoch)+".ckpt"
                save_path = saver.save(sess, os.path.join(LOG_DIR, model_ccc_path))
                log_string("Model saved in file: %s" % save_path)

def train_one_epoch_stage_3(sess, ops, train_writer):
    is_training = True
    permutation = np.random.permutation(328)
    for i in range(328/4):
        load_data_start_time = time.time();
        loadpath = './train_data_stage_3/train_data_stage_3_data_'+str(permutation[i*4]+1)+'.mat'
        train_data = sio.loadmat(loadpath)['Training_data']
        load_data_duration = time.time() - load_data_start_time
        log_string('\t%s: %s load time: %f' % (datetime.now(),loadpath,load_data_duration))
        for j in range(3):
            temp_load_data_start_time = time.time();
            temp_loadpath = './train_data_stage_3/train_data_stage_3_data_'+str(permutation[i*4+j+1]+1)+'.mat'
            temp_train_data = sio.loadmat(temp_loadpath)['Training_data']
            temp_load_data_duration = time.time() - temp_load_data_start_time
            log_string('\t%s: %s load time: %f' % (datetime.now(),temp_loadpath,temp_load_data_duration))
            train_data = np.concatenate((train_data,temp_train_data),axis = 0)
            print(train_data.shape)
        num_data = train_data.shape[0]
        num_batch = num_data // BATCH_SIZE
        total_loss = 0.0
        total_loss1 = 0.0
        total_loss2 = 0.0
        total_acc = 0.0
        total_iou = 0.0
        process_start_time = time.time()
        np.random.shuffle(train_data)
        for j in range(num_batch):
            begin_idx = j*BATCH_SIZE
            end_idx = (j+1)*BATCH_SIZE
            data_cells = train_data[begin_idx: end_idx,0]
            batch_inputs = np.zeros((BATCH_SIZE,NUM_POINT,6),np.float32)
            batch_proposal = np.zeros((BATCH_SIZE,NUM_POINT),np.int32)
            batch_dof_regression = np.zeros((BATCH_SIZE,1,6),np.float32)
            batch_field = np.zeros((BATCH_SIZE,3,NUM_POINT,6),np.float32)
            for cnt in range(BATCH_SIZE):
                tmp_data = data_cells[cnt]
                batch_inputs[cnt,:,:] = tmp_data['inputs_all'][0, 0]
                batch_proposal[cnt,:] = np.squeeze(tmp_data['proposal'][0,0])
                batch_dof_regression[cnt,:,:] = tmp_data['dof_regression'][0,0]
                batch_field[cnt,:,:,:] = tmp_data['field'][0,0]
            feed_dict = {ops['pointclouds_pl']: batch_inputs,
                         ops['proposal_pl']: batch_proposal,
                         ops['dof_regression_pl']: batch_dof_regression,
                         ops['field_pl']: batch_field,
                         ops['is_training_pl']: is_training}
                    
            summary, step, _, loss_val,loss1_val,loss2_val,acc_val,iou_val = sess.run([ops['merged'], ops['step'], \
                                 ops['train_op'],ops['loss'],ops['loss1'],ops['loss2'],ops['task_1_acc'],ops['iou']],feed_dict=feed_dict)
            train_writer.add_summary(summary, step)
            total_loss += loss_val
            total_loss1 += loss1_val
            total_loss2 += loss2_val
            total_acc +=acc_val
            total_iou +=iou_val 
            #print('loss: %f' % loss_val)
        total_loss = total_loss * 1.0 / num_batch
        total_loss1 = total_loss1 * 1.0 / num_batch
        total_loss2 = total_loss2 * 1.0 / num_batch
        total_acc = total_acc * 1.0 / num_batch
        total_iou = total_iou * 1.0 / num_batch
        process_duration = time.time() - process_start_time
        examples_per_sec = num_data/process_duration
        sec_per_batch = process_duration/num_batch
        log_string('\t%s: step: %f loss: %f duration time %.3f (%.1f examples/sec; %.3f sec/batch)' \
           % (datetime.now(),step,total_loss,process_duration,examples_per_sec,sec_per_batch))
        log_string('\tTask1 loss: %f,Task2 loss: %f'%(total_loss1,total_loss2))
        log_string('\tTask1 acc: %f,Task1 iou: %f'%(total_acc,total_iou))


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
