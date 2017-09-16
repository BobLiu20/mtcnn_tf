#coding:utf-8
import tensorflow as tf
import numpy as np
import os
from datetime import datetime
import sys
import argparse
rootPath = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
sys.path.insert(0, rootPath)
from tools.tfrecord_reader import read_multi_tfrecords,read_single_tfrecord
from mtcnn_config import config
from mtcnn_model import P_Net, R_Net, O_Net
import cv2

def train_model(baseLr, loss, data_num):
    """
    train model
    :param baseLr: base learning rate
    :param loss: loss
    :param data_num:
    :return:
    train_op, lr_op
    """
    lr_factor = 0.1
    global_step = tf.Variable(0, trainable=False)
    #LR_EPOCH [8,14]
    #boundaried [num_batch,num_batch]
    boundaries = [int(epoch * data_num / config.BATCH_SIZE) for epoch in config.LR_EPOCH]
    #lr_values[0.01,0.001,0.0001,0.00001]
    lr_values = [baseLr * (lr_factor ** x) for x in range(0, len(config.LR_EPOCH) + 1)]
    #control learning rate
    lr_op = tf.train.piecewise_constant(global_step, boundaries, lr_values)
    optimizer = tf.train.MomentumOptimizer(lr_op, 0.9)
    train_op = optimizer.minimize(loss, global_step)
    return train_op, lr_op

# all mini-batch mirror
def random_flip_images(image_batch,label_batch,landmark_batch):
    #mirror
    if np.random.choice([0,1]) > 0:
        num_images = image_batch.shape[0]
        fliplandmarkindexes = np.where(label_batch==-2)[0]
        flipposindexes = np.where(label_batch==1)[0]
        #only flip
        flipindexes = np.concatenate((fliplandmarkindexes,flipposindexes))
        #random flip    
        for i in flipindexes:
            cv2.flip(image_batch[i],1,image_batch[i])        
        
        #pay attention: flip landmark    
        for i in fliplandmarkindexes:
            landmark_ = landmark_batch[i].reshape((-1,2))
            landmark_ = np.asarray([(1-x, y) for (x, y) in landmark_])
            landmark_[[0, 1]] = landmark_[[1, 0]]#left eye<->right eye
            landmark_[[3, 4]] = landmark_[[4, 3]]#left mouth<->right mouth        
            landmark_batch[i] = landmark_.ravel()
        
    return image_batch,landmark_batch

def train(netFactory, modelPrefix, endEpoch, dataPath, display=200, baseLr=0.01, gpus=""):
    net = modelPrefix.split('/')[-1]
    print("Now start to train...stage: %s"%(net))
    # set GPU
    if gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    if net == 'pnet': # PNet use this method to get data
        dataset_dir = os.path.join(dataPath, 'all.tfrecord')
        total_num = sum(1 for _ in tf.python_io.tf_record_iterator(dataset_dir))
        image_batch, label_batch, bbox_batch, landmark_batch = read_single_tfrecord(dataset_dir, config.BATCH_SIZE, net)
    elif net in ['rnet', 'onet']: # RNet and ONet use 4 tfrecords to get data
        pos_dir = os.path.join(dataPath, 'pos.tfrecord')
        part_dir = os.path.join(dataPath, 'part.tfrecord')
        neg_dir = os.path.join(dataPath, 'neg.tfrecord')
        landmark_dir = os.path.join(dataPath, 'landmark.tfrecord')
        dataset_dirs = [pos_dir, part_dir, neg_dir, landmark_dir]
        pos_ratio, part_ratio, landmark_ratio, neg_ratio = 1.0/6, 1.0/6, 1.0/6, 3.0/6
        pos_batch_size = int(np.ceil(config.BATCH_SIZE*pos_ratio))
        part_batch_size = int(np.ceil(config.BATCH_SIZE*part_ratio))
        neg_batch_size = int(np.ceil(config.BATCH_SIZE*neg_ratio))
        landmark_batch_size = int(np.ceil(config.BATCH_SIZE*landmark_ratio))
        batch_sizes = [pos_batch_size, part_batch_size, neg_batch_size, landmark_batch_size]
        image_batch, label_batch, bbox_batch, landmark_batch = read_multi_tfrecords(dataset_dirs, batch_sizes, net)        
        total_num = 0
        for d in dataset_dirs:
            total_num += sum(1 for _ in tf.python_io.tf_record_iterator(d))
    #ratio 
    if net == 'pnet':
        image_size = 12
        ratio_cls_loss, ratio_bbox_loss, ratio_landmark_loss = 1.0, 0.5, 0.5
    elif net == 'rnet':
        image_size = 24
        ratio_cls_loss, ratio_bbox_loss, ratio_landmark_loss = 1.0, 0.5, 1.0
    elif net == 'onet':
        ratio_cls_loss, ratio_bbox_loss, ratio_landmark_loss = 1.0, 0.5, 1.0
        image_size = 48
    else:
        raise Exception("incorrect net type.")
    #define placeholder
    input_image = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, image_size, image_size, 3], name='input_image')
    label = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE], name='label')
    bbox_target = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, 4], name='bbox_target')
    landmark_target = tf.placeholder(tf.float32,shape=[config.BATCH_SIZE,10],name='landmark_target')
    #class,regression
    cls_loss_op,bbox_loss_op,landmark_loss_op,L2_loss_op,accuracy_op = netFactory(input_image, label, bbox_target,landmark_target,training=True)
    #train,update learning rate(3 loss)
    train_op, lr_op = train_model(baseLr, ratio_cls_loss*cls_loss_op + ratio_bbox_loss*bbox_loss_op + ratio_landmark_loss*landmark_loss_op + L2_loss_op, total_num)
    # init
    init = tf.global_variables_initializer()
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    #save model
    saver = tf.train.Saver(max_to_keep=0)
    sess.run(init)
    #visualize some variables
    tf.summary.scalar("cls_loss",cls_loss_op)#cls_loss
    tf.summary.scalar("bbox_loss",bbox_loss_op)#bbox_loss
    tf.summary.scalar("landmark_loss",landmark_loss_op)#landmark_loss
    tf.summary.scalar("cls_accuracy",accuracy_op)#cls_acc
    summary_op = tf.summary.merge_all()
    logs_dir = os.path.join(rootPath, "tmp", "logs", net)
    if os.path.exists(logs_dir) == False:
        os.makedirs(logs_dir)
    writer = tf.summary.FileWriter(logs_dir, sess.graph)
    #begin 
    coord = tf.train.Coordinator()
    #begin enqueue thread
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    i = 0
    #total steps
    MAX_STEP = int(total_num / config.BATCH_SIZE + 1) * endEpoch
    print "\n\nTotal step: ", MAX_STEP
    epoch = 0
    sess.graph.finalize()    
    try:
        for step in range(MAX_STEP):
            i = i + 1
            if coord.should_stop():
                break
            image_batch_array, label_batch_array, bbox_batch_array,landmark_batch_array = sess.run([image_batch, label_batch, bbox_batch,landmark_batch])
            #random flip
            image_batch_array,landmark_batch_array = random_flip_images(image_batch_array,label_batch_array,landmark_batch_array)
            '''
            print image_batch_array.shape
            print label_batch_array.shape
            print bbox_batch_array.shape
            print landmark_batch_array.shape
            print label_batch_array[0]
            print bbox_batch_array[0]
            print landmark_batch_array[0]
            '''
            _,_,summary = sess.run([train_op, lr_op ,summary_op], feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: bbox_batch_array,landmark_target:landmark_batch_array})
            
            if (step+1) % display == 0:
                #acc = accuracy(cls_pred, labels_batch)
                cls_loss, bbox_loss,landmark_loss,L2_loss,lr,acc = sess.run([cls_loss_op, bbox_loss_op,landmark_loss_op,L2_loss_op,lr_op,accuracy_op],
                                                             feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: bbox_batch_array, landmark_target: landmark_batch_array})                
                print("%s [%s] Step: %d, accuracy: %3f, cls loss: %4f, bbox loss: %4f, landmark loss: %4f,L2 loss: %4f,lr:%f " % (
                datetime.now(), net, step+1, acc, cls_loss, bbox_loss, landmark_loss, L2_loss, lr))
            #save every two epochs
            if i * config.BATCH_SIZE > total_num*2:
                epoch = epoch + 1
                i = 0
                saver.save(sess, modelPrefix, global_step=epoch*2)
            writer.add_summary(summary,global_step=step)
    except tf.errors.OutOfRangeError:
        print("Done!")
    finally:
        coord.request_stop()
        writer.close()
    coord.join(threads)
    sess.close()

def parse_args():
    parser = argparse.ArgumentParser(description='Create hard bbox sample...',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--stage', dest='stage', help='working stage, can be rnet, onet',
                        default='unknow', type=str)
    parser.add_argument('--gpus', dest='gpus', help='specify gpu to run. eg: --gpus=0,1',
                        default='0', type=str)
    parser.add_argument('--epoch', dest='epoch', help='total epoch to training',
                        default=30, type=int)
    parser.add_argument('--display', dest='display', help='how much step to display',
                        default=100, type=int)
    parser.add_argument('--lr', dest='lr', help='base learning rate',
                        default=0.01, type=float)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    print "The training argument info is: ", args
    if args.stage not in ['pnet', 'rnet', 'onet']:
        raise Exception("Please specify stage by --stage=pnet or rnet or onet")
    dataPath = os.path.join(rootPath, "tmp/data/%s"%(args.stage))
    modelPrefix = os.path.join(rootPath, "tmp/model/%s/%s"%(args.stage, args.stage))
    if not os.path.isdir(modelPrefix):
        os.makedirs(modelPrefix)
    
    _net = {'pnet': P_Net, 'rnet': R_Net, 'onet': O_Net}
    train(_net[args.stage], modelPrefix, args.epoch, dataPath, display=args.display, baseLr=args.lr, gpus=args.gpus)

