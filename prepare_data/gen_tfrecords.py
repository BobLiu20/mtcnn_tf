# coding: utf-8
import sys, os
rootPath = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
sys.path.insert(0, rootPath)
import numpy as np
import argparse

from tools.tfrecord_utils import _process_image_withoutcoder, _convert_to_example_simple
import tensorflow as tf

def __iter_all_data(net, iterType):
    saveFolder = os.path.join(rootPath, "tmp/data/%s/"%(net))
    if net not in ['pnet', 'rnet', 'onet']:
        raise Exception("The net type error!")
    if not os.path.isfile(os.path.join(saveFolder, 'pos.txt')):
        raise Exception("Please gen pos.txt in first!")
    if not os.path.isfile(os.path.join(saveFolder, 'landmark.txt')):
        raise Exception("Please gen landmark.txt in first!")
    if iterType == 'all':
        with open(os.path.join(saveFolder, 'pos.txt'), 'r') as f:
            pos = f.readlines()
        with open(os.path.join(saveFolder, 'neg.txt'), 'r') as f:
            neg = f.readlines()
        with open(os.path.join(saveFolder, 'part.txt'), 'r') as f:
            part = f.readlines()
        # keep sample ratio [neg, pos, part] = [3, 1, 1]
        base_num = min([len(neg), len(pos), len(part)])
        if len(neg) > base_num * 3:
            neg_keep = np.random.choice(len(neg), size=base_num * 3, replace=False)
        else:
            neg_keep = np.random.choice(len(neg), size=len(neg), replace=False)
        pos_keep = np.random.choice(len(pos), size=base_num, replace=False)
        part_keep = np.random.choice(len(part), size=base_num, replace=False)
        for i in pos_keep:
            yield pos[i]
        for i in neg_keep:
            yield neg[i]
        for i in part_keep:
            yield part[i]
        for item in open(os.path.join(saveFolder, 'landmark.txt'), 'r'):
            yield item 
    elif iterType in ['pos', 'neg', 'part', 'landmark']:
        for line in open(os.path.join(saveFolder, '%s.txt'%(iterType))):
            yield line
    else:
        raise Exception("Unsupport iter type.")

def __get_dataset(net, iterType):
    dataset = []
    for line in __iter_all_data(net, iterType):
        info = line.strip().split(' ')
        data_example = dict()
        bbox = dict()
        data_example['filename'] = info[0]
        data_example['label'] = int(info[1])
        bbox['xmin'] = 0
        bbox['ymin'] = 0
        bbox['xmax'] = 0
        bbox['ymax'] = 0
        bbox['xlefteye'] = 0
        bbox['ylefteye'] = 0
        bbox['xrighteye'] = 0
        bbox['yrighteye'] = 0
        bbox['xnose'] = 0
        bbox['ynose'] = 0
        bbox['xleftmouth'] = 0
        bbox['yleftmouth'] = 0
        bbox['xrightmouth'] = 0
        bbox['yrightmouth'] = 0        
        if len(info) == 6:
            bbox['xmin'] = float(info[2])
            bbox['ymin'] = float(info[3])
            bbox['xmax'] = float(info[4])
            bbox['ymax'] = float(info[5])
        if len(info) == 12:
            bbox['xlefteye'] = float(info[2])
            bbox['ylefteye'] = float(info[3])
            bbox['xrighteye'] = float(info[4])
            bbox['yrighteye'] = float(info[5])
            bbox['xnose'] = float(info[6])
            bbox['ynose'] = float(info[7])
            bbox['xleftmouth'] = float(info[8])
            bbox['yleftmouth'] = float(info[9])
            bbox['xrightmouth'] = float(info[10])
            bbox['yrightmouth'] = float(info[11])
        data_example['bbox'] = bbox
        dataset.append(data_example)
    return dataset

def __add_to_tfrecord(filename, image_example, tfrecord_writer):
    """
    Loads data from image and annotations files and add them to a TFRecord.
    """
    image_data, height, width = _process_image_withoutcoder(filename)
    example = _convert_to_example_simple(image_example, image_data)
    tfrecord_writer.write(example.SerializeToString())

def gen_tfrecords(net, shuffling=False):
    """
    Runs the conversion operation.
    """
    print(">>>>>> Start tfrecord create...Stage: %s"%(net))
    def _gen(tfFileName, net, iterType, shuffling):
        if tf.gfile.Exists(tfFileName):
            tf.gfile.Remove(tfFileName)
        # GET Dataset, and shuffling.
        dataset = __get_dataset(net=net, iterType=iterType)
        if shuffling:
            np.random.shuffle(dataset)
        # Process dataset files.
        # write the data to tfrecord
        with tf.python_io.TFRecordWriter(tfFileName) as tfrecord_writer:
            for i, image_example in enumerate(dataset):
                if i % 100 == 0:
                    sys.stdout.write('\rConverting[%s]: %d/%d' % (net, i + 1, len(dataset)))
                    sys.stdout.flush()
                filename = image_example['filename']
                __add_to_tfrecord(filename, image_example, tfrecord_writer)
        tfrecord_writer.close()
        print('\n')
    saveFolder = os.path.join(rootPath, "tmp/data/%s/"%(net))
    #tfrecord name 
    if net == 'pnet':
        tfFileName = os.path.join(saveFolder, "all.tfrecord")
        _gen(tfFileName, net, 'all', shuffling)
    elif net in ['rnet', 'onet']:
        for n in ['pos', 'neg', 'part', 'landmark']:
            tfFileName = os.path.join(saveFolder, "%s.tfrecord"%(n))
            _gen(tfFileName, net, n, shuffling)
    # Finally, write the labels file:
    print('\nFinished converting the MTCNN dataset!')
    print('All tf record was saved in %s'%(saveFolder))

def parse_args():
    parser = argparse.ArgumentParser(description='Create hard bbox sample...',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--stage', dest='stage', help='working stage, can be pnet, rnet, onet',
                        default='unknow', type=str)
    parser.add_argument('--gpus', dest='gpus', help='specify gpu to run. eg: --gpus=0,1',
                        default='0', type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    stage = args.stage
    if stage not in ['pnet', 'rnet', 'onet']:
        raise Exception("Please specify stage by --stage=pnet or rnet or onet")
    # set GPU
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    gen_tfrecords(stage, True)

