#coding:utf-8
import sys
import numpy as np
import cv2
import os
import argparse
import pickle
rootPath = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
sys.path.insert(0, rootPath)
from tools.common_utils import IoU, convert_to_square

from training.mtcnn_model import P_Net, R_Net
from training.mtcnn_config import config
from tools.loader import TestLoader
from detection.detector import Detector
from detection.fcn_detector import FcnDetector
from detection.MtcnnDetector import MtcnnDetector

def read_wider_annotation(widerImagesPath, annoTxtPath):
    data = dict()
    images = []
    bboxes = []
    labelfile = open(annoTxtPath, 'r')
    while True:
        # image path
        imagepath = labelfile.readline().strip('\n')
        if not imagepath:
            break
        imagepath = os.path.join(widerImagesPath, imagepath)
        images.append(imagepath)
        # face numbers
        nums = labelfile.readline().strip('\n')
        one_image_bboxes = []
        for i in range(int(nums)):
            bb_info = labelfile.readline().strip('\n').split(' ')
            # only need x, y, w, h
            face_box = [float(bb_info[i]) for i in range(4)]
            xmin = face_box[0]
            ymin = face_box[1]
            xmax = xmin + face_box[2]
            ymax = ymin + face_box[3]
            one_image_bboxes.append([xmin, ymin, xmax, ymax])
        bboxes.append(one_image_bboxes)
    data['images'] = images#all image pathes
    data['bboxes'] = bboxes#all image bboxes
    return data

def __save_data(stage, data, save_path):
    im_idx_list = data['images']
    gt_boxes_list = data['bboxes']
    num_of_images = len(im_idx_list)
    # save files
    saveFolder = os.path.join(rootPath, "tmp/data/%s/"%(stage))
    print(">>>>>> Gen hard samples for %s..."%(stage))
    typeName = ["pos", "neg", "part"]
    saveFiles = {}
    for tp in typeName:
        _saveFolder = os.path.join(saveFolder, tp)
        if not os.path.isdir(_saveFolder):
            os.makedirs(_saveFolder)
        saveFiles[tp] = open(os.path.join(saveFolder, "{}.txt".format(tp)), 'w')
    #read detect result
    det_boxes = pickle.load(open(os.path.join(save_path, 'detections.pkl'), 'rb'))
    assert len(det_boxes) == num_of_images, "incorrect detections or ground truths"
    # index of neg, pos and part face, used as their image names
    n_idx, p_idx, d_idx = 0, 0, 0
    total_idx = 0
    for im_idx, dets, gts in zip(im_idx_list, det_boxes, gt_boxes_list):
        gts = np.array(gts, dtype=np.float32).reshape(-1, 4)
        if dets.shape[0] == 0:
            continue
        img = cv2.imread(im_idx)
        total_idx += 1
        #change to square
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        neg_num = 0
        for box in dets:
            x_left, y_top, x_right, y_bottom, _ = box.astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1
            # ignore box that is too small or beyond image border
            if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
                continue
            # compute intersection over union(IoU) between current box and all gt boxes
            Iou = IoU(box, gts)
            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
            image_size = 24 if stage == "rnet" else 48
            resized_im = cv2.resize(cropped_im, (image_size, image_size),
                                    interpolation=cv2.INTER_LINEAR)
            # save negative images and write label
            # Iou with all gts must below 0.3            
            if np.max(Iou) < 0.3 and neg_num < 60:
                # now to save it
                save_file = os.path.join(saveFolder, "neg", "%s.jpg"%n_idx)
                saveFiles['neg'].write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
                neg_num += 1
            else:
                # find gt_box with the highest iou
                idx = np.argmax(Iou)
                assigned_gt = gts[idx]
                x1, y1, x2, y2 = assigned_gt
                # compute bbox reg label
                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)
                # save positive and part-face images and write labels
                if np.max(Iou) >= 0.65:
                    save_file = os.path.join(saveFolder, "pos", "%s.jpg"%p_idx)
                    saveFiles['pos'].write(save_file + ' 1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1
                elif np.max(Iou) >= 0.4:
                    save_file = os.path.join(saveFolder, "part", "%s.jpg"%d_idx)
                    saveFiles['part'].write(save_file + ' -1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
        printStr = "\r[{}] pos: {}  neg: {}  part:{}".format(total_idx, p_idx, n_idx, d_idx)
        sys.stdout.write(printStr)
        sys.stdout.flush()
    for f in saveFiles.values():
        f.close()
    print '\n'

def test_net(batch_size, stage, thresh, min_face_size, stride):
    print(">>>>>> Detect bbox for %s..."%(stage))
    detectors = [None, None, None]
    if stage in ["rnet", "onet"]:
        modelPath = os.path.join(rootPath, 'tmp/model/pnet/')
        a = [b[5:-6] for b in os.listdir(modelPath) if b.startswith('pnet-') and b.endswith('.index')]
        maxEpoch = max(map(int, a))
        modelPath = os.path.join(modelPath, "pnet-%d"%(maxEpoch))
        print("Use PNet model: %s"%(modelPath))
        PNet = FcnDetector(P_Net, modelPath)
        detectors[0] = PNet
    if stage in ["onet"]:
        modelPath = os.path.join(rootPath, 'tmp/model/rnet/')
        a = [b[5:-6] for b in os.listdir(modelPath) if b.startswith('rnet-') and b.endswith('.index')]
        maxEpoch = max(map(int, a))
        modelPath = os.path.join(modelPath, "rnet-%d"%(maxEpoch))
        print("Use RNet model: %s"%(modelPath))
        RNet = Detector(R_Net, 24, batch_size, modelPath)
        detectors[1] = RNet
    # read annatation(type:dict)
    widerImagesPath = os.path.join(rootPath, "dataset", "WIDER_train", "images")
    annoTxtPath = os.path.join(rootPath, "dataset", "wider_face_train_bbx_gt.txt")
    data = read_wider_annotation(widerImagesPath, annoTxtPath)
    mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                                   stride=stride, threshold=thresh)
    test_data = TestLoader(data['images'])
    # do detect
    detections, _ = mtcnn_detector.detect_face(test_data)
    # save detect result
    save_path = os.path.join(rootPath, "tmp/data", stage)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file = os.path.join(save_path, "detections.pkl")
    with open(save_file, 'wb') as f:
        pickle.dump(detections, f, 1)
    print("\nDone! Start to do OHEM...")
    __save_data(stage, data, save_path)

def parse_args():
    parser = argparse.ArgumentParser(description='Create hard bbox sample...',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--stage', dest='stage', help='working stage, can be rnet, onet',
                        default='unknow', type=str)
    parser.add_argument('--gpus', dest='gpus', help='specify gpu to run. eg: --gpus=0,1',
                        default='0', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    stage = args.stage
    gpus = args.gpus
    # set GPU
    if gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    if stage == "rnet":
        batchSize = 1
        threshold = [0.4, 0.05]
        minFace = 24
        stride = 2
    elif stage == "onet":
        batchSize = 1
        threshold = [0.4, 0.05]
        minFace = 24
        stride = 2
    else:
        raise Exception("Invaild stage...Please use --stage")
    test_net(
          batchSize, #test batch_size 
          stage, # can be 'rnet' or 'onet'
          threshold, #cls threshold
          minFace, #min_face
          stride)
