#!/bin/bash
set -e
### All of your tmp data will be saved in ./tmp folder

echo "Hello! I will prepare training data and starting to training step by step."

# 1. checking dataset if OK
if [ ! -d "./dataset/WIDER_train/images" ]; then
	echo "Error: The WIDER_train/images is not exist. Read dataset/README.md to get useful info."
	exit
fi
if [ ! -d "./dataset/lfw_5590" ]; then
	echo "Error: The lfw_5590 is not exist. Read dataset/README.md to get useful info."
	exit
fi
echo "Checking dataset pass."
if [ -d "./tmp" ]; then
	echo "Warning: The tmp folder is not empty. A good idea is to run ./clearAll.sh to clear it before training."
fi

## 2. stage: P-Net
#### generate training data(Face Detection Part) for PNet
#echo "Preparing P-Net training data: bbox"
#python prepare_data/gen_hard_bbox_pnet.py
#### generate training data(Face Landmark Detection Part) for PNet
#echo "Preparing P-Net training data: landmark"
#python prepare_data/gen_landmark_aug.py --stage=pnet
#### generate tfrecord file for tf training
#echo "Preparing P-Net tfrecord file"
#python prepare_data/gen_tfrecords.py --stage=pnet
#### start to training P-Net
#echo "Start to training P-Net"
#python training/train.py --stage=pnet
#
## 3. stage: R-Net
#### generate training data(Face Detection Part) for RNet
#echo "Preparing R-Net training data: bbox"
#python prepare_data/gen_hard_bbox_rnet_onet.py --stage=rnet
#### generate training data(Face Landmark Detection Part) for RNet
#echo "Preparing R-Net training data: landmark"
#python prepare_data/gen_landmark_aug.py --stage=rnet
#### generate tfrecord file for tf training
#echo "Preparing R-Net tfrecord file"
#python prepare_data/gen_tfrecords.py --stage=rnet
#### start to training R-Net
#echo "Start to training R-Net"
#python training/train.py --stage=rnet

# 4. stage: O-Net
### generate training data(Face Detection Part) for ONet
echo "Preparing O-Net training data: bbox"
python prepare_data/gen_hard_bbox_rnet_onet.py --stage=onet
### generate training data(Face Landmark Detection Part) for ONet
echo "Preparing O-Net training data: landmark"
python prepare_data/gen_landmark_aug.py --stage=onet
### generate tfrecord file for tf training
echo "Preparing O-Net tfrecord file"
python prepare_data/gen_tfrecords.py --stage=onet
### start to training O-Net
echo "Start to training O-Net"
python training/train.py --stage=onet

# 5. Done
echo "Congratulation! All stages had been done. Now you can going to testing and hope you enjoy your result."
echo "haha...bye bye"

