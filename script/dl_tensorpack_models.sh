#!/usr/bin/env bash
ROOT=$( cd $(dirname $0)/../models ; pwd -P )

# Download TensorPack detection models
MODELS=(COCO-R50FPN-MaskRCNN-Standard)
SAVE_PATH=${ROOT}/tensorpack
mkdir -p ${SAVE_PATH}
for model in ${MODELS[@]}
do
    MODEL_FILE=${model}.npz
    wget http://models.tensorpack.com/FasterRCNN/${MODEL_FILE} -P ${SAVE_PATH}
done