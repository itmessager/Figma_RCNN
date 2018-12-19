#!/usr/bin/env bash
ROOT=$( cd $(dirname $0)/../../models ; pwd -P )

# Download TF official object detection models
MODELS=(ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03)
SAVE_PATH=${ROOT}/tfmodels
mkdir -p ${SAVE_PATH}
for model in ${MODELS[@]}
do
    MODEL_FILE=${model}.tar.gz
    wget http://download.tensorflow.org/models/object_detection/${MODEL_FILE} -P ${SAVE_PATH}
    cd ${SAVE_PATH} && tar -xvf ${MODEL_FILE} && rm ${MODEL_FILE}
done