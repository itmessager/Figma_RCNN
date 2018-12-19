CUR_DIR=$( cd $(dirname $0) ; pwd -P )
INPUT_TYPE=image_tensor
MODEL=wider_ssd_resnet50_v1_fpn
MODEL_FOLDER=${CUR_DIR}/../../models/${MODEL}
OBJECT_DETECTION_HOME=${MODELS_HOME}/research/object_detection
CHECKPOINT_ID=${0}

python3 ${OBJECT_DETECTION_HOME}/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${CUR_DIR}/${MODEL}.config \
    --trained_checkpoint_prefix=${MODEL_FOLDER}/model.ckpt-${CHECKPOINT_ID} \
    --output_directory=${MODEL_FOLDER}