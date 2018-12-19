CUR_DIR=$( cd $(dirname $0) ; pwd -P )
OBJECT_DETECTION_HOME=${MODELS_HOME}/research/object_detection
MODEL=wider_ssd_resnet50_v1_fpn

CUDA_VISIBLE_DEVICES="0" python3 ${OBJECT_DETECTION_HOME}/model_main.py \
    --pipeline_config_path=${CUR_DIR}/${MODEL}.config \
    --model_dir=${CUR_DIR}/../models/${MODEL} \
    --alsologtostderr