#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train.py

import argparse
import cv2
import shutil
import itertools
import tqdm
import numpy as np
import json
import six
import os

from detection.tensorpacks.common import CustomResize

try:
    import horovod.tensorflow as hvd
except ImportError:
    pass

import tensorflow as tf

assert six.PY3, "FasterRCNN requires Python 3!"

from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils import optimizer
from tensorpack.tfutils.common import get_tf_version_tuple
import tensorpack.utils.viz as tpviz

from detection.tensorpacks.coco import COCODetection
from detection.tensorpacks.basemodel import (
    image_preprocess, resnet_c4_backbone, resnet_conv5,
    resnet_fpn_backbone)

from detection.tensorpacks.model_frcnn import (
    sample_fast_rcnn_targets, fastrcnn_outputs,
    fastrcnn_predictions, BoxProposals, FastRCNNHead, attrs_head, attrs_predict)
from detection.tensorpacks.model_mrcnn import maskrcnn_upXconv_head, maskrcnn_loss
from detection.tensorpacks.model_rpn import rpn_head, rpn_losses, generate_rpn_proposals
from detection.tensorpacks.model_fpn import (
    fpn_model, multilevel_roi_align,
    generate_fpn_proposals)
from detection.tensorpacks.model_cascade import CascadeRCNNHead
from detection.tensorpacks.model_box import (
    clip_boxes, crop_and_resize, roi_align, RPNAnchors)

from detection.tensorpacks.data import (
    get_train_dataflow, get_eval_dataflow, get_attributes_dataflow,
    get_all_anchors, get_all_anchors_fpn)
from detection.tensorpacks.viz import (
    draw_annotation, draw_proposal_recall,
    draw_predictions, draw_final_outputs)
from detection.tensorpacks.eval import (
    eval_coco, print_evaluation_scores, DetectionResult, fill_full_mask)
from detection.config.tensorpack_config import finalize_configs, config as cfg


class DetectionModel(ModelDesc):
    def preprocess(self, image):
        image = tf.expand_dims(image, 0)
        image = image_preprocess(image, bgr=True)
        return tf.transpose(image, [0, 3, 1, 2])

class ResNetC4Model(DetectionModel):
    def inputs(self):  # OK
        ret = [
            tf.placeholder(tf.float32, (None, None, 3), 'image'),
            # label of each anchor
            tf.placeholder(tf.int64, (None,), 'male'),
            # box of each ground truth
            tf.placeholder(tf.float32, (None, 4), 'gt_boxes')]
        # label of each ground truth
        # tf.placeholder(tf.int64, (None,), 'gt_labels'), # all > 0
        return ret

    def build_graph(self, *inputs):
        inputs = dict(zip(self.input_names, inputs))
        image = self.preprocess(inputs['image'])  # 1CHW

        # build resnet c4
        featuremap = resnet_c4_backbone(image, cfg.BACKBONE.RESNET_NUM_BLOCK[:3])

        boxes_on_featuremap = inputs['gt_boxes'] * (1.0 / cfg.RPN.ANCHOR_STRIDE)  # ANCHOR_STRIDE = 16

        # ROI_align
        roi_resized = roi_align(featuremap, boxes_on_featuremap, 14)  # 14x14 for each roi

        feature_fastrcnn = resnet_conv5(roi_resized,
                                        cfg.BACKBONE.RESNET_NUM_BLOCK[-1])  # nxcx7x7 # RESNET_NUM_BLOCK = [3, 4, 6, 3]
        # Keep C5 feature to be shared with mask branch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='load a model for evaluation. Can overwrite BACKBONE.WEIGHTS')

    args = parser.parse_args()

    MODEL = ResNetC4Model()

    pred = OfflinePredictor(PredictConfig(
        model=MODEL,  # model
        session_init=get_model_loader(args.load),  # weight
        input_names=['image', 'gt_boxes'],
        output_names=['image','gt_boxes','mul','group2/block5/output'
                      ]))

    img = cv2.imread('/root/datasets/wider attribute'
                     '/train/50--Celebration_Or_Party/'
                     '50_Celebration_Or_Party_houseparty_50_985.jpg')
    boxes = np.array([(685.53906,1066.394,157.42007,289.44983),(766.71625,1099.9194,153.01115,265.08054),
     (879.69763,1125.4299,139.75713,235.0632),(865.3703 ,376.64493,131.95422,201.4785),
    (348.02548,1084.0994,149.65096,205.33504)])

    prediction = pred(img,boxes)[0]
    # cv2.imwrite('applied_default.jpg', prediction[0])
