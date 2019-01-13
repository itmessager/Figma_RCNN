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

from detection.visualization.vis_boxes_on_image import vis_one_image


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
        featuremap = resnet_c4_backbone(image, cfg.BACKBONE.RESNET_NUM_BLOCK[:3])
        # rpn_label_logits, rpn_box_logits = rpn_head('rpn', featuremap, cfg.RPN.HEAD_DIM, cfg.RPN.NUM_ANCHOR)
        # # HEAD_DIM = 1024, NUM_ANCHOR = 15
        # # rpn_label_logits: fHxfWxNA
        # # rpn_box_logits: fHxfWxNAx4
        # anchors = RPNAnchors(get_all_anchors(), inputs['anchor_labels'], inputs['anchor_boxes'])
        # # anchor_boxes is Groundtruth boxes corresponding to each anchor
        # anchors = anchors.narrow_to(featuremap)  # ??
        # image_shape2d = tf.shape(image)[2:]  # h,w
        # pred_boxes_decoded = anchors.decode_logits(rpn_box_logits)  # fHxfWxNAx4, floatbox
        #
        # # ProposalCreator (get the topk proposals)
        # proposal_boxes, proposal_scores = generate_rpn_proposals(
        #     tf.reshape(pred_boxes_decoded, [-1, 4]),
        #     tf.reshape(rpn_label_logits, [-1]),
        #     image_shape2d,
        #     cfg.RPN.TEST_PRE_NMS_TOPK,  # 2000
        #     cfg.RPN.TEST_POST_NMS_TOPK)  # 1000







        # build resnet c4

        x, y, w, h = tf.split(inputs['gt_boxes'], 4, axis=1)
        gt_boxes = tf.concat([x, y, x + w, y + h], axis=1)
        boxes_on_featuremap = gt_boxes * (1.0 / cfg.RPN.ANCHOR_STRIDE)  # ANCHOR_STRIDE = 16

        # ROI_align
        roi_resized = roi_align(featuremap, boxes_on_featuremap, 14)  # 14x14 for each roi

        feature_fastrcnn = resnet_conv5(roi_resized,
                                        cfg.BACKBONE.RESNET_NUM_BLOCK[-1])  # nxcx7x7 # RESNET_NUM_BLOCK = [3, 4, 6, 3]
        # Keep C5 feature to be shared with mask branch





if __name__ == '__main__':
    MODEL = ResNetC4Model()

    pred = OfflinePredictor(PredictConfig(
        model=MODEL,  # model
        session_init=get_model_loader('/root/datasets/COCO-R50C4-MaskRCNN-Standard.npz'),  # weight
        input_names=['image', 'gt_boxes'],
        output_names=['image', 'gt_boxes'
                      ]))
    img = cv2.imread('/root/datasets/wider attribute/train/1--Handshaking/1_Handshaking_Handshaking_1_765.jpg')
    boxes = np.array([(93.06605, 121.849365, 380.8992, 593.8957),
                      (477.803, 86.349945, 476.84357, 634.1923)])

    prediction = pred(img, boxes)

    img = (prediction[3].transpose(0, 2, 3, 1)[0][:, :, 1:1024:500]*100).astype(np.uint8)
    vis_one_image(img, prediction[2],16)
    #
    # img = prediction[0].astype(np.uint8)
    # vis_one_image(img, prediction[1])
