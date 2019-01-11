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

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.003, trainable=False)
        tf.summary.scalar('learning_rate-summary', lr)

        # The learning rate is set for 8 GPUs, and we use trainers with average=False.
        lr = lr / 8.
        opt = tf.train.MomentumOptimizer(lr, 0.9)

        opt = optimizer.AccumGradOptimizer(opt, 8 // cfg.TRAIN.NUM_GPUS)  # assume cfg.TRAIN.NUM_GPUS < 8:
        return opt


class ResNetC4Model(DetectionModel):
    def inputs(self):  # OK
        ret = [
            tf.placeholder(tf.float32, (None, None, 3), 'image'),
            # label of each anchor
            tf.placeholder(tf.int32, (None, None, cfg.RPN.NUM_ANCHOR), 'anchor_labels'),  # NUM_ANCHOR = 5*3
            # box of each anchor
            tf.placeholder(tf.float32, (None, None, cfg.RPN.NUM_ANCHOR, 4), 'anchor_boxes'),
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

        mask_logits = maskrcnn_upXconv_head(
            'maskrcnn', feature_fastrcnn, cfg.DATA.NUM_CATEGORY, 0)  # #result x #cat x 14x14
        person_labels = tf.ones_like(inputs['male'])
        indices = tf.stack([tf.range(tf.size(person_labels)), tf.to_int32(person_labels) - 1], axis=1)
        final_mask_logits = tf.gather_nd(mask_logits, indices)  # #resultx14x14
        final_mask_logits = tf.sigmoid(final_mask_logits, name='output/masks')
        final_mask_logits_expand = tf.expand_dims(final_mask_logits, axis=1)
        final_mask_logits_tile = tf.tile(final_mask_logits_expand, multiples=[1, 1024, 1, 1])
        fg_mask_roi_resized = tf.where(final_mask_logits_tile >= 0.5, roi_resized,
                                       roi_resized * 0.0)
        feature_attrs = resnet_conv5(fg_mask_roi_resized,
                                     cfg.BACKBONE.RESNET_NUM_BLOCK[-1])

        feature_gap = GlobalAvgPooling('gap', feature_attrs, data_format='channels_first')  # ??
        #attrs_logits = attrs_head('attrs', feature_gap)
        attrs_labels = attrs_predict(feature_gap)


def predict(pred_func, input_file):
    img = cv2.imread(input_file, cv2.IMREAD_COLOR)
    results = detect_one_image(img, pred_func)
    final = draw_final_outputs(img, results)  # image contain boxes,labels and scores
    viz = np.concatenate((img, final), axis=1)
    tpviz.interactive_imshow(viz)


def detect_one_image(img, model_func):
    """
    Run detection on one image, using the TF callable.
    This function should handle the preprocessing internally.

    Args:
        img: an image
        model_func: a callable from TF model,
            takes image and returns (boxes, probs, labels, [masks])

    Returns:
        [DetectionResult]
    """
    orig_shape = img.shape[:2]
    resizer = CustomResize(cfg.PREPROC.TEST_SHORT_EDGE_SIZE, cfg.PREPROC.MAX_SIZE)
    resized_img = resizer.augment(img)
    scale = np.sqrt(resized_img.shape[0] * 1.0 / img.shape[0] * resized_img.shape[1] / img.shape[1])
    boxes, probs, labels, masks, *attrs = model_func(resized_img)
    boxes = boxes / scale
    # boxes are already clipped inside the graph, but after the floating point scaling, this may not be true any more.
    boxes = clip_boxes(boxes, orig_shape)
    # if masks:
    # has mask
    full_masks = [fill_full_mask(box, mask, orig_shape)
                  for box, mask in zip(boxes, masks)]
    masks = full_masks
    results = [DetectionResult(*args) for args in zip(boxes, probs, labels, masks,
                                                      attrs[0], attrs[1], attrs[2], attrs[3],
                                                      attrs[4], attrs[5], attrs[6], attrs[7],
                                                      attrs[8], attrs[9], attrs[10], attrs[11],
                                                      attrs[12], attrs[13])]
    return results





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='load a model for evaluation. Can overwrite BACKBONE.WEIGHTS')
    parser.add_argument('--predict', help="Run prediction on a given image. "
                                          "This argument is the path to the input image file")
    parser.add_argument('--config', help="A list of KEY=VALUE to overwrite those defined in tensorpack_config.py",
                        nargs='+')

    args = parser.parse_args()
    if args.config:
        cfg.update_args(args.config)

    MODEL = ResNetC4Model()

    if args.predict:
        assert args.load
        finalize_configs(is_training=False)

        pred = OfflinePredictor(PredictConfig(
            model=MODEL,  # model
            session_init=get_model_loader(args.load),  # weight
            input_names=['image', 'gt_boxes'],
            output_names=['output/masks',
                          'male_predict', 'longhair_predict', 'sunglass_predict',
                          'hat_predict', 'tshirt_predict', 'longsleeve_predict',
                          'formal_predict', 'shorts_predict', 'jeans_predict',
                          'skirt_predict', 'facemask_predict', 'logo_predict',
                          'stripe_predict', 'longpants_predict'
                          ]))

        COCODetection(cfg.DATA.BASEDIR, 'val2014')  # load the class names into cfg.DATA.CLASS_NAMES
        predict(pred, args.predict)  # contain vislizaiton
