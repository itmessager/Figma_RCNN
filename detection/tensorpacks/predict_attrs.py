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

from concurrent.futures import ThreadPoolExecutor

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

from detection.tensorpacks import model_frcnn
from detection.tensorpacks import model_mrcnn
from detection.tensorpacks.model_frcnn import (
    sample_fast_rcnn_targets, fastrcnn_outputs,
    fastrcnn_predictions, BoxProposals, FastRCNNHead, attrs_head, all_attrs_losses)
from detection.tensorpacks.model_mrcnn import maskrcnn_upXconv_head, maskrcnn_loss
from detection.tensorpacks.model_rpn import rpn_head, rpn_losses, generate_rpn_proposals
from detection.tensorpacks.model_fpn import (
    fpn_model, multilevel_roi_align,
    generate_fpn_proposals)
from detection.tensorpacks.model_cascade import CascadeRCNNHead
from detection.tensorpacks.model_box import (
    clip_boxes, crop_and_resize, roi_align, RPNAnchors)

from detection.tensorpacks.data import (
    get_train_dataflow, get_eval_dataflow,get_attributes_dataflow,
    get_all_anchors, get_all_anchors_fpn)
from detection.tensorpacks.viz import (
    draw_annotation, draw_proposal_recall,
    draw_predictions, draw_final_outputs)
from detection.tensorpacks.eval import (
    eval_coco, detect_one_image, print_evaluation_scores, DetectionResult)
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
    def inputs(self): # OK
        ret = [
            tf.placeholder(tf.float32, (None, None, 3), 'image'),
            # label of each anchor
            tf.placeholder(tf.int32, (None, None, cfg.RPN.NUM_ANCHOR), 'anchor_labels'), # NUM_ANCHOR = 5*3
            # box of each anchor
            tf.placeholder(tf.float32, (None, None, cfg.RPN.NUM_ANCHOR, 4), 'anchor_boxes'),
            # box of each ground truth
            tf.placeholder(tf.float32, (None, 4), 'gt_boxes'),
            # label of each ground truth
            #tf.placeholder(tf.int64, (None,), 'gt_labels'), # all > 0

            # male_labels of each ground truth
            tf.placeholder(tf.int64, (None,), 'male')]  # all > 0

            # mask of each ground truth
            #tf.placeholder(tf.uint8, (None, None, None), 'gt_masks')]# NR_GT x height x width

        return ret

    def build_graph(self, *inputs):
        inputs = dict(zip(self.input_names, inputs))
        image = self.preprocess(inputs['image'])  # 1CHW

        # build resnet c4
        featuremap = resnet_c4_backbone(image, cfg.BACKBONE.RESNET_NUM_BLOCK[:3])

        # build rpn
        rpn_label_logits, rpn_box_logits = rpn_head('rpn', featuremap, cfg.RPN.HEAD_DIM, cfg.RPN.NUM_ANCHOR)
        # HEAD_DIM = 1024, NUM_ANCHOR = 15
        # rpn_label_logits: fHxfWxNA
        # rpn_box_logits: fHxfWxNAx4
        anchors = RPNAnchors(get_all_anchors(), inputs['anchor_labels'], inputs['anchor_boxes'])
        # anchor_boxes is Groundtruth boxes corresponding to each anchor
        anchors = anchors.narrow_to(featuremap)  # ??
        image_shape2d = tf.shape(image)[2:]  # h,w
        pred_boxes_decoded = anchors.decode_logits(rpn_box_logits)  # fHxfWxNAx4, floatbox

        # ProposalCreator (get the topk proposals)
        proposal_boxes, proposal_scores = generate_rpn_proposals(
            tf.reshape(pred_boxes_decoded, [-1, 4]),
            tf.reshape(rpn_label_logits, [-1]),
            image_shape2d,
            cfg.RPN.TEST_PRE_NMS_TOPK,    # 2000
            cfg.RPN.TEST_POST_NMS_TOPK)  # 1000

        boxes_on_featuremap = proposal_boxes * (1.0 / cfg.RPN.ANCHOR_STRIDE) # ANCHOR_STRIDE = 16

        # ROI_align
        roi_resized = roi_align(featuremap, boxes_on_featuremap, 14)   #14x14 for each roi

        feature_fastrcnn = resnet_conv5(roi_resized, cfg.BACKBONE.RESNET_NUM_BLOCK[-1])  # nxcx7x7 # RESNET_NUM_BLOCK = [3, 4, 6, 3]
        # Keep C5 feature to be shared with mask branch
        feature_gap = GlobalAvgPooling('gap', feature_fastrcnn, data_format='channels_first') # ??
        # build my net branch!
        attrs_logits = attrs_head('attrs', feature_gap)
        #attrs_loss = all_attrs_losses(inputs, attrs_logits)

        fastrcnn_label_logits, fastrcnn_box_logits = fastrcnn_outputs('fastrcnn', feature_gap, cfg.DATA.NUM_CLASS) # ??
        # Returns:
        # cls_logits: Tensor("fastrcnn/class/output:0", shape=(n, 81), dtype=float32)
        # reg_logits: Tensor("fastrcnn/output_box:0", shape=(n, 81, 4), dtype=float32)

        # ------------------Fastrcnn_Head------------------------
        fastrcnn_head = FastRCNNHead(proposal_boxes, fastrcnn_box_logits, fastrcnn_label_logits,  #
                                     tf.constant(cfg.FRCNN.BBOX_REG_WEIGHTS, dtype=tf.float32))  # [10., 10., 5., 5.]

        decoded_boxes = fastrcnn_head.decoded_output_boxes() # pre_boxes_on_images
        decoded_boxes = clip_boxes(decoded_boxes, image_shape2d, name='fastrcnn_all_boxes')


        label_scores = tf.nn.softmax(fastrcnn_label_logits, name='fastrcnn_all_scores')
        # class scores, summed to one for each box.

        final_boxes, final_scores, final_labels = fastrcnn_predictions(
            decoded_boxes, label_scores, name_scope='output')

        # Mask
        roi_resized = roi_align(featuremap, final_boxes * (1.0 / cfg.RPN.ANCHOR_STRIDE), 14)
        feature_maskrcnn = resnet_conv5(roi_resized, cfg.BACKBONE.RESNET_NUM_BLOCK[-1])
        mask_logits = maskrcnn_upXconv_head(
            'maskrcnn', feature_maskrcnn, cfg.DATA.NUM_CATEGORY, 0)  # #result x #cat x 14x14
        indices = tf.stack([tf.range(tf.size(final_labels)), tf.to_int32(final_labels) - 1], axis=1)
        final_mask_logits = tf.gather_nd(mask_logits, indices)  # #resultx14x14
        tf.sigmoid(final_mask_logits, name='output/masks')


def predict(pred_func, input_file):

    img = cv2.imread(input_file, cv2.IMREAD_COLOR)
    results = detect_one_image(img, pred_func)
    final = draw_final_outputs(img, results)  # image contain boxes,labels and scores
    viz = np.concatenate((img, final), axis=1)
    tpviz.interactive_imshow(viz)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='load a model for evaluation. Can overwrite BACKBONE.WEIGHTS')
    parser.add_argument('--logdir', help='log directory', default='train_log/maskrcnn')
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

        # can't input the dataflow ?
        pred = OfflinePredictor(PredictConfig(
            model=MODEL,  # model
            session_init=get_model_loader(args.load), # weight
            input_names=['image'],
            output_names=['output/boxes', 'output/scores', 'output/labels', 'output/masks']))

        COCODetection(cfg.DATA.BASEDIR,'val2014')  # load the class names into cfg.DATA.CLASS_NAMES
        predict(pred, args.predict) # contain vislizaiton

