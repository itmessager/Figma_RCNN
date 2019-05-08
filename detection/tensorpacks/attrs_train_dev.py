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

from detection.tensorpacks.model_mrcnn import maskrcnn_upXconv_head
from detection.tensorpacks.model_rpn import rpn_head, generate_rpn_proposals

try:
    import horovod.tensorflow as hvd
except ImportError:
    pass

import tensorflow as tf

assert six.PY3, "FasterRCNN requires Python 3!"

from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils import optimizer

from detection.tensorpacks.coco import COCODetection
from detection.tensorpacks.basemodel import (
    image_preprocess, resnet_c4_backbone, resnet_conv5, resnet_conv5_attr)

from detection.tensorpacks.model_frcnn import (
    sample_fast_rcnn_targets, fastrcnn_outputs, attrs_head,
    fastrcnn_predictions, BoxProposals, FastRCNNHead, attr_losses, attr_losses_v2, all_attrs_losses)
from detection.tensorpacks.model_box import (
    clip_boxes, crop_and_resize, roi_align, RPNAnchors)

from detection.tensorpacks.data import (
    get_train_dataflow, get_eval_dataflow,
    get_all_anchors, get_all_anchors_fpn, get_attributes_dataflow)

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

        # I haved changed the learning rate
        opt = tf.train.MomentumOptimizer(lr, 0.9)

        opt = optimizer.AccumGradOptimizer(opt, 8 // cfg.TRAIN.NUM_GPUS)  # assume cfg.TRAIN.NUM_GPUS < 8:
        return opt


class ResNetC4Model(DetectionModel):
    def inputs(self):  # OK
        ret = [
            tf.placeholder(tf.float32, (None, None, 3), 'image'),
            # box of each ground truth
            tf.placeholder(tf.float32, (None, 4), 'gt_boxes'),
            # label of each anchor
            tf.placeholder(tf.int32, (None, None, cfg.RPN.NUM_ANCHOR), 'anchor_labels'),  # NUM_ANCHOR = 5*3
            # box of each anchor
            tf.placeholder(tf.float32, (None, None, cfg.RPN.NUM_ANCHOR, 4), 'anchor_boxes'),
            # attributes labels of each ground truth
            tf.placeholder(tf.int64, (None,), 'male'),
            tf.placeholder(tf.int64, (None,), 'longhair'),
            tf.placeholder(tf.int64, (None,), 'sunglass'),
            tf.placeholder(tf.int64, (None,), 'hat'),
            tf.placeholder(tf.int64, (None,), 'tshirt'),
            tf.placeholder(tf.int64, (None,), 'longsleeve'),
            tf.placeholder(tf.int64, (None,), 'formal'),
            tf.placeholder(tf.int64, (None,), 'shorts'),
            tf.placeholder(tf.int64, (None,), 'jeans'),
            tf.placeholder(tf.int64, (None,), 'longpants'),
            tf.placeholder(tf.int64, (None,), 'skirt'),
            tf.placeholder(tf.int64, (None,), 'facemask'),
            tf.placeholder(tf.int64, (None,), 'logo'),
            tf.placeholder(tf.int64, (None,), 'stripe')]
        return ret

    def build_graph(self, *inputs):
        inputs = dict(zip(self.input_names, inputs))
        image = self.preprocess(inputs['image'])  # 1CHW
        # build resnet c4
        featuremap = resnet_c4_backbone(image, cfg.BACKBONE.RESNET_NUM_BLOCK[:3])

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
            cfg.RPN.TEST_PRE_NMS_TOPK,  # 2000
            cfg.RPN.TEST_POST_NMS_TOPK)  # 1000
        x, y, w, h = tf.split(inputs['gt_boxes'], 4, axis=1)
        gt_boxes = tf.concat([x, y, x + w, y + h], axis=1)
        boxes_on_featuremap = gt_boxes * (1.0 / cfg.RPN.ANCHOR_STRIDE)  # ANCHOR_STRIDE = 16
        roi_resized = roi_align(featuremap, boxes_on_featuremap, 14)

        feature_fastrcnn = resnet_conv5(roi_resized,
                                        cfg.BACKBONE.RESNET_NUM_BLOCK[
                                            -1])  # nxcx7x7 # RESNET_NUM_BLOCK = [3, 4, 6, 3]
        # Keep C5 feature to be shared with mask branch
        feature_gap = GlobalAvgPooling('gap', feature_fastrcnn, data_format='channels_first')  # ??

        fastrcnn_label_logits, fastrcnn_box_logits = fastrcnn_outputs('fastrcnn', feature_gap,
                                                                      cfg.DATA.NUM_CLASS)  # ??
        # Returns:
        # cls_logits: Tensor("fastrcnn/class/output:0", shape=(n, 81), dtype=float32)
        # reg_logits: Tensor("fastrcnn/output_box:0", shape=(n, 81, 4), dtype=float32)

        # ------------------Fastrcnn_Head------------------------
        fastrcnn_head = FastRCNNHead(proposal_boxes, fastrcnn_box_logits, fastrcnn_label_logits,  #
                                     tf.constant(cfg.FRCNN.BBOX_REG_WEIGHTS,
                                                 dtype=tf.float32))  # [10., 10., 5., 5.]

        decoded_boxes = fastrcnn_head.decoded_output_boxes()  # pre_boxes_on_images
        decoded_boxes = clip_boxes(decoded_boxes, image_shape2d, name='fastrcnn_all_boxes')

        label_scores = tf.nn.softmax(fastrcnn_label_logits, name='fastrcnn_all_scores')
        # class scores, summed to one for each box.

        final_boxes, final_scores, final_labels = fastrcnn_predictions(
            decoded_boxes, label_scores, name_scope='output')

        # attributes branch
        feature_attributes = resnet_conv5_attr(roi_resized, cfg.BACKBONE.RESNET_NUM_BLOCK[-1])
        feature_attrs_gap = GlobalAvgPooling('gap', feature_attributes, data_format='channels_first')
        add_conv = False
        if add_conv:
            attrs_logits = attrs_head('attrs', feature_attributes)
        else:
            attrs_logits = attrs_head('attrs', feature_attrs_gap)
        attrs_loss = all_attrs_losses(inputs, attrs_logits, attr_losses_v2)

        all_losses = [attrs_loss]
        # male loss
        wd_cost = regularize_cost(
            '.*/W', l2_regularizer(cfg.TRAIN.WEIGHT_DECAY), name='wd_cost')
        all_losses.append(wd_cost)
        total_cost = tf.add_n(all_losses, 'total_cost')
        add_moving_summary(wd_cost, total_cost)

        return total_cost


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='load a model for evaluation. Can overwrite BACKBONE.WEIGHTS')
    parser.add_argument('--logdir', help='log directory', default='train_log/maskrcnn')
    parser.add_argument('--config', help="A list of KEY=VALUE to overwrite those defined in tensorpack_config.py",
                        nargs='+')
    args = parser.parse_args()
    if args.config:
        cfg.update_args(args.config)

    MODEL = ResNetC4Model()

    is_horovod = cfg.TRAINER == 'horovod'
    if is_horovod:
        hvd.init()
        logger.info("Horovod Rank={}, Size={}".format(hvd.rank(), hvd.size()))

    if not is_horovod or hvd.rank() == 0:
        logger.set_logger_dir(args.logdir, 'd')

    finalize_configs(is_training=True)
    stepnum = cfg.TRAIN.STEPS_PER_EPOCH  # STEPS_PER_EPOCH = 5000
    # warmup is step based, lr is epoch based
    init_lr = cfg.TRAIN.BASE_LR * 0.33 * min(8. / cfg.TRAIN.NUM_GPUS, 1.)
    warmup_schedule = [(0, init_lr), (cfg.TRAIN.WARMUP, cfg.TRAIN.BASE_LR)]
    warmup_end_epoch = cfg.TRAIN.WARMUP * 1. / stepnum  # 1000/500
    lr_schedule = [(int(warmup_end_epoch + 0.5), cfg.TRAIN.BASE_LR)]

    factor = 8. / cfg.TRAIN.NUM_GPUS
    for idx, steps in enumerate(cfg.TRAIN.LR_SCHEDULE[:-1]):
        mult = 0.1 ** (idx + 1)
        lr_schedule.append(
            (steps * factor // stepnum, cfg.TRAIN.BASE_LR * mult))
    logger.info("Warm Up Schedule (steps, value): " + str(warmup_schedule))
    logger.info("LR Schedule (epochs, value): " + str(lr_schedule))
    # train_dataflow = get_train_dataflow()   # get the coco datasets

    train_attrs_dataflow = get_attributes_dataflow()  # get the wider datasets
    # This is what's commonly referred to as "epochs"
    total_passes = cfg.TRAIN.LR_SCHEDULE[-1] * factor / train_attrs_dataflow.size()
    logger.info("Total passes of the training set is: {}".format(total_passes))
    callbacks = [
        PeriodicCallback(
            ModelSaver(max_to_keep=10, keep_checkpoint_every_n_hours=1),
            every_k_epochs=20),
        # linear warmup
        ScheduledHyperParamSetter(
            'learning_rate', warmup_schedule, interp='linear', step_based=True),
        ScheduledHyperParamSetter('learning_rate', lr_schedule),
        PeakMemoryTracker(),
        EstimatedTimeLeft(median=True),
        SessionRunTimeout(60000).set_chief_only(True),  # 1 minute timeout
    ]
    if not is_horovod:
        callbacks.append(GPUUtilizationTracker())

    if is_horovod and hvd.rank() > 0:
        session_init = None
    else:
        if args.load:
            session_init = get_model_loader(args.load)
        else:
            session_init = get_model_loader(cfg.BACKBONE.WEIGHTS) if cfg.BACKBONE.WEIGHTS else None

    traincfg = TrainConfig(
        model=MODEL,
        data=QueueInput(train_attrs_dataflow),
        callbacks=callbacks,
        steps_per_epoch=stepnum,
        max_epoch=cfg.TRAIN.LR_SCHEDULE[-1] * factor // stepnum,
        session_init=session_init,
    )

    if is_horovod:
        trainer = HorovodTrainer(average=False)
    else:
        # nccl mode has better speed than cpu mode
        trainer = SyncMultiGPUTrainerReplicated(cfg.TRAIN.NUM_GPUS, average=False, mode='nccl')

    launch_train_with_config(traincfg, trainer)
