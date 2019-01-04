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
    image_preprocess, resnet_c4_backbone, resnet_conv5)

from detection.tensorpacks.model_frcnn import (
    sample_fast_rcnn_targets, fastrcnn_outputs, attrs_head,
    fastrcnn_predictions, BoxProposals, FastRCNNHead, attr_losses, all_attrs_losses)
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
            # male_labels of each ground truth
            tf.placeholder(tf.int64, (None,), 'male'),
            # longhair_labels of each ground truth
            tf.placeholder(tf.int64, (None,), 'longhair'),
            # sunglass_labels of each ground truth
            tf.placeholder(tf.int64, (None,), 'sunglass'),
            # hat_labels of each ground truth
            tf.placeholder(tf.int64, (None,), 'hat'),
            # tshort_labels of each ground truth
            tf.placeholder(tf.int64, (None,), 'tshirt'),
            # 6
            tf.placeholder(tf.int64, (None,), 'longsleeve'),
            # 7
            tf.placeholder(tf.int64, (None,), 'formal'),
            # 8
            tf.placeholder(tf.int64, (None,), 'shorts'),
            # 9
            tf.placeholder(tf.int64, (None,), 'jeans'),
            # 10
            tf.placeholder(tf.int64, (None,), 'longpants'),
            # 11
            tf.placeholder(tf.int64, (None,), 'skirt'),
            # 12
            tf.placeholder(tf.int64, (None,), 'facemask'),
            # 13
            tf.placeholder(tf.int64, (None,), 'logo'),
            # 14
            tf.placeholder(tf.int64, (None,), 'stripe')]

        return ret

    def build_graph(self, *inputs):
        mask = True
        if mask:
            inputs = dict(zip(self.input_names, inputs))
            image = self.preprocess(inputs['image'])  # 1CHW
            # build resnet c4
            featuremap = resnet_c4_backbone(image, cfg.BACKBONE.RESNET_NUM_BLOCK[:3])
            # predict attrs b
            boxes_on_featuremap = inputs['gt_boxes'] * (1.0 / cfg.RPN.ANCHOR_STRIDE)  # ANCHOR_STRIDE = 16
            roi_resized = roi_align(featuremap, boxes_on_featuremap, 14)
            feature_maskrcnn = resnet_conv5(roi_resized,
                                            cfg.BACKBONE.RESNET_NUM_BLOCK[
                                                -1])  # nxcx7x7 # RESNET_NUM_BLOCK = [3, 4, 6, 3]
            # Keep C5 feature to be shared with mask branch
            mask_logits = maskrcnn_upXconv_head(
                'maskrcnn', feature_maskrcnn, cfg.DATA.NUM_CATEGORY, 0)  # #result x #cat x 14x14
            # Assume only person here
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
            # build attrs branch

            attrs_logits = attrs_head('attrs', feature_gap)
            attrs_loss = all_attrs_losses(inputs, attrs_logits)

            all_losses = [attrs_loss]
            # male loss
            wd_cost = regularize_cost(
                '.*/W', l2_regularizer(cfg.TRAIN.WEIGHT_DECAY), name='wd_cost')
            all_losses.append(wd_cost)
            total_cost = tf.add_n(all_losses, 'total_cost')

            add_moving_summary(wd_cost, total_cost)
            return total_cost

        else:
            inputs = dict(zip(self.input_names, inputs))
            image = self.preprocess(inputs['image'])  # 1CHW
            # build resnet c4
            featuremap = resnet_c4_backbone(image, cfg.BACKBONE.RESNET_NUM_BLOCK[:3])
            # predict attrs b
            boxes_on_featuremap = inputs['gt_boxes'] * (1.0 / cfg.RPN.ANCHOR_STRIDE)  # ANCHOR_STRIDE = 16
            roi_resized = roi_align(featuremap, boxes_on_featuremap, 14)
            feature_attrs = resnet_conv5(roi_resized,
                                         cfg.BACKBONE.RESNET_NUM_BLOCK[-1])  # nxcx7x7 # RESNET_NUM_BLOCK = [3, 4, 6, 3]
            # Keep C5 feature to be shared with mask branch
            feature_gap = GlobalAvgPooling('gap', feature_attrs, data_format='channels_first')  # ??
            # build attrs branch

            attrs_logits = attrs_head('attrs', feature_gap)
            attrs_loss = all_attrs_losses(inputs, attrs_logits)

            all_losses = [attrs_loss]
            # male loss
            wd_cost = regularize_cost(
                '.*/W', l2_regularizer(cfg.TRAIN.WEIGHT_DECAY), name='wd_cost')
            all_losses.append(wd_cost)
            total_cost = tf.add_n(all_losses, 'total_cost')

            add_moving_summary(wd_cost, total_cost)
            return total_cost


class EvalCallback(Callback):
    """
    A callback that runs COCO evaluation once a while.
    It supports multi-gpu evaluation.
    """

    _chief_only = False

    def __init__(self, in_names, out_names):
        self._in_names, self._out_names = in_names, out_names

    def _setup_graph(self):
        num_gpu = cfg.TRAIN.NUM_GPUS
        if cfg.TRAINER == 'replicated':
            # Use two predictor threads per GPU to get better throughput
            self.num_predictor = num_gpu * 2
            self.predictors = [self._build_coco_predictor(k % num_gpu) for k in range(self.num_predictor)]
            self.dataflows = [get_eval_dataflow(shard=k, num_shards=self.num_predictor)
                              for k in range(self.num_predictor)]
        else:
            # Only eval on the first machine.
            # Alternatively, can eval on all ranks and use allgather, but allgather sometimes hangs
            self._horovod_run_eval = hvd.rank() == hvd.local_rank()
            if self._horovod_run_eval:
                self.predictor = self._build_coco_predictor(0)
                self.dataflow = get_eval_dataflow(shard=hvd.local_rank(), num_shards=hvd.local_size())

            self.barrier = hvd.allreduce(tf.random_normal(shape=[1]))

    def _build_coco_predictor(self, idx):
        graph_func = self.trainer.get_predictor(self._in_names, self._out_names, device=idx)
        return lambda img: detect_one_image(img, graph_func)

    def _before_train(self):
        num_eval = cfg.TRAIN.NUM_EVALS
        interval = max(self.trainer.max_epoch // (num_eval + 1), 1)
        self.epochs_to_eval = set([interval * k for k in range(1, num_eval + 1)])
        self.epochs_to_eval.add(self.trainer.max_epoch)
        if len(self.epochs_to_eval) < 15:
            logger.info("[EvalCallback] Will evaluate at epoch " + str(sorted(self.epochs_to_eval)))
        else:
            logger.info("[EvalCallback] Will evaluate every {} epochs".format(interval))

    def _eval(self):
        logdir = args.logdir
        if cfg.TRAINER == 'replicated':
            with ThreadPoolExecutor(max_workers=self.num_predictor) as executor, \
                    tqdm.tqdm(total=sum([df.size() for df in self.dataflows])) as pbar:
                futures = []
                for dataflow, pred in zip(self.dataflows, self.predictors):
                    futures.append(executor.submit(eval_coco, dataflow, pred, pbar))
                all_results = list(itertools.chain(*[fut.result() for fut in futures]))
        else:
            if self._horovod_run_eval:
                local_results = eval_coco(self.dataflow, self.predictor)
                output_partial = os.path.join(
                    logdir, 'outputs{}-part{}.json'.format(self.global_step, hvd.local_rank()))
                with open(output_partial, 'w') as f:
                    json.dump(local_results, f)
            self.barrier.eval()
            if hvd.rank() > 0:
                return
            all_results = []
            for k in range(hvd.local_size()):
                output_partial = os.path.join(
                    logdir, 'outputs{}-part{}.json'.format(self.global_step, k))
                with open(output_partial, 'r') as f:
                    obj = json.load(f)
                all_results.extend(obj)
                os.unlink(output_partial)

        output_file = os.path.join(
            logdir, 'outputs{}.json'.format(self.global_step))
        with open(output_file, 'w') as f:
            json.dump(all_results, f)
        try:
            scores = print_evaluation_scores(output_file)
            for k, v in scores.items():
                self.trainer.monitors.put_scalar(k, v)
        except Exception:
            logger.exception("Exception in COCO evaluation.")

    def _trigger_epoch(self):
        if self.epoch_num in self.epochs_to_eval:
            logger.info("Running evaluation ...")
            self._eval()


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
    stepnum = cfg.TRAIN.STEPS_PER_EPOCH  # STEPS_PER_EPOCH = 500
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
    total_passes = cfg.TRAIN.LR_SCHEDULE[-1] * 8 / train_attrs_dataflow.size()
    logger.info("Total passes of the training set is: {}".format(total_passes))
    callbacks = [
        PeriodicCallback(
            ModelSaver(max_to_keep=10, keep_checkpoint_every_n_hours=1),
            every_k_epochs=20),
        # linear warmup
        ScheduledHyperParamSetter(
            'learning_rate', warmup_schedule, interp='linear', step_based=True),
        ScheduledHyperParamSetter('learning_rate', lr_schedule),
        # EvalCallback(*MODEL.get_inference_tensor_names()),
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
