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
from detection.tensorpacks.wider_attr import load_many
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
    resnet_fpn_backbone, resnet_conv5_attr)

from detection.tensorpacks import model_frcnn
from detection.tensorpacks import model_mrcnn
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
    def inputs(self):  # OK
        ret = [
            tf.placeholder(tf.float32, (None, None, 3), 'image'),
            # label of each anchor
            tf.placeholder(tf.int32, (None, None, cfg.RPN.NUM_ANCHOR), 'anchor_labels'),  # NUM_ANCHOR = 5*3
            # box of each anchor
            tf.placeholder(tf.float32, (None, None, cfg.RPN.NUM_ANCHOR, 4), 'anchor_boxes'),
            # male_labels of each ground truth
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

        # ROI_align
        x, y, w, h = tf.split(inputs['gt_boxes'], 4, axis=1)
        gt_boxes = tf.concat([x, y, x + w, y + h], axis=1)
        boxes_on_featuremap = gt_boxes * (1.0 / cfg.RPN.ANCHOR_STRIDE)  # ANCHOR_STRIDE = 16

        roi_resized = roi_align(featuremap, boxes_on_featuremap, 14)  # 14x14 for each roi

        person_labels = tf.ones_like(inputs['male'])
        feature_maskrcnn = resnet_conv5(roi_resized, cfg.BACKBONE.RESNET_NUM_BLOCK[-1])
        mask_logits = maskrcnn_upXconv_head(
            'maskrcnn', feature_maskrcnn, cfg.DATA.NUM_CATEGORY, 0)  # #result x #cat x 14x14
        indices = tf.stack([tf.range(tf.size(person_labels)), tf.to_int32(person_labels) - 1], axis=1)
        final_mask_logits = tf.gather_nd(mask_logits, indices)  # #resultx14x14
        final_mask_logits = tf.sigmoid(final_mask_logits, name='output/masks')

        final_mask_logits_expand = tf.expand_dims(final_mask_logits, axis=1)
        final_mask_logits_tile = tf.tile(final_mask_logits_expand, multiples=[1, 1024, 1, 1])
        fg_mask_roi_resized = tf.where(final_mask_logits_tile >= 0.5, roi_resized,
                                       roi_resized * 1.0)
        feature_attrs = resnet_conv5_attr(fg_mask_roi_resized,
                                          cfg.BACKBONE.RESNET_NUM_BLOCK[-1])

        feature_gap = GlobalAvgPooling('gap', feature_attrs, data_format='channels_first')  # ??
        # attrs_logits = attrs_head('attrs', feature_gap)
        attrs_labels = attrs_predict(feature_gap)


def predict(pred_func, input_file):
    img = cv2.imread(input_file, cv2.IMREAD_COLOR)
    results = detect_one_image(img, pred_func)
    final = draw_final_outputs(img, results)  # image contain boxes,labels and scores
    viz = np.concatenate((img, final), axis=1)
    tpviz.interactive_imshow(viz)


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

        # can't input the dataflow ?
        pred = OfflinePredictor(PredictConfig(
            model=MODEL,  # model
            session_init=get_model_loader(args.load),  # weight
            input_names=['image', 'male', 'gt_boxes'],
            output_names=['male_predict', 'longhair_predict', 'sunglass_predict',
                          'hat_predict', 'tshirt_predict', 'longsleeve_predict',
                          'formal_predict', 'shorts_predict', 'jeans_predict',
                          'skirt_predict', 'facemask_predict', 'logo_predict',
                          'stripe_predict', 'longpants_predict'
                          ]))
        male_predict = []
        male_label = []

        longhair_predict = []
        longhair_label = []

        sunglass_predict = []
        sunglass_label = []

        hat_predict = []
        hat_label = []

        tshirt_predict = []
        tshirt_label = []

        longsleeve_predict = []
        longsleeve_label = []

        formal_predict = []
        formal_label = []

        shorts_predict = []
        shorts_label = []

        jeans_predict = []
        jeans_label = []

        skirt_predict = []
        skirt_label = []

        facemask_predict = []
        facemask_label = []

        logo_predict = []
        logo_label = []

        stripe_predict = []
        stripe_label = []

        longpants_predict = []
        longpants_label = []

        roidbs_val = load_many('/root/datasets/wider attribute', 'val')
        for val in roidbs_val:
            img = cv2.imread(val['img'])
            results = pred(img, val['male'], val['bbox'])

            male_predict = np.append(male_predict, results[0])
            male_label = np.append(male_label, val['male'])
            longhair_predict = np.append(longhair_predict, results[1])
            longhair_label = np.append(longhair_label, val['longhair'])
            sunglass_predict = np.append(sunglass_predict, results[2])
            sunglass_label = np.append(sunglass_label, val['sunglass'])
            hat_predict = np.append(hat_predict, results[3])
            hat_label = np.append(hat_label, val['hat'])
            tshirt_predict = np.append(tshirt_predict, results[4])
            tshirt_label = np.append(tshirt_label, val['tshirt'])
            longsleeve_predict = np.append(longsleeve_predict, results[5])
            longsleeve_label = np.append(longsleeve_label, val['longsleeve'])
            formal_predict = np.append(formal_predict, results[6])
            formal_label = np.append(formal_label, val['formal'])
            shorts_predict = np.append(shorts_predict, results[7])
            shorts_label = np.append(shorts_label, val['shorts'])
            jeans_predict = np.append(jeans_predict, results[8])
            jeans_label = np.append(jeans_label, val['jeans'])
            skirt_predict = np.append(skirt_predict, results[9])
            skirt_label = np.append(skirt_label, val['skirt'])
            facemask_predict = np.append(facemask_predict, results[10])
            facemask_label = np.append(facemask_label, val['facemask'])

            logo_predict = np.append(logo_predict, results[11])
            logo_label = np.append(logo_label, val['logo'])
            stripe_predict = np.append(stripe_predict, results[12])
            stripe_label = np.append(stripe_label, val['stripe'])
            longpants_predict = np.append(longpants_predict, results[13])
            longpants_label = np.append(longpants_label, val['longpants'])

        correct_male = np.equal(male_predict, male_label).astype(np.float32)
        accuracy_male = np.mean(correct_male)

        correct_longhair = np.equal(longhair_predict, longhair_label).astype(np.float32)
        accuracy_longhair = np.mean(correct_male)

        correct_sunglass = np.equal(sunglass_predict, sunglass_label).astype(np.float32)
        accuracy_sunglass = np.mean(correct_sunglass)

        correct_hat = np.equal(hat_predict, hat_label).astype(np.float32)
        accuracy_hat = np.mean(correct_hat)

        correct_tshirt = np.equal(tshirt_predict, tshirt_label).astype(np.float32)
        accuracy_tshirt = np.mean(correct_tshirt)

        correct_longsleeve = np.equal(longsleeve_predict, longsleeve_label).astype(np.float32)
        accuracy_longsleeve = np.mean(correct_longsleeve)

        correct_formal = np.equal(formal_predict, formal_label).astype(np.float32)
        accuracy_formal = np.mean(correct_formal)

        correct_shorts = np.equal(shorts_predict, shorts_label).astype(np.float32)
        accuracy_shorts = np.mean(correct_shorts)

        correct_jeans = np.equal(jeans_predict, jeans_label).astype(np.float32)
        accuracy_jeans = np.mean(correct_jeans)

        correct_skirt = np.equal(skirt_predict,skirt_label).astype(np.float32)
        accuracy_skirt = np.mean(correct_skirt)

        correct_facemask = np.equal(facemask_predict, facemask_label).astype(np.float32)
        accuracy_facemask = np.mean(correct_facemask)

        correct_logo = np.equal(logo_predict, logo_label).astype(np.float32)
        accuracy_logo = np.mean(correct_logo)

        correct_stripe = np.equal(stripe_predict, stripe_label).astype(np.float32)
        accuracy_stripe = np.mean(correct_stripe)

        correct_longpants = np.equal(longpants_predict, longpants_label).astype(np.float32)
        accuracy_longpants = np.mean(correct_longpants)

        COCODetection(cfg.DATA.BASEDIR, 'val2014')  # load the class names into cfg.DATA.CLASS_NAMES
        predict(pred, args.predict)  # contain vislizaiton

'''
--config
DATA.BASEDIR=/root/datasets/COCO/DIR
--predict
/root/Figma_RCNN/detection/test/celebrities.jpg
--load
/root/datasets/COCO-R50C4-MaskRCNN-Standard.npz
'''
# if __name__ == '__main__':
#     MODEL = ResNetC4Model()
#
#     pred = OfflinePredictor(PredictConfig(
#         model=MODEL,  # model
#         session_init=get_model_loader('/root/datasets/COCO-R50C4-MaskRCNN-Standard.npz'),  # weight
#         input_names=['image', 'gt_boxes'],
#         output_names=['image', 'gt_boxes'
#                       ]))
#     roidbs_val = load_many('/root/datasets/wider attribute', 'val')
#
#     img = cv2.imread('/root/datasets/wider attribute/train/1--Handshaking/1_Handshaking_Handshaking_1_765.jpg')
#     boxes = np.array([(93.06605, 121.849365, 380.8992, 593.8957),
#                       (477.803, 86.349945, 476.84357, 634.1923)])
#
#     prediction = pred(img, boxes)
#
#     img = (prediction[3].transpose(0, 2, 3, 1)[0][:, :, 1:1024:500] * 100).astype(np.uint8)
#     vis_one_image(img, prediction[2], 16)
