#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train.py
from collections import namedtuple
import argparse
import cv2
import tqdm
import numpy as np
import json
import six
import os
from contextlib import ExitStack
from detection.tensorpacks.common import CustomResize
from tensorpack.utils.utils import get_tqdm_kwargs
from sklearn.metrics import average_precision_score
from detection.config.tensorpack_config import finalize_configs, config as cfg

try:
    import horovod.tensorflow as hvd
except ImportError:
    pass

import tensorflow as tf

assert six.PY3, "FasterRCNN requires Python 3!"

from tensorpack.tfutils.common import get_tf_version_tuple

from detection.tensorpacks.data import (
    get_train_dataflow, get_eval_dataflow,
    get_all_anchors, get_all_anchors_fpn, get_wider_dataflow, get_wider_eval_dataflow)

from tensorpack import *
from tensorpack.tfutils import optimizer
import tensorpack.utils.viz as tpviz
from detection.tensorpacks.basemodel import (
    image_preprocess, resnet_c4_backbone, resnet_conv5, resnet_conv5_attr)

from detection.tensorpacks.model_frcnn import (fastrcnn_outputs,
                                               fastrcnn_predictions, BoxProposals, FastRCNNHead, attrs_predict,
                                               logits_to_predict_v2,
                                               logits_to_predict)

from detection.tensorpacks.model_rpn import rpn_head, generate_rpn_proposals
from detection.tensorpacks.model_box import (
    clip_boxes, roi_align, RPNAnchors)


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
        if cfg.TRAIN.NUM_GPUS < 8:
            opt = optimizer.AccumGradOptimizer(opt, 8 // cfg.TRAIN.NUM_GPUS)
        return opt


class ResNetC4Model(DetectionModel):
    def inputs(self):  # OK
        ret = [
            tf.placeholder(tf.float32, (None, None, 3), 'image'),
            # label of each anchor
            tf.placeholder(tf.int32, (None, None, cfg.RPN.NUM_ANCHOR), 'anchor_labels'),  # NUM_ANCHOR = 5*3
            # box of each anchor
            tf.placeholder(tf.float32, (None, None, cfg.RPN.NUM_ANCHOR, 4), 'anchor_boxes'),
            # box of each ground truth
            tf.placeholder(tf.float32, (None, 4), 'gt_boxes'),
            # 14 attributes of each ground truth
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
            tf.placeholder(tf.int64, (None,), 'stripe')
        ]
        return ret

    def build_graph(self, *inputs):
        inputs = dict(zip(self.input_names, inputs))
        image = self.preprocess(inputs['image'])  # 1CHW

        # build resnet c4
        featuremap = resnet_c4_backbone(image, cfg.BACKBONE.RESNET_NUM_BLOCK[:3])
        #
        # # build rpn
        # rpn_label_logits, rpn_box_logits = rpn_head('rpn', featuremap, cfg.RPN.HEAD_DIM, cfg.RPN.NUM_ANCHOR)
        # # HEAD_DIM = 1024, NUM_ANCHOR = 15
        # # rpn_label_logits: fHxfWxNA
        # # rpn_box_logits: fHxfWxNAx4
        # anchors = RPNAnchors(get_all_anchors(), inputs['anchor_labels'], inputs['anchor_boxes'])
        # # anchor_boxes is Groundtruth boxes corresponding to each anchor
        # anchors = anchors.narrow_to(featuremap)
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
        #
        # boxes_on_featuremap = proposal_boxes * (1.0 / cfg.RPN.ANCHOR_STRIDE)  # ANCHOR_STRIDE = 16
        #
        # # ROI_align
        # roi_resized = roi_align(featuremap, boxes_on_featuremap, 14)  # 14x14 for each roi
        #
        # feature_fastrcnn = resnet_conv5(roi_resized,
        #                                 cfg.BACKBONE.RESNET_NUM_BLOCK[-1])  # nxcx7x7 # RESNET_NUM_BLOCK = [3, 4, 6, 3]
        # # Keep C5 feature to be shared with mask branch
        # feature_gap = GlobalAvgPooling('gap', feature_fastrcnn, data_format='channels_first')
        #
        # fastrcnn_label_logits, fastrcnn_box_logits = fastrcnn_outputs('fastrcnn', feature_gap, cfg.DATA.NUM_CLASS)
        # # Returns:
        # # cls_logits: Tensor("fastrcnn/class/output:0", shape=(n, 81), dtype=float32)
        # # reg_logits: Tensor("fastrcnn/output_box:0", shape=(n, 81, 4), dtype=float32)
        #
        # # ------------------Fastrcnn_Head------------------------
        # proposals = BoxProposals(proposal_boxes)
        # fastrcnn_head = FastRCNNHead(proposals, fastrcnn_box_logits, fastrcnn_label_logits,  #
        #                              tf.constant(cfg.FRCNN.BBOX_REG_WEIGHTS, dtype=tf.float32))  # [10., 10., 5., 5.]
        #
        # decoded_boxes = fastrcnn_head.decoded_output_boxes()  # pre_boxes_on_images
        # decoded_boxes = clip_boxes(decoded_boxes, image_shape2d, name='fastrcnn_all_boxes')
        #
        # label_scores = tf.nn.softmax(fastrcnn_label_logits, name='fastrcnn_all_scores')
        # # class scores, summed to one for each box.
        #
        # final_boxes, final_scores, final_labels = fastrcnn_predictions(
        #     decoded_boxes, label_scores, name_scope='output')
        #
        # person_slice = tf.where(final_labels <= 1)
        # person_labels = tf.gather(final_labels, person_slice)
        # final_person_labels = tf.reshape(person_labels, (-1,), name='person_labels')
        #
        # person_boxes = tf.gather(final_boxes, person_slice)
        # final_person_boxes = tf.reshape(person_boxes, (-1, 4), name='person_boxes')
        #
        # person_scores = tf.gather(final_scores, person_slice)
        # tf.reshape(person_scores, (-1,), name='person_scores')
        #
        # # Attributes branch
        x1, y1, w, h = tf.split(inputs['gt_boxes'], 4, axis=1)
        gt_boxes = tf.concat([x1, y1, x1 + w, y1 + h], axis=1)
        person_roi_resized = roi_align(featuremap, gt_boxes * (1.0 / cfg.RPN.ANCHOR_STRIDE), 14)
        feature_attrs = resnet_conv5_attr(person_roi_resized, cfg.BACKBONE.RESNET_NUM_BLOCK[-1])
        feature_attrs_gap = GlobalAvgPooling('gap', feature_attrs, data_format='channels_first')  #
        attrs_predict(feature_attrs_gap)


def eval_W(df, eval_one_image, tqdm_bar=None):
    df.reset_state()
    all_results = []
    with ExitStack() as stack:
        if tqdm_bar is None:
            tqdm_bar = stack.enter_context(
                tqdm.tqdm(total=df.size(), **get_tqdm_kwargs()))
        for img, boxes, img_id, male, longhair, sunglass, hat, tshirt, longsleeve, \
            formal, shorts, jeans, skirt, facemask, logo, stripe, longpants in df:
            results = eval_one_image(img, boxes, male, longhair, sunglass, hat, tshirt, longsleeve,
                                     formal, shorts, jeans, skirt, facemask, logo, stripe, longpants)

            male, longhair, sunglass, hat, tshirt, longsleeve, \
            formal, shorts, jeans, skirt, facemask, logo, stripe, longpants, \
            male_predict, longhair_predict, sunglass_predict, hat_predict, tshirt_predict, \
            longsleeve_predict, formal_predict, shorts_predict, jeans_predict, skirt_predict, \
            facemask_predict, logo_predict, stripe_predict, longpants_predict = list(
                zip(*[[int(r.male), int(r.longhair), int(r.sunglass), int(r.hat), int(r.tshirt), int(r.longsleeve),
                       int(r.formal), int(r.shorts), int(r.jeans), int(r.skirt), int(r.facemask), int(r.logo),
                       int(r.stripe), int(r.longpants), float(r.male_predict), float(r.longhair_predict),
                       float(r.sunglass_predict), float(r.hat_predict), float(r.tshirt_predict),
                       float(r.longsleeve_predict),
                       float(r.formal_predict), float(r.shorts_predict), float(r.jeans_predict), float(r.skirt_predict),
                       float(r.facemask_predict), float(r.logo_predict),
                       float(r.stripe_predict), float(r.longpants_predict)]
                      for r in results]))

            res = {
                'image_id': int(img_id),
                'male': male,
                'longhair': longhair,
                'sunglass': sunglass,
                'hat': hat,
                'tshirt': tshirt,
                'longsleeve': longsleeve,
                'formal': formal,
                'shorts': shorts,
                'jeans': jeans,
                'skirt': skirt,
                'facemask': facemask,
                'logo': logo,
                'stripe': stripe,
                'longpants': longpants,

                'male_predict': male_predict,
                'longhair_predict': longhair_predict,
                'sunglass_predict': sunglass_predict,
                'hat_predict': hat_predict,
                'tshirt_predict': tshirt_predict,
                'longsleeve_predict': longsleeve_predict,
                'formal_predict': formal_predict,
                'shorts_predict': shorts_predict,
                'jeans_predict': jeans_predict,
                'skirt_predict': skirt_predict,
                'facemask_predict': facemask_predict,
                'logo_predict': logo_predict,
                'stripe_predict': stripe_predict,
                'longpants_predict': longpants_predict

            }

            all_results.append(res)
            tqdm_bar.update(1)
    return all_results


# WIDER test sets are used by default
def offline_evaluate(pred_func, output_file):
    datasets, df = get_wider_eval_dataflow()
    all_results = eval_W(
        df, lambda img, boxes, male, longhair, sunglass, hat, tshirt,
                   longsleeve, formal, shorts, jeans, skirt, facemask,
                   logo, stripe, longpants: eval_one_image(img, boxes, male,
                                                           longhair, sunglass,
                                                           hat, tshirt,
                                                           longsleeve,
                                                           formal, shorts,
                                                           jeans, skirt,
                                                           facemask, logo,
                                                           stripe, longpants,
                                                           pred_func))
    with open(output_file, 'w') as f:
        json.dump(all_results, f)

    print_evaluation_scores(output_file)


DetectionResult = namedtuple(
    'DetectionResult',
    ['male', 'longhair', 'sunglass', 'hat', 'tshirt', 'longsleeve', 'formal', 'shorts',
     'jeans', 'skirt', 'facemask', 'logo', 'stripe', 'longpants', 'male_predict', 'longhair_predict',
     'sunglass_predict', 'hat_predict', 'tshirt_predict', 'longsleeve_predict', 'formal_predict',
     'shorts_predict', 'jeans_predict', 'skirt_predict', 'facemask_predict', 'logo_predict',
     'stripe_predict', 'longpants_predict'])


def eval_one_image(img, box, male, longhair, sunglass, hat, tshirt, longsleeve, formal, shorts, jeans, skirt, facemask,
                   logo, stripe, longpants, model_func):
    resizer = CustomResize(cfg.PREPROC.TEST_SHORT_EDGE_SIZE, cfg.PREPROC.MAX_SIZE)
    resized_img = resizer.augment(img)
    scale = np.sqrt(resized_img.shape[0] * 1.0 / img.shape[0] * resized_img.shape[1] / img.shape[1])
    box = box * scale

    male, longhair, sunglass, hat, tshirt, \
    longsleeve, formal, shorts, jeans, \
    skirt, facemask, logo, stripe, longpants, \
    male_predict, longhair_predict, sunglass_predict, hat_predict, tshirt_predict, \
    longsleeve_predict, formal_predict, shorts_predict, jeans_predict, skirt_predict, \
    facemask_predict, logo_predict, stripe_predict, longpants_predict = model_func(resized_img, box, male, longhair,
                                                                                   sunglass, hat, tshirt, longsleeve,
                                                                                   formal, shorts, jeans, skirt,
                                                                                   facemask, logo, stripe, longpants)

    male_predict, longhair_predict, sunglass_predict, \
    hat_predict, tshirt_predict, longsleeve_predict, \
    formal_predict, shorts_predict, jeans_predict, \
    skirt_predict, facemask_predict, logo_predict, \
    stripe_predict, longpants_predict = male_predict[:, 1], longhair_predict[:, 1], sunglass_predict[:, 1], \
                                        hat_predict[:, 1], tshirt_predict[:, 1], longsleeve_predict[:, 1], \
                                        formal_predict[:, 1], shorts_predict[:, 1], jeans_predict[:, 1], \
                                        skirt_predict[:, 1], facemask_predict[:, 1], logo_predict[:, 1], \
                                        stripe_predict[:, 1], longpants_predict[:, 1]

    results = [DetectionResult(*args) for args in zip(male, longhair, sunglass, hat, tshirt,
                                                      longsleeve, formal, shorts, jeans,
                                                      skirt, facemask, logo, stripe, longpants,
                                                      male_predict, longhair_predict,
                                                      sunglass_predict,
                                                      hat_predict, tshirt_predict, longsleeve_predict,
                                                      formal_predict,
                                                      shorts_predict, jeans_predict, skirt_predict,
                                                      facemask_predict, logo_predict, stripe_predict,
                                                      longpants_predict)]
    return results


def filter_and_evaluate(name, attr, attr_predict):
    index = np.where(attr >= 0)
    attr = attr[index]
    attr_predict = attr_predict[index]
    average_precision = average_precision_score(attr, attr_predict)
    print(name + ':', average_precision)
    return average_precision


def print_evaluation_scores(json_file):
    assert cfg.WIDER.BASEDIR and os.path.isdir(cfg.WIDER.BASEDIR)

    with open(json_file, 'r') as f:
        predict_results = json.load(f)

    results = list(zip(
        *[[item['male'], item['male_predict'],
           item['longhair'], item['longhair_predict'],
           item['sunglass'], item['sunglass_predict'],
           item['hat'], item['hat_predict'],
           item['tshirt'], item['tshirt_predict'],
           item['longsleeve'], item['longsleeve_predict'],
           item['formal'], item['formal_predict'],
           item['shorts'], item['shorts_predict'],
           item['jeans'], item['jeans_predict'],
           item['skirt'], item['skirt_predict'],
           item['facemask'], item['facemask_predict'],
           item['logo'], item['logo_predict'],
           item['stripe'], item['stripe_predict'],
           item['longpants'], item['longpants_predict']
           ] for item in predict_results]))

    male, male_predict, longhair, longhair_predict, sunglass, sunglass_predict, \
    hat, hat_predict, tshirt, tshirt_predict, longsleeve, longsleeve_predict, \
    formal, formal_predict, shorts, shorts_predict, jeans, jeans_predict, \
    skirt, skirt_predict, facemask, facemask_predict, logo, logo_predict, \
    stripe, stripe_predict, longpants, longpants_predict = [np.concatenate(r, axis=0) for r in results]

    ret = {'male': filter_and_evaluate('male', male, male_predict),
           'longhair': filter_and_evaluate('longhair', longhair, longhair_predict),
           'sunglass': filter_and_evaluate('sunglass', sunglass, sunglass_predict),
           'hat': filter_and_evaluate('hat', hat, hat_predict),
           'tshirt': filter_and_evaluate('tshirt', tshirt, tshirt_predict),
           'longsleeve': filter_and_evaluate('longsleeve', longsleeve, longsleeve_predict),
           'formal': filter_and_evaluate('formal', formal, formal_predict),
           'shorts': filter_and_evaluate('shorts', shorts, shorts_predict),
           'jeans': filter_and_evaluate('jeans', jeans, jeans_predict),
           'skirt': filter_and_evaluate('skirt', skirt, skirt_predict),
           'facemask': filter_and_evaluate('facemask', facemask, facemask_predict),
           'logo': filter_and_evaluate('logo', logo, logo_predict),
           'stripe': filter_and_evaluate('stripe', stripe, stripe_predict),
           'longpants': filter_and_evaluate('longpants', longpants, longpants_predict)}

    print('mAP:', np.mean(list(ret.values())))

    return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='load a model for evaluation or training. Can overwrite BACKBONE.WEIGHTS')
    parser.add_argument('--evaluate', help="Run evaluation on WIDER. "
                                           "This argument is the path to the output json evaluation file")
    parser.add_argument('--config', help="A list of KEY=VALUE to overwrite those defined in tensorpack_config.py",
                        nargs='+')

    if get_tf_version_tuple() < (1, 6):
        logger.warn("TF<1.6 has a bug which may lead to crash in FasterRCNN if you're unlucky.")

    args = parser.parse_args()
    if args.config:
        cfg.update_args(args.config)

    MODEL = ResNetC4Model()
    # predict part
    if args.evaluate:
        assert args.load
        finalize_configs(is_training=False)

        pred = OfflinePredictor(PredictConfig(
            model=MODEL,
            session_init=get_model_loader(args.load),
            input_names=['image', 'gt_boxes', 'male', 'longhair', 'sunglass',
                         'hat', 'tshirt', 'longsleeve',
                         'formal', 'shorts', 'jeans',
                         'skirt', 'facemask', 'logo',
                         'stripe', 'longpants'],

            output_names=['male', 'longhair', 'sunglass',
                          'hat', 'tshirt', 'longsleeve',
                          'formal', 'shorts', 'jeans',
                          'skirt', 'facemask', 'logo',
                          'stripe', 'longpants',

                          'pmale', 'plonghair', 'psunglass',
                          'phat', 'ptshirt', 'plongsleeve',
                          'pformal', 'pshorts', 'pjeans',
                          'pskirt', 'pfacemask', 'plogo',
                          'pstripe', 'plongpants'
                          ]))

        assert args.evaluate.endswith('.json'), args.evaluate

        offline_evaluate(pred, args.evaluate)

'''
--evaluate
/root/datasets/wider_results.json
--config
FRCNN.BATCH_PER_IM=64
PREPROC.SHORT_EDGE_SIZE=600
PREPROC.MAX_SIZE=1024
--load
/root/datasets/0509/checkpoint
'''
