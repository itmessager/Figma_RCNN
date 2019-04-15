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
    image_preprocess, resnet_c4_backbone, resnet_conv5)

from detection.tensorpacks.model_frcnn import (fastrcnn_outputs,
                                               fastrcnn_predictions, BoxProposals, FastRCNNHead, attrs_predict,
                                               logits_to_predict_v2,
                                               logits_to_predict)

from detection.tensorpacks.model_rpn import rpn_head, generate_rpn_proposals
from detection.tensorpacks.model_box import (
    clip_boxes, roi_align, RPNAnchors)
from detection.config.config import finalize_configs, config as cfg




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
            tf.placeholder(tf.float32, (None, 4), 'gt_boxes')]
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
        gt_boxes = inputs['gt_boxes']
        person_roi_resized = roi_align(featuremap, gt_boxes * (1.0 / cfg.RPN.ANCHOR_STRIDE), 14)
        feature_attrs = resnet_conv5(person_roi_resized, cfg.BACKBONE.RESNET_NUM_BLOCK[-1])
        feature_attrs_gap = GlobalAvgPooling('gap', feature_attrs, data_format='channels_first')  #
        attrs_labels = attrs_predict(feature_attrs_gap, logits_to_predict)

def eval_W(df, detect_func, tqdm_bar=None):
    """
    Args:
        df: a DataFlow which produces (image, image_id)
        detect_func: a callable, takes [image] and returns [DetectionResult]
        tqdm_bar: a tqdm object to be shared among multiple evaluation instances. If None,
            will create a new one.
    Returns:
        list of dict, to be dumped to COCO json format
    """
    df.reset_state()
    all_results = []
    # tqdm is not quite thread-safe: https://github.com/tqdm/tqdm/issues/323
    with ExitStack() as stack:
        if tqdm_bar is None:
            tqdm_bar = stack.enter_context(
                tqdm.tqdm(total=df.size(), **get_tqdm_kwargs()))
        for img, boxes, img_id in df:
            results = detect_func(img, boxes)
            for r in results:
                res = {
                    'image_id': int(img_id),
                    'male': int(r.male),
                    'longhair':int(r.longhair),
                    'sunglass':int(r.sunglass),
                    'hat':int(r.hat),
                    'tshirt':int(r.tshirt),
                    'longsleeve':int(r.longsleeve),
                    'formal':int(r.formal),
                    'shorts':int(r.shorts),
                    'jeans':int(r.jeans),
                    'skirt':int(r.skirt),
                    'facemask':int(r.facemask),
                    'logo':int(r.logo),
                    'stripe':int(r.stripe),
                    'longpants':int(r.longpants)
                }

                all_results.append(res)
            tqdm_bar.update(1)
    return all_results

def offline_evaluate(pred_func, output_file):
    df = get_wider_eval_dataflow()
    all_results = eval_W(
        df, lambda img, box: eval_one_image(img, box, pred_func))
    with open(output_file, 'w') as f:
        json.dump(all_results, f)

    # output_file = /root/datasets/wider_results.json
    print_evaluation_scores(output_file)


def print_evaluation_scores(json_file):
    ret = {}
    assert cfg.DATA.BASEDIR and os.path.isdir(cfg.DATA.BASEDIR)
    annofile = os.path.join(
        cfg.DATA.BASEDIR, 'annotations',
        'instances_{}.json'.format(cfg.DATA.VAL))
    coco = COCO(annofile)
    cocoDt = coco.loadRes(json_file)
    cocoEval = COCOeval(coco, cocoDt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    fields = ['IoU=0.5:0.95', 'IoU=0.5', 'IoU=0.75', 'small', 'medium', 'large']
    for k in range(6):
        ret['mAP(bbox)/' + fields[k]] = cocoEval.stats[k]

    if cfg.MODE_MASK:
        cocoEval = COCOeval(coco, cocoDt, 'segm')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        for k in range(6):
            ret['mAP(segm)/' + fields[k]] = cocoEval.stats[k]
    return ret








# def predict(pred_func, input_file):
#
#     img = cv2.imread(input_file, cv2.IMREAD_COLOR)
#     results = detect_one_image(img, pred_func)
#     final = draw_final_outputs(img, results)
#     viz = np.concatenate((img, final), axis=1)
#     tpviz.interactive_imshow(viz)


DetectionResult = namedtuple(
    'DetectionResult',
    ['male', 'longhair', 'sunglass', 'hat', 'tshirt', 'longsleeve', 'formal', 'shorts',
     'jeans', 'skirt', 'facemask', 'logo', 'stripe', 'longpants'])

def eval_one_image(img, box, model_func):
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
    resizer = CustomResize(cfg.PREPROC.TEST_SHORT_EDGE_SIZE, cfg.PREPROC.MAX_SIZE)
    resized_img = resizer.augment(img)
    scale = np.sqrt(resized_img.shape[0] * 1.0 / img.shape[0] * resized_img.shape[1] / img.shape[1])
    box = box * scale
    attrs = model_func(resized_img, box)

    results = [DetectionResult(*args) for args in zip(attrs[0], attrs[1], attrs[2], attrs[3],
                                                      attrs[4], attrs[5], attrs[6], attrs[7],
                                                      attrs[8], attrs[9], attrs[10], attrs[11],
                                                      attrs[12], attrs[13])]
    return results




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='load a model for evaluation or training. Can overwrite BACKBONE.WEIGHTS')
    parser.add_argument('--evaluate', help="Run evaluation on COCO. "
                                           "This argument is the path to the output json evaluation file")
    parser.add_argument('--config', help="A list of KEY=VALUE to overwrite those defined in tensorpack_config.py",
                        nargs='+')

    if get_tf_version_tuple() < (1, 6):
        # https://github.com/tensorflow/tensorflow/issues/14657
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
            input_names=['image', 'gt_boxes'],
            output_names=['male_predict', 'longhair_predict', 'sunglass_predict',
                          'hat_predict', 'tshirt_predict', 'longsleeve_predict',
                          'formal_predict', 'shorts_predict', 'jeans_predict',
                          'skirt_predict', 'facemask_predict', 'logo_predict',
                          'stripe_predict', 'longpants_predict'
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
/root/datasets/0317/checkpoint
'''