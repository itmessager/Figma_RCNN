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

from detection.tensorpacks.data import (
    get_train_dataflow, get_eval_dataflow,
    get_all_anchors, get_all_anchors_fpn, get_wider_dataflow)
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
        if cfg.TRAIN.NUM_GPUS < 8:
            opt = optimizer.AccumGradOptimizer(opt, 8 // cfg.TRAIN.NUM_GPUS)
        return opt

    def get_inference_tensor_names(self):
        """
        Returns two lists of tensor names to be used to create an inference callable.

        Returns:
            [str]: input names
            [str]: output names
        """
        out = ['person_boxes', 'person_scores', 'person_labels',
               'male_predict', 'longhair_predict', 'sunglass_predict',
               'hat_predict', 'tshirt_predict', 'longsleeve_predict',
               'formal_predict', 'shorts_predict', 'jeans_predict',
               'skirt_predict', 'facemask_predict', 'logo_predict',
               'stripe_predict', 'longpants_predict'
               ]
        if cfg.MODE_MASK:
            out.append('output/masks')
        return ['image'], out

from detection.tensorpacks.attrs_predict import ResNetC4Model


def visualize(model, model_path, nr_visualize=100, output_dir='output'):
    """
    Visualize some intermediate results (proposals, raw predictions) inside the pipeline.
    """
    df = get_train_dataflow()  # we don't visualize mask stuff
    df.reset_state()

    pred = OfflinePredictor(PredictConfig(
        model=model,
        session_init=get_model_loader(model_path),
        input_names=['image', 'gt_boxes', 'gt_labels'],
        output_names=[
            'generate_{}_proposals/boxes'.format('fpn' if cfg.MODE_FPN else 'rpn'),
            'generate_{}_proposals/scores'.format('fpn' if cfg.MODE_FPN else 'rpn'),
            'fastrcnn_all_scores',
            'output/boxes',
            'output/scores',
            'output/labels',
        ]))

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    utils.fs.mkdir_p(output_dir)
    with tqdm.tqdm(total=nr_visualize) as pbar:
        for idx, dp in itertools.islice(enumerate(df), nr_visualize):
            # img = dp[0]
            img = dp['image']
            if cfg.MODE_MASK:
                # gt_boxes, gt_labels, gt_masks = dp[-3:]
                gt_boxes, gt_labels, gt_masks = dp['gt_boxes'], dp['gt_labels'], dp['gt_masks']
            else:
                # gt_boxes, gt_labels = dp[-2:]
                gt_boxes, gt_labels = dp['gt_boxes'], dp['gt_labels']

            rpn_boxes, rpn_scores, all_scores, \
            final_boxes, final_scores, final_labels = pred(img, gt_boxes, gt_labels)

            # draw groundtruth boxes
            gt_viz = draw_annotation(img, gt_boxes, gt_labels)
            # draw best proposals for each groundtruth, to show recall
            proposal_viz, good_proposals_ind = draw_proposal_recall(img, rpn_boxes, rpn_scores, gt_boxes)
            # draw the scores for the above proposals
            score_viz = draw_predictions(img, rpn_boxes[good_proposals_ind], all_scores[good_proposals_ind])

            results = [DetectionResult(*args) for args in
                       zip(final_boxes, final_scores, final_labels,
                           [None] * len(final_labels))]
            final_viz = draw_final_outputs(img, results)

            viz = tpviz.stack_patches([
                gt_viz, proposal_viz,
                score_viz, final_viz], 2, 2)

            if os.environ.get('DISPLAY', None):
                tpviz.interactive_imshow(viz)
            cv2.imwrite("{}/{:03d}.png".format(output_dir, idx), viz)
            pbar.update()


def offline_evaluate(pred_func, output_file):
    df = get_eval_dataflow()
    all_results = eval_coco(
        df, lambda img: detect_one_image(img, pred_func))
    with open(output_file, 'w') as f:
        json.dump(all_results, f)
    print_evaluation_scores(output_file)


def predict(pred_func, input_file):

    img = cv2.imread(input_file, cv2.IMREAD_COLOR)
    results = detect_one_image(img, pred_func)
    final = draw_final_outputs(img, results)
    viz = np.concatenate((img, final), axis=1)
    tpviz.interactive_imshow(viz)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='load a model for evaluation or training. Can overwrite BACKBONE.WEIGHTS')
    parser.add_argument('--logdir', help='log directory', default='train_log/maskrcnn')
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
            input_names=['image'],
            # output_names=['person_boxes', 'person_scores', 'person_labels',
            #               'male_predict', 'longhair_predict', 'sunglass_predict',
            #               'hat_predict', 'tshirt_predict', 'longsleeve_predict',
            #               'formal_predict', 'shorts_predict', 'jeans_predict',
            #               'skirt_predict', 'facemask_predict', 'logo_predict',
            #               'stripe_predict', 'longpants_predict'
            #               ]))

            output_names=['output/boxes', 'output/scores', 'output/labels',
                          'male_predict', 'longhair_predict', 'sunglass_predict',
                          'hat_predict', 'tshirt_predict', 'longsleeve_predict',
                          'formal_predict', 'shorts_predict', 'jeans_predict',
                          'skirt_predict', 'facemask_predict', 'logo_predict',
                          'stripe_predict', 'longpants_predict'
                          ]))

        assert args.evaluate.endswith('.json'), args.evaluate



        offline_evaluate(pred, args.evaluate)


'''

--config
MODE_MASK=False
FRCNN.BATCH_PER_IM=64
PREPROC.SHORT_EDGE_SIZE=600
PREPROC.MAX_SIZE=1024
TRAIN.LR_SCHEDULE=[150000,230000,280000]
BACKBONE.WEIGHTS=/root/datasets/COCO-R50C4-MaskRCNN-Standard.npz
DATA.BASEDIR=/root/datasets/COCO/DIR/

'''


'''

--predict 
/root/datasetsimg-folder/1.jpg
--load
/home/Figma_RCNN/detection/tensorpacks/train_log/maskrcnn/checkpoint
--config 
MODE_MASK=False


'''