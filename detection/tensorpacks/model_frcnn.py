# -*- coding: utf-8 -*-
# File: model.py

import tensorflow as tf
from tensorpack.tfutils import varreplace
from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.argscope import argscope
from tensorpack.tfutils.scope_utils import under_name_scope
from tensorpack.models import (
    Conv2D, FullyConnected, layer_register)
from tensorpack.utils.argtools import memoized

from detection.tensorpacks.basemodel import GroupNorm
from detection.tensorpacks.utils.box_ops import pairwise_iou
from detection.tensorpacks.model_box import encode_bbox_target, decode_bbox_target
from detection.config.config import config as cfg


@under_name_scope()
def proposal_metrics(iou):
    """
    Add summaries for RPN proposals.

    Args:
        iou: nxm, #proposal x #gt
    """
    # find best roi for each gt, for summary only
    best_iou = tf.reduce_max(iou, axis=0)
    mean_best_iou = tf.reduce_mean(best_iou, name='best_iou_per_gt')
    summaries = [mean_best_iou]
    with tf.device('/cpu:0'):
        for threshold in [0.3, 0.5]:
            recall = tf.truediv(
                tf.count_nonzero(best_iou >= threshold),
                tf.size(best_iou, out_type=tf.int64),
                name='recall_iou{}'.format(threshold))
            summaries.append(recall)
    add_moving_summary(*summaries)


@under_name_scope()
def sample_fast_rcnn_targets(boxes, gt_boxes, gt_labels):
    """
    Sample some ROIs from all proposals for training.
    #fg is guaranteed to be > 0, because grount truth boxes are added as RoIs.

    Args:
        boxes: nx4 region proposals, floatbox
        gt_boxes: mx4, floatbox ,from datasets
        gt_labels: m, int32  ,from datasets

    Returns:
        A BoxProposals instance.
        sampled_boxes: tx4 floatbox, the rois
        sampled_labels: t int64 labels, in [0, #class). Positive means foreground.
        fg_inds_wrt_gt: #fg indices, each in range [0, m-1].
            It contains the matching GT of each foreground roi.
    """
    iou = pairwise_iou(boxes, gt_boxes)  # nxm
    proposal_metrics(iou)

    # add ground truth as proposals as well
    boxes = tf.concat([boxes, gt_boxes], axis=0)  # (n+m) x 4
    iou = tf.concat([iou, tf.eye(tf.shape(gt_boxes)[0])], axis=0)  # (n x m + m x m)  # OK

    # #proposal=n+m from now on

    def sample_fg_bg(iou):
        fg_mask = tf.reduce_max(iou, axis=1) >= cfg.FRCNN.FG_THRESH

        fg_inds = tf.reshape(tf.where(fg_mask), [-1])  # 2-D call mask,1-D call indices
        num_fg = tf.minimum(int(
            cfg.FRCNN.BATCH_PER_IM * cfg.FRCNN.FG_RATIO),
            tf.size(fg_inds), name='num_fg')
        fg_inds = tf.random_shuffle(fg_inds)[:num_fg]

        bg_inds = tf.reshape(tf.where(tf.logical_not(fg_mask)), [-1])
        num_bg = tf.minimum(
            cfg.FRCNN.BATCH_PER_IM - num_fg,
            tf.size(bg_inds), name='num_bg')
        bg_inds = tf.random_shuffle(bg_inds)[:num_bg]

        add_moving_summary(num_fg, num_bg)  # ??
        return fg_inds, bg_inds  # len_fg + len_bg = m + n

    fg_inds, bg_inds = sample_fg_bg(iou)
    # fg,bg indices w.r.t proposals

    best_iou_ind = tf.argmax(iou, axis=1)  # #proposal, each in 0~m-1  because after shuffle # OK
    fg_inds_wrt_gt = tf.gather(best_iou_ind, fg_inds)  # best_fg_indices    m

    all_indices = tf.concat([fg_inds, bg_inds], axis=0)  # indices w.r.t all n+m proposal boxes
    ret_boxes = tf.gather(boxes, all_indices)

    ret_labels = tf.concat(
        [tf.gather(gt_labels, fg_inds_wrt_gt),
         tf.zeros_like(bg_inds, dtype=tf.int64)], axis=0)  # OK
    # stop the gradient -- they are meant to be training targets
    return BoxProposals(
        tf.stop_gradient(ret_boxes, name='sampled_proposal_boxes'),
        tf.stop_gradient(ret_labels, name='sampled_labels'),
        tf.stop_gradient(fg_inds_wrt_gt),
        gt_boxes, gt_labels)


# @layer_register(log_shape=True)    # add layer_register if the npz contain this layer
def attrs_head(name, feature):
    """
    Attribute network branchs
    Args:
        name: name scope
        feature: feature of rois
    Returns:
        A Dict
        attribute name: attribute logits
    """
    with tf.name_scope(name):
        attrs_logits = {'male': attr_output('male', feature), 'longhair': attr_output('longhair', feature),
                        'sunglass': attr_output('sunglass', feature), 'hat': attr_output('hat', feature),
                        'tshirt': attr_output('tshirt', feature), 'longsleeve': attr_output('longsleeve', feature),
                        'formal': attr_output('formal', feature), 'shorts': attr_output('shorts', feature),
                        'jeans': attr_output('jeans', feature), 'skirt': attr_output('skirt', feature),
                        'facemask': attr_output('facemask', feature), 'logo': attr_output('logo', feature),
                        'stripe': attr_output('stripe', feature), 'longpants': attr_output('longpants', feature)}

        return attrs_logits


# def attr_output(name, feature):
#     with argscope([Conv2D], data_format='channels_first',
#                   kernel_initializer=tf.variance_scaling_initializer(
#                       scale=2.0, mode='fan_out', distribution='normal')):
#         feature_attributes = Conv2D('conv_{}'.format(name), feature, 512, 3, activation=tf.nn.relu)
#
#     feature_gap_ = GlobalAvgPooling('gap', feature_attributes, data_format='channels_first')
#
#     hidden = FullyConnected('{}_hidden'.format(name), feature_gap_, 128, activation=tf.nn.relu,
#                             kernel_initializer=tf.random_normal_initializer(stddev=0.01))
#     attr = FullyConnected(
#         name, hidden, 2,
#         kernel_initializer=tf.random_normal_initializer(stddev=0.01))
#     return attr

# 2048-->512-->2
def attr_output(name, feature):
    hidden = FullyConnected('{}_hidden'.format(name), feature, 512, activation=tf.nn.relu,
                            kernel_initializer=tf.random_normal_initializer(stddev=0.01))
    attr = FullyConnected(
        name, hidden, 2,
        kernel_initializer=tf.random_normal_initializer(stddev=0.01))
    return attr


def attrs_predict(feature, predict=None):
    """
    """
    if predict:
        attrs_predict = [predict(attr_output('male', feature), 'male'),
                        predict(attr_output('longhair', feature), 'longhair'),
                        predict(attr_output('sunglass', feature), 'sunglass'),
                        predict(attr_output('hat', feature), 'hat'),
                        predict(attr_output('tshirt', feature), 'tshirt'),
                        predict(attr_output('longsleeve', feature), 'longsleeve'),
                        predict(attr_output('formal', feature), 'formal'),
                        predict(attr_output('shorts', feature), 'shorts'),
                        predict(attr_output('jeans', feature), 'jeans'),
                        predict(attr_output('skirt', feature), 'skirt'),
                        predict(attr_output('facemask', feature), 'facemask'),
                        predict(attr_output('logo', feature), 'logo'),
                        predict(attr_output('stripe', feature), 'stripe'),
                        predict(attr_output('longpants', feature), 'longpants')]
    else:
        attrs_predict = [tf.nn.softmax(attr_output('male', feature), name='pmale'),
                        tf.nn.softmax(attr_output('longhair', feature), name='plonghair'),
                        tf.nn.softmax(attr_output('sunglass', feature), name='psunglass'),
                        tf.nn.softmax(attr_output('hat', feature), name='phat'),
                        tf.nn.softmax(attr_output('tshirt', feature), name='ptshirt'),
                        tf.nn.softmax(attr_output('longsleeve', feature), name='plongsleeve'),
                        tf.nn.softmax(attr_output('formal', feature), name='pformal'),
                        tf.nn.softmax(attr_output('shorts', feature), name='pshorts'),
                        tf.nn.softmax(attr_output('jeans', feature), name='pjeans'),
                        tf.nn.softmax(attr_output('skirt', feature), name='pskirt'),
                        tf.nn.softmax(attr_output('facemask', feature), name='pfacemask'),
                        tf.nn.softmax(attr_output('logo', feature), name='plogo'),
                        tf.nn.softmax(attr_output('stripe', feature), name='pstripe'),
                        tf.nn.softmax(attr_output('longpants', feature), name='plongpants')]
    return attrs_predict


def logits_to_predict(attr_logits, name=None):
    """
    Args:
        :param name: add name for tensor if name is not None
        :param attr_logits:
    Returns:
        predict_label nx1 [-1,1,0,-1,-1] int64

    """
    specific_logits = attr_logits[:, 0]    # 1 means sure    0 means not sure
    attribute_logits = attr_logits[:, 1]   # 1 means Yes     0 means No

    prediction = tf.where(attribute_logits > 0.5, tf.ones_like(attribute_logits), tf.zeros_like(attribute_logits))
    prediction = tf.where(specific_logits < 0.5, -tf.ones_like(prediction), prediction)

    predict_label = tf.to_int32(prediction)
    if name:
        return tf.identity(predict_label, name='{}_predict'.format(name))
    else:
        return predict_label


def logits_to_predict_v2(attr_logits, name=None):
    """
    this function is only contains two type
    Args:
        :param name: add name for tensor if name is not None
        :param attr_logits:
    Returns:
        predict_label nx1 [-1,1,0,-1,-1] int64

    """
    prediction = tf.argmax(attr_logits, axis=-1)
    predict_label = tf.to_int32(prediction)
    if name:
        return tf.identity(predict_label, name='{}_predict'.format(name))
    else:
        return predict_label



def all_attrs_losses(attr_labels, attr_logits,loss_function):
    """
    Args:
        :param attr_logits: n,
        :param attr_labels: nxC
    Returns:
        label_loss, box_loss
    """
    attrs_loss = [loss_function('male', attr_labels['male'], attr_logits['male']),
                  loss_function('longhair', attr_labels['longhair'], attr_logits['longhair']),
                  loss_function('sunglass', attr_labels['sunglass'], attr_logits['sunglass']),
                  loss_function('hat', attr_labels['hat'], attr_logits['hat']),
                  loss_function('tshirt', attr_labels['tshirt'], attr_logits['tshirt']),
                  loss_function('longsleeve', attr_labels['longsleeve'], attr_logits['longsleeve']),
                  loss_function('formal', attr_labels['formal'], attr_logits['formal']),
                  loss_function('shorts', attr_labels['shorts'], attr_logits['shorts']),
                  loss_function('jeans', attr_labels['jeans'], attr_logits['jeans']),
                  loss_function('skirt', attr_labels['skirt'], attr_logits['skirt']),
                  loss_function('facemask', attr_labels['facemask'], attr_logits['facemask']),
                  loss_function('logo', attr_labels['logo'], attr_logits['logo']),
                  loss_function('stripe', attr_labels['stripe'], attr_logits['stripe']),
                  loss_function('longpants', attr_labels['longpants'], attr_logits['longpants'])]
    attrs_loss = tf.add_n(attrs_loss)
    return attrs_loss


def all_cor_cost(attr_logits):
    """
    Args:
        :param attr_logits: n,
    Returns:
        label_loss, box_loss
    """

    def correlation(name1, name2, f, k=10):
        vector = f(tf.nn.softmax(attr_logits[name1])[:, 1],
                   tf.nn.softmax(attr_logits[name2])[:, 1], k)
        return tf.reduce_sum(vector)

    cor_cost = [correlation('male', 'skirt', f3),
                correlation('longsleeve', 'shorts', f3),
                correlation('formal', 'longpants', f2),
                correlation('tshirt', 'shorts', f1),
                correlation('longpants', 'skirt', f3),
                correlation('formal', 'shorts', f3)]
    cor_cost = tf.add_n(cor_cost)
    return cor_cost


def f1(a1, a2, k):
    return tf.pow(a1 - a2, k)


def f2(a1, a2, k):
    return tf.exp(-k * (tf.pow(a1 - 1, 2) + tf.pow(a2, 2)))


def f3(a1, a2, k):
    return tf.exp(-k * (tf.pow(a1 - 1, 2) + tf.pow(a2 - 1, 2)))


def attr_losses(attr_name, labels, logits):
    """
    Args:
        labels: n,[-1,0,1,1,0]
        logits: nx2 [(0.4,0.6),(0.72,0.28),(0.84,0.16),(0.17,0.83),(0.49,0.51)]
    Returns:
        loss_sum:contain specific_loss and attr_loss
    """
    # the first num of logits is to determine whether the attribute is identifiable


    valid_inds_ = tf.where(labels >= -1)
    labels = tf.reshape(tf.gather(labels, valid_inds_), [-1])
    logits = tf.reshape(tf.gather(logits, valid_inds_), (-1, 2))

    specific_labels = tf.where(labels >= 0, tf.ones_like(labels), tf.zeros_like(labels))
    specific_logits = tf.reshape(logits[:, 0], [-1])

    specific_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.to_float(specific_labels), logits=specific_logits)

    specific_loss_sum = tf.reduce_sum(specific_loss) * 0.05


    # the second num of logits is to determine whether the attribute is positive or negative
    # only use the recognizable attribute to train the second num of logits
    # filter the unrecognizable attribute out
    attr_inds = tf.where(labels >= 0)
    attribute_logits = logits[:, 1]

    attr_labels = tf.reshape(tf.gather(labels, attr_inds), [-1])
    attr_logits = tf.reshape(tf.gather(attribute_logits, attr_inds), [-1])

    attr_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.to_float(attr_labels), logits=attr_logits)
    attr_loss_sum = tf.reduce_sum(attr_loss)
    # attr_loss_sum = tf.reduce_mean(attr_loss, name='attr_loss')

    loss_sum = tf.add_n([attr_loss_sum, specific_loss_sum], name='{}_loss'.format(attr_name))
    prediction = convert2D(tf.gather(attribute_logits, attr_inds))


    with tf.name_scope('{}_metrics'.format(attr_name)), tf.device('/cpu:0'):
        predict_label = logits_to_predict(logits)
        # 0  1  2 means - + not sure
        new_pre = tf.where(predict_label < 0, 2 * tf.ones_like(predict_label), predict_label)
        new_lab = tf.where(labels < 0, 2 * tf.ones_like(labels), labels)

        accuracy = tf.metrics.mean_per_class_accuracy(labels=new_lab, predictions=new_pre, num_classes=3)[1]
        AP = tf.metrics.average_precision_at_k(labels=attr_labels, predictions=prediction, k=1)[1]
        # accuracy = tf.metrics.accuracy(labels=labels, predictions=prediction, )[1]
        accuracy = tf.reduce_mean(accuracy, name='{}_mAcc'.format(attr_name))
        average_precision = tf.identity(AP, name='{}_AP'.format(attr_name))

    add_moving_summary(loss_sum, accuracy, average_precision)

    return loss_sum



def attr_losses_v2(attr_name, labels, logits):
    """
    Args:
        labels: n,[-1,0,1,1,0]
        logits: nx2 [(0.4,0.6),(0.72,0.28),(0.84,0.16),(0.17,0.83),(0.49,0.51)]
    Returns:
        loss_sum: only contain attr_loss(filter out the unspecific attributes)
    """

    # filter the positive and negative samples
    valid_inds = tf.where(labels >= 0)
    valid_labels = tf.reshape(tf.gather(labels, valid_inds), [-1])
    valid_logits = tf.reshape(tf.gather(logits, valid_inds), (-1, 2))

    loss = tf.losses.sparse_softmax_cross_entropy(labels=valid_labels, logits=valid_logits)
    loss_sum = tf.reduce_sum(loss, name='{}_loss'.format(attr_name))

    with tf.name_scope('{}_metrics'.format(attr_name)), tf.device('/cpu:0'):
        prediction = tf.argmax(valid_logits, axis=-1)

        accuracy = tf.metrics.mean_per_class_accuracy(labels=valid_labels, predictions=prediction, num_classes=2)[1]
        AP = tf.metrics.average_precision_at_k(labels=valid_labels, predictions=valid_logits, k=1)[1]
        mean_acc = tf.reduce_mean(accuracy, name='{}_mAcc'.format(attr_name))
        average_precision = tf.identity(AP, name='{}_AP'.format(attr_name))

    add_moving_summary(loss_sum, mean_acc, average_precision)

    return loss_sum


def convert2D(logits):
    logits2D = tf.ones_like(logits) - logits
    return tf.concat([logits2D, logits], 1)



@layer_register(log_shape=True)
def fastrcnn_outputs(feature, num_classes, class_agnostic_regression=False):
    """
    Args:
        feature (any shape):
        num_classes(int): num_category + 1
        class_agnostic_regression (bool): if True, regression to N x 1 x 4

    Returns:
        cls_logits: N x num_class classification logits   2-D
        reg_logits: N x num_class x 4 or Nx2x4 if class agnostic  3-D
    """

    # cls

    classification = FullyConnected(
        'class', feature, num_classes,
        kernel_initializer=tf.random_normal_initializer(stddev=0.01))

    num_classes_for_box = 1 if class_agnostic_regression else num_classes

    # reg
    box_regression = FullyConnected(
        'box', feature, num_classes_for_box * 4,
        kernel_initializer=tf.random_normal_initializer(stddev=0.001))
    box_regression = tf.reshape(box_regression, (-1, num_classes_for_box, 4), name='output_box')

    return classification, box_regression


@under_name_scope()
def fastrcnn_losses(labels, label_logits, fg_boxes, fg_box_logits):
    """
    Args:
        labels: n,
        label_logits: nxC
        fg_boxes: nfgx4, encoded
        fg_box_logits: nfgxCx4 or nfgx1x4 if class agnostic

    Returns:
        label_loss, box_loss
    """
    label_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=label_logits)
    label_loss = tf.reduce_mean(label_loss, name='label_loss')

    fg_inds = tf.where(labels > 0)[:, 0]
    fg_labels = tf.gather(labels, fg_inds)
    num_fg = tf.size(fg_inds, out_type=tf.int64)
    empty_fg = tf.equal(num_fg, 0)
    if int(fg_box_logits.shape[1]) > 1:
        indices = tf.stack(
            [tf.range(num_fg), fg_labels], axis=1)  # #fgx2
        fg_box_logits = tf.gather_nd(fg_box_logits, indices)
    else:
        fg_box_logits = tf.reshape(fg_box_logits, [-1, 4])

    with tf.name_scope('label_metrics'), tf.device('/cpu:0'):
        prediction = tf.argmax(label_logits, axis=1, name='label_prediction')
        correct = tf.to_float(tf.equal(prediction, labels))  # boolean/integer gather is unavailable on GPU
        accuracy = tf.reduce_mean(correct, name='accuracy')
        fg_label_pred = tf.argmax(tf.gather(label_logits, fg_inds), axis=1)
        num_zero = tf.reduce_sum(tf.to_int64(tf.equal(fg_label_pred, 0)), name='num_zero')
        false_negative = tf.where(
            empty_fg, 0., tf.to_float(tf.truediv(num_zero, num_fg)), name='false_negative')
        fg_accuracy = tf.where(
            empty_fg, 0., tf.reduce_mean(tf.gather(correct, fg_inds)), name='fg_accuracy')

    box_loss = tf.losses.huber_loss(
        fg_boxes, fg_box_logits, reduction=tf.losses.Reduction.SUM)
    box_loss = tf.truediv(
        box_loss, tf.to_float(tf.shape(labels)[0]), name='box_loss')

    add_moving_summary(label_loss, box_loss, accuracy,
                       fg_accuracy, false_negative, tf.to_float(num_fg, name='num_fg_label'))
    return label_loss, box_loss


@under_name_scope()
def fastrcnn_predictions(boxes, scores):  # pre_boxes_on_images,label_scores
    """
    Generate final results from predictions of all proposals.

    Args:
        boxes: nx#classx4 floatbox in float32
        scores: nx#class

    Returns:
        boxes: Kx4
        scores: K
        labels: K
    """
    assert boxes.shape[1] == cfg.DATA.NUM_CLASS
    assert scores.shape[1] == cfg.DATA.NUM_CLASS
    boxes = tf.transpose(boxes, [1, 0, 2])[1:, :, :]  # #catxnx4
    boxes.set_shape([None, cfg.DATA.NUM_CATEGORY, None])
    scores = tf.transpose(scores[:, 1:], [1, 0])  # #catxn

    def f(X):
        """
        prob: n probabilities
        box: nx4 boxes

        Returns: n boolean, the selection
        """
        prob, box = X
        output_shape = tf.shape(prob)
        # filter by score threshold
        ids = tf.reshape(tf.where(prob > cfg.TEST.RESULT_SCORE_THRESH), [-1])  # RESULT_SCORE_THRESH = 0.05
        prob = tf.gather(prob, ids)
        box = tf.gather(box, ids)
        # NMS within each class
        selection = tf.image.non_max_suppression(
            box, prob, cfg.TEST.RESULTS_PER_IM, cfg.TEST.FRCNN_NMS_THRESH)  # 100, 0.3
        selection = tf.to_int32(tf.gather(ids, selection))
        # sort available in TF>1.4.0
        # sorted_selection = tf.contrib.framework.sort(selection, direction='ASCENDING')
        sorted_selection = -tf.nn.top_k(-selection, k=tf.size(selection))[0]
        mask = tf.sparse_to_dense(
            sparse_indices=sorted_selection,
            output_shape=output_shape,
            sparse_values=True,
            default_value=False)
        return mask

    masks = tf.map_fn(f, (scores, boxes), dtype=tf.bool,
                      parallel_iterations=10)  # #cat x N
    selected_indices = tf.where(masks)  # #selection x 2, each is (cat_id, box_id)
    scores = tf.boolean_mask(scores, masks)

    # filter again by sorting scores
    topk_scores, topk_indices = tf.nn.top_k(
        scores,
        tf.minimum(cfg.TEST.RESULTS_PER_IM, tf.size(scores)),
        sorted=False)
    filtered_selection = tf.gather(selected_indices, topk_indices)
    cat_ids, box_ids = tf.unstack(filtered_selection, axis=1)

    final_scores = tf.identity(topk_scores, name='scores')
    final_labels = tf.add(cat_ids, 1, name='labels')
    final_ids = tf.stack([cat_ids, box_ids], axis=1, name='all_ids')
    final_boxes = tf.gather_nd(boxes, final_ids, name='boxes')
    return final_boxes, final_scores, final_labels


"""
FastRCNN heads for FPN:
"""


@layer_register(log_shape=True)
def fastrcnn_2fc_head(feature):
    """
    Args:
        feature (any shape):

    Returns:
        2D head feature
    """
    dim = cfg.FPN.FRCNN_FC_HEAD_DIM
    init = tf.variance_scaling_initializer()
    hidden = FullyConnected('fc6', feature, dim, kernel_initializer=init, activation=tf.nn.relu)
    hidden = FullyConnected('fc7', hidden, dim, kernel_initializer=init, activation=tf.nn.relu)
    return hidden


@layer_register(log_shape=True)
def fastrcnn_Xconv1fc_head(feature, num_convs, norm=None):
    """
    Args:
        feature (NCHW):
        num_classes(int): num_category + 1
        num_convs (int): number of conv layers
        norm (str or None): either None or 'GN'

    Returns:
        2D head feature
    """
    assert norm in [None, 'GN'], norm
    l = feature
    with argscope(Conv2D, data_format='channels_first',
                  kernel_initializer=tf.variance_scaling_initializer(
                      scale=2.0, mode='fan_out', distribution='normal')):
        for k in range(num_convs):
            l = Conv2D('conv{}'.format(k), l, cfg.FPN.FRCNN_CONV_HEAD_DIM, 3, activation=tf.nn.relu)
            if norm is not None:
                l = GroupNorm('gn{}'.format(k), l)
        l = FullyConnected('fc', l, cfg.FPN.FRCNN_FC_HEAD_DIM,
                           kernel_initializer=tf.variance_scaling_initializer(), activation=tf.nn.relu)
    return l


def fastrcnn_4conv1fc_head(*args, **kwargs):
    return fastrcnn_Xconv1fc_head(*args, num_convs=4, **kwargs)


def fastrcnn_4conv1fc_gn_head(*args, **kwargs):
    return fastrcnn_Xconv1fc_head(*args, num_convs=4, norm='GN', **kwargs)


class BoxProposals(object):
    """
    A structure to manage box proposals and their relations with ground truth.
    """

    def __init__(self, boxes,
                 labels=None, fg_inds_wrt_gt=None,
                 gt_boxes=None, gt_labels=None):
        """
        Args:
            boxes: Nx4
            labels: N, each in [0, #class), the true label for each input box
            fg_inds_wrt_gt: #fg, each in [0, M)
            gt_boxes: Mx4
            gt_labels: M

        The last four arguments could be None when not training.
        """
        for k, v in locals().items():
            if k != 'self' and v is not None:
                setattr(self, k, v)

    @memoized
    def fg_inds(self):
        """ Returns: #fg indices in [0, N-1] """
        return tf.reshape(tf.where(self.labels > 0), [-1], name='fg_inds')

    @memoized
    def fg_boxes(self):
        """ Returns: #fg x4"""
        return tf.gather(self.boxes, self.fg_inds(), name='fg_boxes')

    @memoized
    def fg_labels(self):
        """ Returns: #fg"""
        return tf.gather(self.labels, self.fg_inds(), name='fg_labels')

    @memoized
    def matched_gt_boxes(self):
        """ Returns: #fg x 4"""
        return tf.gather(self.gt_boxes, self.fg_inds_wrt_gt)  #



class FastRCNNHead(object):
    """
    A class to process & decode inputs/outputs of a fastrcnn classification+regression head.
    """

    def __init__(self, proposals, box_logits, label_logits,  bbox_regression_weights):
        """
        Args:
            proposals: BoxProposals
            box_logits: Nx#classx4 or Nx1x4, the output of the head
            label_logits: Nx#class, the output of the head
            bbox_regression_weights: a 4 element tensor
        """
        for k, v in locals().items():  # locals is a dict
            if k != 'self' and v is not None:
                setattr(self, k, v)
        self._bbox_class_agnostic = int(box_logits.shape[1]) == 1

    @memoized
    def decoded_output_boxes(self):
        """ Returns: N x #class x 4 """
        anchors = tf.tile(tf.expand_dims(self.proposals.boxes, 1),
                          [1, cfg.DATA.NUM_CLASS, 1])  # N x #class x 4
        decoded_boxes = decode_bbox_target(
            self.box_logits / self.bbox_regression_weights,  # [10., 10., 5., 5.]
            anchors
        )
        return decoded_boxes  # pre_boxes_on_images
    @memoized
    def output_scores(self, name=None):
        """ Returns: N x #class scores, summed to one for each box."""
        return tf.nn.softmax(self.label_logits, name=name)

    @memoized
    def fg_box_logits(self):
        """ Returns: #fg x ? x 4 """
        return tf.gather(self.box_logits, self.proposals.fg_inds(), name='fg_box_logits')



    @memoized
    def losses(self):
        encoded_fg_gt_boxes = encode_bbox_target(
            self.proposals.matched_gt_boxes(),
            self.proposals.fg_boxes()) * self.bbox_regression_weights
        return fastrcnn_losses(
            self.proposals.labels, self.label_logits,
            encoded_fg_gt_boxes, self.fg_box_logits()
        )

    @memoized
    def decoded_output_boxes_for_true_label(self):
        """ Returns: Nx4 decoded boxes """
        return self._decoded_output_boxes_for_label(self.proposals.labels)

    @memoized
    def decoded_output_boxes_for_predicted_label(self):
        """ Returns: Nx4 decoded boxes """
        return self._decoded_output_boxes_for_label(self.predicted_labels())

    @memoized
    def decoded_output_boxes_for_label(self, labels):
        assert not self._bbox_class_agnostic
        indices = tf.stack([
            tf.range(tf.size(labels, out_type=tf.int64)),
            labels
        ])
        needed_logits = tf.gather_nd(self.box_logits, indices)
        decoded = decode_bbox_target(
            needed_logits / self.bbox_regression_weights,
            self.proposals.boxes
        )
        return decoded

    @memoized
    def decoded_output_boxes_class_agnostic(self):
        """ Returns: Nx4 """
        assert self._bbox_class_agnostic
        box_logits = tf.reshape(self.box_logits, [-1, 4])
        decoded = decode_bbox_target(
            box_logits / self.bbox_regression_weights,
            self.proposals.boxes
        )
        return decoded

    @memoized
    def output_scores(self, name=None):
        """ Returns: N x #class scores, summed to one for each box."""
        return tf.nn.softmax(self.label_logits, name=name)

    @memoized
    def predicted_labels(self):
        """ Returns: N ints """
        return tf.argmax(self.label_logits, axis=1, name='predicted_labels')

