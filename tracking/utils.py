import numpy as np
from numba import jit


@jit
def iou(bb_test, bb_gt):
    """
    Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return o

# @jit
# def iof(bb_face, bb_person):
#     """
#     Computes "Intersection over Face" between two bboxes in the form [x1,y1,x2,y2]
#     """
#     xx1 = np.maximum(bb_face[0], bb_person[0])
#     yy1 = np.maximum(bb_face[1], bb_person[1])
#     xx2 = np.minimum(bb_face[2], bb_person[2])
#     yy2 = np.minimum(bb_face[3], bb_person[3])
#     w = np.maximum(0., xx2 - xx1)
#     h = np.maximum(0., yy2 - yy1)
#     wh = w * h
#     iof = wh / ((bb_face[2] - bb_face[0]) * (bb_face[3] - bb_face[1]))
#     return iof


@jit
def iof(bb_face, mask_person):
    """
    Computes "Intersection over Face" between bb_face and mask_person
    """
    ranks = range(int(bb_face[1]), int(bb_face[3]))  # 行数
    cavalcade = range(int(bb_face[0]), int(bb_face[2]))  # 列数
    mask_person_r = mask_person[ranks]
    mask_person_rc = mask_person_r[:,cavalcade]
    intersection = np.sum(mask_person_rc)
    iof = intersection / ((bb_face[2] - bb_face[0]) * (bb_face[3] - bb_face[1]))
    return iof