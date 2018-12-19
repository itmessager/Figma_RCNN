import numpy as np


def clip_bboxes(bboxes, w, h, inplace=True):
    """Clip coordinates of the given bounding boxes to be inside the given size of image. The 0-based index is used.

    :param bboxes: Coordinates of bounding boxes. numpy.ndarray of size (N, 4) where the coordinate is represents as (left, top, right, bottom)
    :param w: Width of the image
    :param h: Height of the image.
    :param inplace:
    :return:
    """
    if not inplace:
        bboxes = np.copy(bboxes)
    np.clip(bboxes[:, 0], 0, w-1, bboxes[:, 0])
    np.clip(bboxes[:, 1], 0, h-1, bboxes[:, 1])
    np.clip(bboxes[:, 2], 0, w-1, bboxes[:, 2])
    np.clip(bboxes[:, 3], 0, h-1, bboxes[:, 3])
    return bboxes



