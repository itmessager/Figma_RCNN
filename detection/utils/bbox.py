import numpy as np


def clip_boxes(boxes, shape):
    """
    Args:
        boxes: Numpy array of size N x 4, where each row represents box coordinate (xmin, ymin, xmax, ymax)
        shape: h, w
    """
    orig_shape = boxes.shape
    boxes = boxes.reshape([-1, 4])
    h, w = shape
    boxes[:, [0, 1]] = np.maximum(boxes[:, [0, 1]], 0)
    boxes[:, 2] = np.minimum(boxes[:, 2], w)
    boxes[:, 3] = np.minimum(boxes[:, 3], h)
    return boxes.reshape(orig_shape)
