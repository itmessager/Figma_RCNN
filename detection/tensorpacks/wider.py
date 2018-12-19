import tensorflow as tf
import numpy as np
import cv2
import os
import hashlib
import config


def parse_example(f, images_path, add_gt):

    xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = []  # List of normalized right x coordinates in bounding box (1 per box)
    ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = []  # List of normalized bottom y coordinates in bounding box (1 per box)
    classes_text = []  # List of string class name of bounding box (1 per box)
    classes = []  # List of integer class id of bounding box (1 per box)

    filename = f.readline().rstrip()
    if not filename:
        raise IOError()

    filepath = os.path.join(images_path, filename)

    image_raw = cv2.imread(filepath)

    height, width, _ = image_raw.shape

    face_num = int(f.readline().rstrip())
    if not face_num:
        raise Exception()

    for i in range(face_num):
        annot = f.readline().rstrip().split()
        if not annot:
            raise Exception()

        # WIDER FACE DATASET CONTAINS SOME ANNOTATIONS WHAT EXCEEDS THE IMAGE BOUNDARY
        if (float(annot[2]) > 30.0):
            if (float(annot[3]) > 30.0):
                xmins.append(max(0.005 * width, (float(annot[0]))))
                ymins.append(max(0.005 * height, (float(annot[1]))))
                xmaxs.append(min(0.995 * width, ((float(annot[0]) + float(annot[2])))))
                ymaxs.append(min(0.995 * height, ((float(annot[1]) + float(annot[3])))))
                classes_text.append(b'face')
                classes.append(1)

    boxlist = xmins, ymins, xmaxs, ymaxs
    boxes = np.zeros((len(xmins), 4))
    for i in range(len(xmins)):
        boxes[i:] = [x[i] for x in boxlist]
    if add_gt:
        feature = {
            'boxes': boxes,
            'width': int(width),
            'height': int(height),
            'file_name': filepath,
            'license': 1,
            'coco_url': "null",
            'flickr_url': "null",
            'date_captured': '2018-9-25 09:22:11',
            'is_crowd': np.array([0] * len(xmins)),
            'class': np.array(classes, dtype=int),
            'segmentation': "null",
            'id': 1,
        }
    else:
        feature = {
            'width': int(width),
            'height': int(height),
            'file_name': filepath,
            'license': 1,
            'coco_url': "null",
            'flickr_url': "null",
            'date_captured': '2018-9-25 09:22:11',
            'id': 1,
        }
    return feature


# get the images path and description file
def load_many(datadir, description_file, add_gt=True):
    """
    :param datadir: path of dataset
    :param description_file: path of description file
    :param add_gt: bool type, add ground truth or no ground truth
    :return: a dict about boxes,... of images
    """

    images_path = os.path.join(datadir, "images")
    i = 0
    f = open(description_file)
    roidb = []

    while True:
        try:
            feature = parse_example(f, images_path, add_gt)
            roidb.append(feature)
            i += 1
        except IOError:
            break
        except Exception:
            raise
    return roidb

if __name__ == '__main__':
    roidb = load_many(config.VAL_WIDER_PATH, config.VAL_GT_PATH)

