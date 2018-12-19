from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = os.path.dirname(__file__)
add_path(this_dir)
# print(this_dir)
add_path(os.path.join(this_dir, '..', '..'))


# INFO = {
#     "description": "CelebA Face Dataset",
#     "url": "http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html",
#     "version": "0.1.0",
#     "year": 2018,
#     "contributor": "umass vision",
#     "date_created": datetime.datetime.utcnow().isoformat(' ')
# }

# LICENSES = [
#     {
#         "id": 1,
#         "name": "placeholder",
#         "url": "placeholder"
#     }
# ]

# CATEGORIES = [
#     {
#         'id': 1,
#         'name': 'face',
#         'supercategory': 'face',
#     },
# ]


class MyEncoder(json.JSONEncoder):

    # to solve the problem that TypeError(repr(o) + " is not JSON serializable"
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        return json.JSONEncoder.default(self, obj)


def parse_args():
    parser = argparse.ArgumentParser(description='Convert dataset')
    parser.add_argument(
        '--rootdir', help="root directory for the dataset", required=True, type=str)
    parser.add_argument(
        '--outdir', help="output dir for json files. Default to 'annotations' folder under dataset root dir.", type=str)
    return parser.parse_args()


def parse_celeba_gt(ann_file):
    """
    :param ann_file: the path of annotations file
    :return: a dict like [im-name] = [[x,y,w,h], ...]
    """
    celeba_annot_dict = {}
    with open(ann_file) as f:
        while True:
            xs = []
            ys = []
            ws = []
            hs = []
            filename = f.readline().rstrip()
            if not filename:
                break
            face_num = int(f.readline().rstrip())
            for i in range(face_num):
                annot = f.readline().rstrip().split() # x, y, w, h, ..., other annotations, ...
                if int(annot[2]) >= 10 or int(annot[3]) >= 10:  # Ignore super small image (<10 pixels) as common practice
                    xs.append(float(annot[0]))
                    ys.append(float(annot[1]))
                    ws.append(float(annot[2]))
                    hs.append(float(annot[3]))
            boxlist = xs, ys, ws, hs
            boxes = np.zeros((len(xs), 4))
            for i in range(len(xs)):
                boxes[i:] = [x[i] for x in boxlist]
            celeba_annot_dict[filename] = boxes
    return celeba_annot_dict


def parse_celeba_partition(partition_file):
    train_imgs, val_imgs, test_imgs = [], [], []
    with open(partition_file) as f:
        for line in f.readlines():
            filename, subset_id = line.rstrip().split()
            if subset_id == '0':
                train_imgs.append(filename)
            elif subset_id == '1':
                val_imgs.append(filename)
            else:
                test_imgs.append(filename)
    return train_imgs, val_imgs, test_imgs


def convert_celeba_annots(root_dir, out_dir):
    """Convert from CelebA format to COCO bounding box"""

    img_id = 0
    ann_id = 0
    cat_id = 1


    categories = [{"id": 1, "name": 'face'}]
    images = []
    annotations = []

    print('Parsing annotation file')
    ann_file = os.path.join(root_dir, 'Anno', 'list_bbox_celeba.txt')
    celeba_annot_dict = parse_celeba_gt(ann_file)  # [im-file] = [[x,y,w,h], ...]

    # Get partition
    print('Parsing partition file')
    partition_file = os.path.join(root_dir, 'Eval', 'list_eval_partition.txt')
    train_imgs, val_imgs, test_imgs = parse_celeba_partition(partition_file)

    for subset, img_ids in zip(['train', 'val'], [train_imgs, val_imgs]):
        print('Starting %s' % subset)
        ann_dict = {}
        for filename in img_ids:
            if len(images) % 500 == 0:
                print("Processed %s images, %s annotations" % (
                    len(images), len(annotations)))

            image = {}
            image['id'] = img_id
            img_id += 1
            im = Image.open(os.path.join(root_dir, 'img_celeba', filename))
            image['width'] = im.width
            image['height'] = im.height
            image['file_name'] = filename
            images.append(image)

            for gt_bbox in celeba_annot_dict[filename]:  # CelebA only has one box per image, though
                ann = {}
                ann['id'] = ann_id
                ann_id += 1
                ann['image_id'] = image['id']
                ann['segmentation'] = []
                ann['category_id'] = cat_id  # 1:"face" for CelebA
                ann['iscrowd'] = 0
                ann['area'] = gt_bbox[2] * gt_bbox[3]
                ann['bbox'] = gt_bbox
                annotations.append(ann)

        ann_dict['images'] = images
        ann_dict['categories'] = categories
        ann_dict['annotations'] = annotations
        print("Num categories: %s" % len(categories))
        print("Num images: %s" % len(images))
        print("Num annotations: %s" % len(annotations))

        json_name = 'instances_CelebA_{}.json'.format(subset)
        with open(os.path.join(out_dir, json_name), 'w', encoding='utf8') as outfile:
            outfile.write(json.dumps(ann_dict, cls=MyEncoder, indent=4))
            outfile.close()


if __name__ == '__main__':
    args = parse_args()
    if args.outdir is None:
        args.outdir = os.path.join(args.rootdir, 'annotations')

    convert_celeba_annots(args.rootdir, args.outdir)
