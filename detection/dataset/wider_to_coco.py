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
#     "description": "WIDER Face Dataset",
#     "url": "http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/",
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
        '--dataset', help="wider", default='wider', type=str)
    parser.add_argument(
        '--rootdir', help="root directory for the dataset", required=True, type=str)
    parser.add_argument(
        '--outdir', help="output dir for json files", type=str)
    # parser.add_argument(
    #     '--datadir', help="data dir for annotations to be converted",
    #     default='/root/datasets/wider/wider_face_split', type=str)
    # parser.add_argument(
    #     '--imdir', help="root directory for loading dataset images",
    #     default='/root/datasets/wider/WIDER_train/images/', type=str)
    # parser.add_argument(
    #     '--annotfile', help="directly specify the annotations file",
    #     default='', type=str)
    return parser.parse_args()


def parse_wider_gt(ann_file):
    """
    :param ann_file: the path of annotations file
    :return: a dict like [im-name] = [[x,y,w,h], ...]
    """
    wider_annot_dict = {}
    f = open(ann_file)
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
        wider_annot_dict[filename] = boxes
    return wider_annot_dict


def convert_wider_annots(root_dir, out_dir):
    """Convert from WIDER FDDB-style format to COCO bounding box"""

    subsets = ['train', 'val']

    for subset in subsets:
        json_name = 'instances_WIDER_{}.json'.format(subset)
        img_id = 0
        ann_id = 0
        cat_id = 1

        print('Starting %s' % subset)
        ann_dict = {}
        categories = [{"id": 1, "name": 'face'}]
        images = []
        annotations = []
        ann_file = os.path.join(root_dir, 'wider_face_split', 'wider_face_{}_bbx_gt.txt'.format(subset))

        wider_annot_dict = parse_wider_gt(ann_file)  # [im-file] = [[x,y,w,h], ...]

        for filename in wider_annot_dict.keys():
            if len(images) % 500 == 0:
                print("Processed %s images, %s annotations" % (
                    len(images), len(annotations)))

            image = {}
            image['id'] = img_id
            img_id += 1
            im = Image.open(os.path.join(root_dir, 'WIDER_{}'.format(subset), 'images', filename))
            image['width'] = im.width
            image['height'] = im.height
            image['file_name'] = filename
            images.append(image)

            for gt_bbox in wider_annot_dict[filename]:
                ann = {}
                ann['id'] = ann_id
                ann_id += 1
                ann['image_id'] = image['id']
                ann['segmentation'] = []
                ann['category_id'] = cat_id  # 1:"face" for WIDER
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
        with open(os.path.join(out_dir, json_name), 'w', encoding='utf8') as outfile:
            outfile.write(json.dumps(ann_dict, cls=MyEncoder, indent=4))
            outfile.close()


def convert_cs6_annots(ann_file, im_dir, out_dir, data_set='CS6-subset'):
    """Convert from WIDER FDDB-style format to COCO bounding box"""

    if data_set == 'CS6-subset':
        json_name = 'cs6-subset_face_train_annot_coco_style.json'
        # ann_file = os.path.join(data_dir, 'wider_face_train_annot.txt')
    else:
        raise NotImplementedError

    img_id = 0
    ann_id = 0
    cat_id = 1

    print('Starting %s' % data_set)
    ann_dict = {}
    categories = [{"id": 1, "name": 'face'}]
    images = []
    annotations = []

    wider_annot_dict = parse_wider_gt(ann_file)  # [im-file] = [[x,y,w,h], ...]

    for filename in wider_annot_dict.keys():
        if len(images) % 50 == 0:
            print("Processed %s images, %s annotations" % (
                len(images), len(annotations)))

        image = {}
        image['id'] = img_id
        img_id += 1
        im = Image.open(os.path.join(im_dir, filename))
        image['width'] = im.height
        image['height'] = im.width
        image['file_name'] = filename
        images.append(image)

        for gt_bbox in wider_annot_dict[filename]:
            ann = {}
            ann['id'] = ann_id
            ann_id += 1
            ann['image_id'] = image['id']
            ann['segmentation'] = []
            ann['category_id'] = cat_id  # 1:"face" for WIDER
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
    with open(os.path.join(out_dir, json_name), 'w', encoding='utf8') as outfile:
        outfile.write(json.dumps(ann_dict))


if __name__ == '__main__':
    args = parse_args()
    if args.outdir is None:
        args.outdir = os.path.join(args.rootdir, 'annotations')

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    if args.dataset == "wider":
        convert_wider_annots(args.rootdir, args.outdir)
    # elif args.dataset == "cs6-subset":
    #     convert_cs6_annots(args.annotfile, args.imdir,
    #                        args.outdir, data_set='CS6-subset')
    else:
        print("Dataset not supported: %s" % args.dataset)
