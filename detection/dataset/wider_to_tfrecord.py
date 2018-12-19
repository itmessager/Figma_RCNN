import tensorflow as tf
import numpy
import cv2
import os
import hashlib
import argparse
# from utils import dataset_util
from object_detection.utils import dataset_util


def parse_test_example(f, images_path):
    filename = f.readline().rstrip()
    if not filename:
        raise IOError()

    filepath = os.path.join(images_path, filename)

    height, width, channel = cv2.imread(filepath).shape

    encoded_image_data = open(filepath, "rb").read()
    key = hashlib.sha256(encoded_image_data).hexdigest()

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(int(height)),
        'image/width': dataset_util.int64_feature(int(width)),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf-8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf-8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
    }))

    return tf_example


def parse_example(f, images_path):
    xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = []  # List of normalized right x coordinates in bounding box (1 per box)
    ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = []  # List of normalized bottom y coordinates in bounding box (1 per box)
    classes_text = []  # List of string class name of bounding box (1 per box)
    classes = []  # List of integer class id of bounding box (1 per box)
    poses = []
    truncated = []
    difficult_obj = []

    filename = f.readline().rstrip()
    if not filename:
        raise IOError()

    filepath = os.path.join(images_path, filename)

    height, width, channel = cv2.imread(filepath).shape

    encoded_image_data = open(filepath, "rb").read()
    key = hashlib.sha256(encoded_image_data).hexdigest()

    face_num = int(f.readline().rstrip())
    if not face_num:
        raise Exception()

    for i in range(face_num):
        annot = f.readline().rstrip().split()
        if not annot:
            raise Exception()

        # WIDER FACE DATASET CONTAINS SOME ANNOTATIONS WHAT EXCEEDS THE IMAGE BOUNDARY
        if float(annot[2]) > 25.0:
            if float(annot[3]) > 30.0:
                xmins.append(max(0.005, (float(annot[0]) / width)))
                ymins.append(max(0.005, (float(annot[1]) / height)))
                xmaxs.append(min(0.995, ((float(annot[0]) + float(annot[2])) / width)))
                ymaxs.append(min(0.995, ((float(annot[1]) + float(annot[3])) / height)))
                classes_text.append(b'face')
                classes.append(1)
                poses.append("front".encode('utf8'))
                truncated.append(int(0))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(int(height)),
        'image/width': dataset_util.int64_feature(int(width)),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf-8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf-8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(int(0)),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }))

    return tf_example


def run(images_path, description_file, output_path, no_bbox=False):
    f = open(description_file)
    writer = tf.python_io.TFRecordWriter(output_path)

    i = 0

    print("Processing {}".format(images_path))
    while True:
        try:
            if no_bbox:
                tf_example = parse_test_example(f, images_path)
            else:
                tf_example = parse_example(f, images_path)

            writer.write(tf_example.SerializeToString())
            i += 1

        except IOError:
            break
        except Exception:
            raise

    writer.close()

    print("Correctly created record for {} images\n".format(i))


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root_path',
        required=True,
        type=str,
        help='Root path to data directory')
    args = parser.parse_args()

    anno_path = os.path.join(args.root_path, 'wider_face_split')

    # Training and Validation
    # # of Train/Val samples: 12880/3226
    for subset in ['train', 'val']:
        subset_path = os.path.join(args.root_path, 'WIDER_{}'.format(subset))
        if os.path.exists(subset_path):
            images_path = os.path.join(subset_path, "images")
            description_file = os.path.join(anno_path, "wider_face_{}_bbx_gt.txt".format(subset))
            output_path = os.path.join(args.root_path, "{}.tfrecord".format(subset))
            if os.path.exists(output_path):
                print("Skip existing {}.tfrecord".format(subset))
                continue
            print("Converting {} set".format(subset))
            run(images_path, description_file, output_path)
        else:
            print("Skip {} set".format(subset))

    # Testing. This set does not contain bounding boxes, so the tfrecord will contain images only
    test_path = os.path.join(args.root_path, "WIDER_test")
    if os.path.exists(test_path):
        images_path = os.path.join(test_path, "images")
        description_file = os.path.join(anno_path, "wider_face_test_filelist.txt")
        output_path = os.path.join(args.root_path, "test.py.tfrecord")
        if os.path.exists(output_path):
            print("Skip existing test.py.tfrecord".format(subset))
        else:
            print("Converting test.py set")
            run(images_path, description_file, output_path, no_bbox=True)


if __name__ == '__main__':
    tf.app.run()
