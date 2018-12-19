import cv2
import os
import argparse
import numpy as np

from detection.core.detector_factory import get_detector
from detection.tensorpacks.viz import draw_final_outputs


def pick_best_faces(detection_results, num):
    if len(detection_results) == 0:
        return []

    # Trivial solution: just pick the largest faces
    bbox_areas = []
    for r in detection_results:
        area = (r.box[2] - r.box[0]) * (r.box[3] - r.box[1])
        bbox_areas.append(area)
    bbox_areas = np.array(bbox_areas)
    best_inds = np.argpartition(bbox_areas, -num, )[-num:]

    # Collect selected faces
    ret = []
    for ind in best_inds:
        ret.append(detection_results[ind])
    return ret


def write_output(result_map, anno, skiplist):
    n_skipped = 0
    for img_name, results in result_map.items():
        if len(results) > 0:
            # Output faces in the following format
            # 0--Parade/0_Parade_marchingband_1_799.jpg (image name)
            # 22 (num of faces)
            # 78 221 85 229 (xmin, ymin, xmax, ymax)
            # 78 238 92 255
            anno.write(img_name + '\n')
            anno.write(str(len(results)) + '\n')
            for r in results:
                anno.write('{:.1f} {:.1f} {:.1f} {:.1f}\n'.format(*r.box))
        else:
            print("Image {} doesn't have any face detected, ignored.".format(img_name))
            n_skipped += 1
            skiplist.write(img_name + '\n')
    print("{} images are skipped due to no face detected.".format(n_skipped))


def generate_face_bbox(args):
    n_visualize = args.n_visualize
    face_detector = get_detector(args.face_model, args.face_ckpt, args.face_config)

    # Input/output folder and file paths
    image_dir = os.path.join(args.root, args.image_dir)
    outdir = os.path.join(args.root, args.out_dir)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    output_file = os.path.join(args.root, args.out_dir, args.out_file)
    skiplist = os.path.join(args.root, args.out_dir, args.skip_list)

    def process_image(detector, img_path):
        # Detect faces
        image_bgr = cv2.imread(img_path)
        face_results = detector.detect(image_bgr, rgb=False)

        # Select best face to ignore false positives, such as faces on the clothes
        best_faces = pick_best_faces(face_results, args.max_num_faces)

        # Visualize detected faces for quick verification
        nonlocal n_visualize
        if n_visualize > 0:
            image_bgr = draw_final_outputs(image_bgr, best_faces, show_ids=face_detector.get_class_ids())
            cv2.imshow('face detection', image_bgr)
            cv2.waitKey(0)
            n_visualize -= 1

        return best_faces

    result_map = {}
    if args.mode == 'all':
        # Walk into every subdirectory of image directory and process every image files, preserving its relative path
        for root, dirs, files in os.walk(image_dir):
            for filename in files:
                if not filename.endswith(tuple(args.image_ext.split("|"))):  # Ignore non image files
                    continue

                relative_path = os.path.join(root.replace(image_dir, ""), filename)[1:]
                print("Processing image {}".format(relative_path))
                result_map[relative_path] = process_image(face_detector, os.path.join(root, filename))
    elif args.mode == 'skip':
        # When dealing with skipped images, just read filename from skiplist file and process them one by one
        with open(skiplist, 'r') as f:
            for line in f.readlines():
                filename = line.strip()
                print("Processing image {}".format(filename))
                result_map[filename] = process_image(face_detector, os.path.join(image_dir, filename))
    else:
        raise Exception("Non supported mode {}".format(args.mode))

    print("Writing output")
    with open(skiplist, 'w') as s:
        if args.mode == 'all':
            with open(output_file, 'w') as o:
                write_output(result_map, o, s)
        elif args.mode == 'skip':
            # Append detection results of previously skipped images to the end of annotation file
            with open(output_file, 'a') as o:
                write_output(result_map, o, s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode',
        type=str,
        default='all',
        help="all | skip"
    )
    parser.add_argument(
        '--root',
        required=True,
        type=str,
        help="Root directory to dataset")
    parser.add_argument(
        '--image_dir',
        default='Images',
        type=str,
        help="Image directory relative to root directory of dataset"
    )
    parser.add_argument(
        '--image_ext',
        default='jpg|png',
        type=str,
        help="Image extensions to be supported, separated by '|'."
    )
    parser.add_argument(
        '--out_dir',
        default='Annotations',
        type=str,
        help="Output directory relative to root directory of dataset"
    )
    parser.add_argument(
        '--out_file',
        default="bbox_gt.txt",
        type=str,
        help="Filename for output bounding box file"
    )
    parser.add_argument(
        '--skip_list',
        default="skiplist.txt",
        type=str,
        help="List files for skipped images (which means no face are detected)."
    )
    parser.add_argument(
        '--max_num_faces',
        default=1,
        type=int,
        help="Maximum number of faces to output for each image. If 0 or negative value is specified, all detected faces will be output"
    )
    parser.add_argument(
        '--n_visualize',
        default=0,
        type=int,
        help="Number of images to be visualized before going on."
    )
    parser.add_argument(
        '--face_model',
        type=str,
        help='tensorpack | s3fd | tf-model')
    parser.add_argument(
        '--face_ckpt',
        default='',
        type=str,
        help='Checkpoint of face detection model')
    parser.add_argument(
        '--face_config',
        default='',
        type=str,
        help='Configurations of face detection model',
        nargs='+'
    )
    args = parser.parse_args()

    generate_face_bbox(args)
