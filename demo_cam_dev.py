import numpy as np
import time

import cv2
import argparse

from PIL import ImageDraw, Image

from tracking.tracker import PersonTracker
from attributer.attributer import PersonAttrs, PersonBoxes
from utils.viz_utils import draw_tracked_people, draw_person_attributes


def run(init_models, process_func, args, cam=None, video=None, image=None):
    if cam:
        # Read camera
        cap = cv2.VideoCapture(0)
    elif video:
        # Read video
        cap = cv2.VideoCapture(video)
    elif image:
        cap = cv2.VideoCapture(image)
    else:
        raise Exception("Either cam or video need to be specified as input")

    # Initialize model
    width, height = cap.get(3), cap.get(4)
    print((width, height))
    models = init_models(args)

    # cv2.namedWindow("video", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("video", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    frame_count = 0

    while True:

        grabbed, image_bgr = cap.read()

        if not grabbed:
            break

        frame_count += 1
        t = time.time()
        img_to_show = process_func(models, image_bgr)
        print("Process frame {} takes {}s".format(frame_count, time.time() - t))

        if img_to_show is not None:
            cv2.imshow('video', img_to_show)
            if image:
                cv2.waitKey(0)
            elif video or cam:
                k = cv2.waitKey(1)
                if k == 27:  # Esc key to stop
                    break

from detection.core.tensorpack_detector import TensorPackDetector
#from detection.tensorpacks.tensorpack_detector_dev import TensorPackDetector

def get_detector(weight_file, config):
    from detection.config.tensorpack_config import config as cfg
    if config:
        cfg.update_args(config)
    return TensorPackDetector(weight_file)


# generate models
def init_models(args):
    obj_detector = get_detector(args.obj_ckpt, args.obj_config)
    tracker = PersonTracker()
    return (obj_detector, tracker)

# use models to detect
def process_detector_func(models, image_bgr):
    # Get models
    obj_detector, tracker = models

    # Perform detection
    person_results = obj_detector.detect(image_bgr, rgb=False)

    # Tracking
    people_boxes = [PersonBoxes(r) for r in person_results]
    # Calculate people's attributes
    people_attrs = [PersonAttrs(r) for r in person_results]
    # Draw detection and tracking results
    image_disp = draw_tracked_people(image_bgr, people_boxes)

    # Draw attribute results
    image_pil = Image.fromarray(cv2.cvtColor(image_disp, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    for trk, attr in zip(people_boxes, people_attrs):
        draw_person_attributes(draw, attr, trk.body_box)

    image_disp = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    return image_disp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--video',
        type=str,
        help="Run prediction on a given video. "
             "This argument is the path to the input video file")
    parser.add_argument(
        '--image',
        type=str,
        help="Run prediction on a given image. "
             "This argument is the path to the input image file")
    parser.add_argument(
        '--cam',
        type=int,
        help='Specify which camera to detect.')
    parser.add_argument(
        '--obj_model',
        default='tensorpack',
        type=str,
        help='tensorpack | tf-model')
    parser.add_argument(
        '--obj_ckpt',
        default='',
        type=str,
        help='Checkpoint of object detection model')
    parser.add_argument(
        '--obj_config',
        default='',
        type=str,
        help='Configurations of object detection model',
        nargs='+'
    )
    parser.add_argument(
        '--pretrain', action='store_true', help='Whether to use pretrained weights in conv models')
    args = parser.parse_args()

    run(init_models, process_detector_func, args, args.cam, args.video,args.image)

'''

--image
/root/datasets/img-folder/1.jpg
--cam
0
--obj_model
tensorpack
--obj_ckpt
/home/Figma_RCNN/detection/tensorpacks/train_log/figmarcnn/checkpoint
--obj_config
DATA.BASEDIR=/root/datasets/COCO/DIR

'''
