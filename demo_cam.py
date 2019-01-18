import numpy as np
import time

import cv2
import argparse

from PIL import ImageDraw, Image

from detection.core.detector_factory import get_detector
from tracking.tracker import PersonTracker
from attributer.attributer import PersonAttrs
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

# generate models
def init_models(args):
    face_detector = get_detector(args.face_model, args.face_ckpt, args.face_config)
    obj_detector = get_detector(args.obj_model, args.obj_ckpt, args.obj_config)
    tracker = PersonTracker()
    return (face_detector, obj_detector, tracker)

# use models to detect
def process_detector_func(models, image_bgr):
    # Get models
    face_detector, obj_detector, tracker = models

    # Perform detection
    face_results = face_detector.detect(image_bgr, rgb=False)
    person_results = obj_detector.detect(image_bgr, rgb=False)
    #people_results = [r for r in obj_results if r.class_id == 1]  # Extract person detection result

    # Tracking
    tracked_people, removed_ids = tracker.update(face_results, person_results, image_bgr, rgb=False)
    # Calculate people's attributes
    people_attrs = [PersonAttrs(r) for r in person_results]
    # Draw detection and tracking results
    image_disp = draw_tracked_people(image_bgr, tracked_people)

    # Draw attribute results
    image_pil = Image.fromarray(cv2.cvtColor(image_disp, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    for trk, attr in zip(tracked_people, people_attrs):
        draw_person_attributes(draw, attr, trk.face_box, trk.body_box)

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
        '--face_model',
        default='s3fd',
        type=str,
        help='s3fd | tf-model')
    parser.add_argument(
        '--obj_model',
        default='tensorpack',
        type=str,
        help='tensorpack | tf-model')
    parser.add_argument(
        '--face_ckpt',
        default='',
        type=str,
        help='Checkpoint of face detection model')
    parser.add_argument(
        '--obj_ckpt',
        default='',
        type=str,
        help='Checkpoint of object detection model')
    parser.add_argument(
        '--face_config',
        default='',
        type=str,
        help='Configurations of face detection model',
        nargs='+'
    )
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

--video
/root/datasets/2018-08-30-170107.webm
--cam
0
--face_model
s3fd
--face_ckpt
/root/datasets/s3fd_convert.pth
--obj_model
tensorpack
--obj_ckpt
/home/Figma_RCNN/detection/tensorpacks/train_log/maskrcnn/checkpoint
--obj_config
DATA.BASEDIR=/root/datasets/COCO/DIR

'''
