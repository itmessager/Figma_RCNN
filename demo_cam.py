import numpy as np
import time

import cv2
import argparse

from PIL import ImageDraw, Image

from detection.core.detector_factory import get_detector
from tracking.tracker import PersonTracker
from attributer.attributer import AllInOneAttributer, Attributer
from utils.viz_utils import draw_tracked_people, draw_person_attributes


def run(init_func, process_func, args, cam=None, video=None):
    if cam:
        # Read camera
        cap = cv2.VideoCapture(0)
    elif video:
        # Read video
        cap = cv2.VideoCapture(video)
    else:
        raise Exception("Either cam or video need to be specified as input")

    # Initialize model
    width, height = cap.get(3), cap.get(4)
    print((width, height))
    models = init_func(args, width, height)

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
            k = cv2.waitKey(1)
            if k == 27:  # Esc key to stop
                break


# generate models
def init_detector_func(args, width, height):
    face_detector = get_detector(args.face_model, args.face_ckpt, args.face_config)
    obj_detector = get_detector(args.obj_model, args.obj_ckpt, args.obj_config)
    tracker = PersonTracker()
    attributer = Attributer(AllInOneAttributer(args))
    return (face_detector, obj_detector, tracker, attributer)

# use models to detect
def process_detector_func(models, image_bgr):
    # Get models
    face_detector, obj_detector, tracker, attributer = models

    # Perform detection
    face_results = face_detector.detect(image_bgr, rgb=False)
    obj_results = obj_detector.detect(image_bgr, rgb=False)
    people_results = [r for r in obj_results if r.class_id == 1]  # Extract person detection result

    # Tracking
    tracked_people, removed_ids = tracker.update(face_results, people_results, image_bgr, rgb=False)

    # Calculate people's attributes
    attributer.remove_people(removed_ids)
    people = [(p.id, p.face_box, p.body_box) for p in tracked_people]  # Convert to attributer accepted format
    people_attrs = attributer(image_bgr, people)

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
        '--model',
        default='all_in_one',
        type=str,
        help='all_in_one | hyperface')
    parser.add_argument(
        '--conv',
        default='resnet18',
        type=str)
    parser.add_argument(
        '--checkpoint',
        default='',
        type=str,
        help='Save data (.pth) of previous training')
    parser.add_argument(
        '--pretrain', action='store_true', help='Whether to use pretrained weights in conv models')
    args = parser.parse_args()

    run(init_detector_func, process_detector_func, args, args.cam, args.video)
