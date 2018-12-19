import os
import time

import cv2
import argparse

from tensorpack.utils.viz import interactive_imshow

from detection.core.detector_factory import get_detector
from detection.tensorpacks.viz import draw_final_outputs


def run(init_func, process_func, args):
    if args.cam:
        # Read camera
        process_video(init_func, process_func, cv2.VideoCapture(args.cam), args)
    elif args.video:
        # Read video
        process_video(init_func, process_func, cv2.VideoCapture(args.video), args)
    elif args.img_folder:
        process_image(init_func, process_func, args)
    else:
        raise Exception("Either one of cam|video|img_folder need to be specified as input")


def process_video(init_func, process_func, cap, args):
    # Initialize model
    width, height = cap.get(3), cap.get(4)
    print((width, height))
    models = init_func(args)

    # cv2.namedWindow("video", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("video", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    frame_count = 0

    while True:
        grabbed, image_bgr = cap.read()

        if not grabbed:
            break

        frame_count += 1
        t = time.time()
        img_to_show = process_func(models, image_bgr, args)
        print("Process frame {} takes {}s".format(frame_count, time.time() - t))

        if img_to_show is not None:
            cv2.imshow('video', img_to_show)
            k = cv2.waitKey(1)
            if k == 27:  # Esc key to stop
                break


def process_image(init_func, process_func, args):
    models = init_func(args)

    images = [file for file in os.listdir(args.img_folder) if file.lower().endswith(('.jpg', '.png', '.jpeg')) ]
    for img in images:
        image_bgr = cv2.imread(os.path.join(args.img_folder, img))

        t = time.time()
        img_to_show = process_func(models, image_bgr, args)
        print("Process {} takes {}s".format(img, time.time() - t))

        if img_to_show is not None:
            interactive_imshow(img_to_show)


def init_detector_func(args):
    face_detector, obj_detector = None, None
    if args.face_model:
        face_detector = get_detector(args.face_model, args.face_ckpt, args.face_config)
    if args.obj_model:
        obj_detector = get_detector(args.obj_model, args.obj_ckpt, args.obj_config)
    return (face_detector, obj_detector)


def process_detector_func(models, image_bgr, args):
    face_detector, obj_detector = models
    if face_detector:
        face_results = face_detector.detect(image_bgr, rgb=False)
    if obj_detector:
        obj_results = obj_detector.detect(image_bgr, rgb=False)

    # Draw object detection results first
    if obj_detector:
        if args.show_obj_classes:
            show_ids = set([int(cls) for cls in args.show_obj_classes.split(",")])
        else:
            show_ids = obj_detector.get_class_ids()
        image_bgr = draw_final_outputs(image_bgr, obj_results, show_ids=show_ids)
    if face_detector:
        image_bgr = draw_final_outputs(image_bgr, face_results, show_ids=face_detector.get_class_ids())

    return image_bgr


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
        '--img_folder',
        type=str,
        help="Specify path to a folder of images to be tested. Use only one type of input among 'video|cam|img_folder'."
    )
    parser.add_argument(
        '--face_model',
        type=str,
        help='s3fd | tf-model')
    parser.add_argument(
        '--obj_model',
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
        '--show_obj_classes',
        type=str,
        help='What classes will be shown among all supported classes. If not specified, all classes will be displayed.'
    )
    args = parser.parse_args()

    run(init_detector_func, process_detector_func, args)
