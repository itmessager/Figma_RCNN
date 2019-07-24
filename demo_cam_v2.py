import cv2
import argparse
import numpy as np
import time

from PIL import Image
from utils.viz_utils_en import draw_tracked_people


def run(process_func, args, cam=None, video=None, image=None):
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
    assert width, "Load resource failed!"
    models = get_detector(args.load_ckpt, args.config)

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


def get_detector(weight_file, config):
    from detection.tensorpacks.inference import AttributeDetector
    from detection.config.config import config as cfg
    if config:
        cfg.update_args(config)
    return AttributeDetector(weight_file)


# use models to detect
def process_detector_func(models, image_bgr):
    # Perform detection
    person_results = models.detect(image_bgr, rgb=False)
    image_disp = draw_tracked_people(image_bgr, person_results)

    # Draw attribute results
    image_pil = Image.fromarray(cv2.cvtColor(image_disp, cv2.COLOR_BGR2RGB))
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
        '--load_ckpt',
        default='',
        type=str,
        help='Checkpoint of object detection model')
    parser.add_argument(
        '--config',
        default='',
        type=str,
        help='Configurations of object detection model',
        nargs='+'
    )
    parser.add_argument(
        '--pretrain', action='store_true', help='Whether to use pretrained weights in conv models')
    args = parser.parse_args()

    run(process_detector_func, args, args.cam, args.video, args.image)

'''
--image
/root/datasets/img-folder/1.jpg
--cam
0
--load_ckpt
/root/datasets/0601/checkpoint
--config
DATA.BASEDIR=/root/datasets/COCO/DIR

'''
