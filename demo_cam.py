# coding=utf-8
import argparse
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from attributer import AllInOneAttributer
from detection import S3fdFaceDetector
from utils.drawing import draw_bounding_box_pil
from gaze.gaze_estimator import estimate_gaze_from_headpose
from tracking.sort import Sort


# Entity class holding attributes of each person
# TODO Separate static attributes from dynamic attributes like head pose, as they tend to follow different update logics
from utils.viz_utils import draw_person_attributes


class Person1:
    age = None
    gender = None
    eyeglasses = False
    receding_hairline = False
    smiling = False
    last_update = 0
    head_yaw = None
    head_pitch = None
    head_roll = None
    gazing_at_screen = False
    total_gaze_time = 0
    total_gaze_number = 0

    def __init__(self, id):
        self.id = id

    def update(self, attributes):
        age, gender, eyeglasses, receding_hairline, smiling, head_yaw, head_pitch, head_roll = attributes
        self.age = age[0]
        self.gender = gender
        self.eyeglasses = eyeglasses
        self.receding_hairline = receding_hairline
        self.smiling = smiling
        self.head_yaw = head_yaw
        self.head_pitch = head_pitch
        self.head_roll = head_roll

        # Update the status of whether the person is gazing at screen and the total gazing time
        gazing = estimate_gaze_from_headpose(head_yaw, head_pitch, head_roll)
        if self.gazing_at_screen is True and gazing:
            self.total_gaze_time += time.time() - self.last_update
        if self.gazing_at_screen is False and gazing: # Switch from not gazing to gazing
            self.total_gaze_number += 1
        self.gazing_at_screen = gazing

        self.last_update = time.time()


class Person:
    age = None
    gender = None
    eyeglasses = False
    receding_hairline = False
    smiling = False
    last_update = 0
    head_yaw = None
    head_pitch = None
    head_roll = None
    gazing_at_screen = False
    total_gaze_time = 0
    total_gaze_number = 0

    def __init__(self, id):
        self.id = id
        self.pool_class_list = [-1, 2, 2, 2]
        self.attr_num = 4
        self.pool_num = 20
        self.pool = self.generate_pool()

    def update(self, attributes):
        age, gender, eyeglasses, receding_hairline, smiling, head_yaw, head_pitch, head_roll = attributes
        self.pool_update(attributes[:4])
        self.age, self.gender, self.eyeglasses, self.receding_hairline = self.weight_result()
        self.smiling = smiling
        self.head_yaw = head_yaw
        self.head_pitch = head_pitch
        self.head_roll = head_roll

        # Update the status of whether the person is gazing at screen and the total gazing time
        gazing = estimate_gaze_from_headpose(head_yaw, head_pitch, head_roll)
        if self.gazing_at_screen is True and gazing:
            self.total_gaze_time += time.time() - self.last_update
        if self.gazing_at_screen is False and gazing: # Switch from not gazing to gazing
            self.total_gaze_number += 1
        self.gazing_at_screen = gazing

        self.last_update = time.time()

    def generate_pool(self):
        # [calss_num, prob]
        return [[] for _ in range(self.attr_num)]

    def pool_update(self, value):
        for i in range(self.attr_num):
            if self.pool_class_list[i] == -1:
                if len(self.pool[i]) < self.pool_num:
                    self.pool[i].append(value[i])
                else:
                    self.pool[i].pop(0)
                    self.pool[i].append(value[i])
                pass
            else:
                if len(self.pool[i]) < self.pool_num:
                    self.pool[i].append(value[i])
                    self.pool[i] = sorted(self.pool[i], key=lambda attr: attr[1])
                elif value[i][1] > self.pool[i][0][1]:
                        self.pool[i].pop(0)
                        self.pool[i].append(value[i])
                        self.pool[i] = sorted(self.pool[i], key=lambda attr: attr[1])

    def weight_result(self):
        classification_list = []
        for i in range(self.attr_num):
            result_scores = []
            if self.pool_class_list[i] == -1:
                if len(self.pool[i]) != 0:
                    classification_list.append(sum(self.pool[i]) / len(self.pool[i]))
                else:
                    classification_list.append(0)
            else:
                for j in range(self.pool_class_list[i]):
                    prob = [class_attr[1] for class_attr in self.pool[i] if class_attr[0] == j]
                    # get the max prob index, if max has more than one value, return the first one
                    if prob:
                        sum_num = 0
                        #scale = 10 * (prob - 1/self.pool_class_list[i])
                        for k in range(len(prob)):
                            scale = (prob[k] - 1/self.pool_class_list[i]) * 10
                            sum_num += pow(prob[k], scale)
                        result_scores.append(sum_num)
                    else:
                        # if prob is null, return 0
                        result_scores.append(0)
                # get the attr classification
                classification = np.uint8(result_scores.index(max(result_scores)))
                classification_list.append(classification)
        return tuple(classification_list)


def main(gaze_opt, attr_opt):
    # gaze_estimator = GazeEstimator(gaze_opt)
    detector = S3fdFaceDetector()
    fa = AllInOneAttributer(attr_opt)
    tracker = Sort(max_age=3, max_trajectory_len=20)
    update_attr_interval = 0.03 # Attribute update interval in seconds

    people = {}

    # Read video by opencv
    cap = cv2.VideoCapture('/root/models/detection/output/2018-08-30-155619.webm')
    width, height = cap.get(3), cap.get(4)
    print((width, height))

    cv2.namedWindow("video", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("video", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        grabbed, image_bgr = cap.read()

        if not grabbed:
            break
        # Some algorithms only take RGB image, and possibly only in PIL format
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        # Detect and track-by-detection
        faces = detector.detect(image_bgr)
        tracked_faces = tracker.update(faces)

        if len(tracked_faces) > 0:
            faces = tracked_faces[..., 0:4]
            ids = tracked_faces[..., 5]

            # Update the list of tracked people and decide whether to update attributes
            cur_time = time.time()
            faces_to_detect_attr = []
            ids_to_detect_attr = []
            for id, f in zip(ids, faces):
                if id not in people:
                    people[id] = Person(id)
                if people[id].last_update + update_attr_interval < cur_time:
                    faces_to_detect_attr.append(f)
                    ids_to_detect_attr.append(id)

            # Detect and update attributes for faces that are necessary to be updated
            if len(faces_to_detect_attr) > 0:
                attributes = fa(image_pil, faces_to_detect_attr) # In RGB color space

                # Update attributes
                for id, a in zip(ids_to_detect_attr, attributes):
                    people[id].update(a)

            # gaze_targets = gaze_estimator(image_pil, [(f[0], f[1], f[2] - f[0], f[3] - f[1]) for f in faces])

            # Remove stale people
            valid_ids = set(tracker.get_valid_ids())
            for id in people.keys() - valid_ids:
                people.pop(id)

            draw = ImageDraw.Draw(image_pil)
            # for f, id, g in zip(faces, ids, gaze_targets):
            for f, id in zip(faces, ids):
                # bbox_color = (255, 0, 0) if people[id].gender == 0 else (0, 0, 255)
                bbox_color = (255, 0, 0)
                draw_person_attributes(draw, people[id], f, f)
                draw_bounding_box_pil(f, draw, bbox_color)
                # draw_gaze_target_pil((int((g[0] - 1280) / 3 + 320), int(g[1] / 3)), draw, (0, 0, 255))
        image_rgb = np.array(image_pil)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow('video', image_bgr)
        k = cv2.waitKey(1)
        if k == 27:  # Esc key to stop
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gaze_model',
        default='facegaze',
        type=str,
        help='facegaze | eyegaze')
    parser.add_argument(
        '--gaze_conv',
        default='resnet18',
        type=str)
    parser.add_argument(
        '--gaze_checkpoint',
        default='',
        type=str,
        help='Save data (.pth) of previous training')
    parser.add_argument(
        '--attribute_model',
        default='all_in_one',
        type=str,
        help='all_in_one')
    parser.add_argument(
        '--attribute_conv',
        default='resnet18',
        type=str)
    parser.add_argument(
        '--attribute_checkpoint',
        default='',
        type=str,
        help='Save data (.pth) of previous training')

    opt = parser.parse_args()

    gaze_opt, attr_opt = argparse.Namespace(), argparse.Namespace()
    gaze_opt.model = opt.gaze_model
    gaze_opt.conv = opt.gaze_conv
    gaze_opt.checkpoint = opt.gaze_checkpoint
    gaze_opt.pretrain = False
    attr_opt.model = opt.attribute_model
    attr_opt.conv = opt.attribute_conv
    attr_opt.checkpoint = opt.attribute_checkpoint
    attr_opt.pretrain = False

    main(gaze_opt, attr_opt)
