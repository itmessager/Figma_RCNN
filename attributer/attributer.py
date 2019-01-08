import time
from abc import abstractmethod
import logging

import cv2
import torch
import numpy as np

import torch.nn.functional as F
from PIL import Image
from torch.utils.data.dataloader import default_collate
from ignite._utils import convert_tensor
from attributer.model import generate_model
from attributer.transforms import get_inference_transform
from attributer.attributes import FaceAttributes as A
from gaze.gaze_estimator import estimate_gaze_from_headpose


class PersonAttrs:
    def __init__(self, attribute):
        self.facemask = attribute.facemask
        self.formal = attribute.formal
        self.hat = attribute.hat
        self.jeans = attribute.jeans
        self.logo = attribute.logo
        self.longhair = attribute.longhair
        self.longpants = attribute.longpants
        self.longsleeve = attribute.longsleeve
        self.male = attribute.male
        self.shorts = attribute.shorts
        self.skirt = attribute.skirt
        self.stripe = attribute.stripe
        self.sunglass = attribute.sunglass
        self.tshirt = attribute.tshirt

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
            # else:
            #     if len(self.pool[i]) < self.pool_num:
            #         self.pool[i].append(value[i])
            #         self.pool[i] = sorted(self.pool[i], key=lambda attr: attr[1])
            #     elif value[i][1] > self.pool[i][0][1]:
            #             self.pool[i].pop(0)
            #             self.pool[i].append(value[i])
            #             self.pool[i] = sorted(self.pool[i], key=lambda attr: attr[1])
            else:
                if len(self.pool[i]) < self.pool_num:
                    self.pool[i].append(value[i])
                    #self.pool[i] = sorted(self.pool[i], key=lambda attr: attr[1])
                else:
                        self.pool[i].pop(0)
                        self.pool[i].append(value[i])
                        #self.pool[i] = sorted(self.pool[i], key=lambda attr: attr[1])
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
                            sum_num += pow(2, scale)
                        result_scores.append(sum_num)
                    else:
                        # if prob is null, return 0
                        result_scores.append(0)
                # get the attr classification
                classification = np.uint8(result_scores.index(max(result_scores)))
                classification_list.append(classification)
        return tuple(classification_list)


# TODO Make it more flexible to support different configurations on face and body model.
class Attributer:
    def __init__(self, face_model):
        self.face_model = face_model
        self.people = {}

    def __call__(self, img, people, rgb=False):
        """
        Estimate attributes for each of the input person based on image content. Also track each person's information
        through time for more accurate estimation.
        :param img: Original image whereby people are detected and tracked.
        :param people: a list of tuple (id, face_box, body_box) where face_box/body_box can either be None or a tuple of
        (xmin, ymin, xmax, ymax) on the original image
        :param rgb: incidate whether the image, as a numpy array, is in RGB or BGR space
        """
        # Convert to PIL image
        if not rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(img)

        # Estimate attribute for the current image
        faces_to_estimate = []
        ind_to_estimate = []
        for i, (id, face_box, body_box) in enumerate(people):
            if face_box is not None:
                faces_to_estimate.append(face_box)
                ind_to_estimate.append(i)
        attrs_cur_frame = self.face_model(image_pil, faces_to_estimate)

        # Update each person's attributes with the newest estimation
        all_attributes = [None] * len(people)
        for i, ind in enumerate(ind_to_estimate):
            all_attributes[ind] = attrs_cur_frame[i]
        ids = [id for (id, face_box, body_box) in people]
        self.update_attributes([(id, attrs) for id, attrs in zip(ids, all_attributes)])

        updated_results = [self.people[id] for id in ids]
        return updated_results

    def remove_people(self, ids):
        if not isinstance(ids, list):
            ids = [ids]
        for id in ids:
            self.people.pop(id)

    def update_attributes(self, new_results):
        for id, attrs in new_results:
            if id not in self.people:
                self.people[id] = Person(id)
            if attrs is not None:  # Attributes can be None if face is not detected.
                self.people[id].update(attrs)


class AllInOneAttributer(object):
    # TODO Consolidate separate Attribute related definitions
    thresh = {A.GENDER: 0.5, A.RECEDING_HAIRLINES: 0.5, A.EYEGLASSES: 0.5, A.SMILING: 0.5}
    attr_ret = [A.AGE, A.GENDER, A.RECEDING_HAIRLINES, A.EYEGLASSES, A.SMILING, A.HEAD_YAW, A.HEAD_PITCH, A.HEAD_ROLL]

    age_idx = 0
    age_mean = 50
    age_std = 50

    def unnormalize_age(self, ages):
        return ages * self.age_std + self.age_mean

    def __init__(self, opt):
        model, parameters, mean, std = generate_model(opt) # args return object of nn.Module,weight and bias of objct
        # parameters is useless
        logging.info('loading checkpoint {}'.format(opt.checkpoint))
        checkpoint = torch.load(opt.checkpoint)
        model.load_state_dict(checkpoint['state_dict']) #load model's weight and bias
        model.eval()

        self.model = model
        self.transform = get_inference_transform(mean, std)

    def __call__(self, img_pil, faces):
        if len(faces) == 0:
            return []
        crops = [self.transform((img_pil, face)) for face in faces]
        n_samples = len(crops)

        # Batch crops to Tensor
        batch = default_collate(crops)
        batch = convert_tensor(batch, device='cuda')

        # Feed-forward
        preds = self.model(batch)  # [n_attributes, n_face_samples]

        # Convert each attribute to desired output format
        converted_preds = [] # [n_attributes, n_face_samples]
        for i, attr in enumerate(A):
            p = preds[i]
            if attr == A.AGE:
                p = self.unnormalize_age(p) # Post-process age
            elif attr == A.SMILING:
                p = F.softmax(p, dim=1)
                p = (p[:, 1] > self.thresh[attr])
            elif attr in self.thresh:
                p = F.softmax(p, dim=1)
                classification = (p[:, 1] > self.thresh[attr]).float().unsqueeze(1)
                prob = torch.max(p, 1)[0].unsqueeze(1)
                p = torch.cat([classification, prob], 1)
                #(p[:, 1] > self.thresh[attr]), p.max(1)))
            elif attr not in [A.HEAD_ROLL, A.HEAD_PITCH, A.HEAD_YAW]:
                continue
            converted_preds.append(p.data.cpu().numpy())

        # Transpose results so that output will have size [n_face_samples, n_attributes]
        ret = []
        for i in range(n_samples):
            ret.append([])
            # Only return desired output results. E.g ignoring HEAD_YAW_BIN
            for j, _ in enumerate(self.attr_ret):
                ret[i].append(converted_preds[j][i])
        return ret