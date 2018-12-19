# Refer to https://github.com/natanielruiz/deep-head-pose/blob/master/code/datasets.py

import os

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data.dataset import Dataset
from attributer.attributes import FaceAttributes as FA, AttributeType as AT, Attribute

import scipy.io as sio  #use to read mat


def get_list_from_filenames(file_path):
    # input:    relative path to .txt file with file names
    # output:   list of relative path names
    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines


def get_pose_params_from_mat(mat_path):
    # This functions gets the pose parameters from the .mat
    # Annotations that come with the Pose_300W_LP dataset.
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll, tdx, tdy]
    pose_params = pre_pose_params[:5]
    return pose_params


def get_ypr_from_mat(mat_path):
    # Get yaw, pitch, roll from .mat annotation.
    # They are in radians
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll]
    pose_params = pre_pose_params[:3]
    return pose_params


def get_pt2d_from_mat(mat_path):
    # Get 2D landmarks
    mat = sio.loadmat(mat_path)
    pt2d = mat['pt2d']
    return pt2d


class Pose_300W_LP(Dataset):
    _attrs = [Attribute(FA.HEAD_YAW_BIN, AT.MULTICLASS), Attribute(FA.HEAD_PITCH_BIN, AT.MULTICLASS),
              Attribute(FA.HEAD_ROLL_BIN, AT.MULTICLASS), Attribute(FA.HEAD_YAW, AT.NUMERICAL),
              Attribute(FA.HEAD_PITCH, AT.NUMERICAL), Attribute(FA.HEAD_ROLL, AT.NUMERICAL)]

    # Head pose from 300W-LP dataset
    def __init__(self, data_dir, transform=None, target_transform=None, training=True, img_ext='.jpg', annot_ext='.mat',
                 image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext
        self.training = training

        filename_list = get_list_from_filenames(os.path.join(data_dir, '300W_LP_filename_filtered.txt'))

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.X_train[index] + self.img_ext))
        img = img.convert(self.image_mode)
        mat_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)

        # Crop the face loosely
        pt2d = get_pt2d_from_mat(mat_path)
        x_min = min(pt2d[0, :])
        y_min = min(pt2d[1, :])
        x_max = max(pt2d[0, :])
        y_max = max(pt2d[1, :])

        # k = 0.2 to 0.40
        k = np.random.random_sample() * 0.2 + 0.2
        x_min -= 0.6 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 0.6 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # We get the pose in radians
        pose = get_ypr_from_mat(mat_path)
        # And convert to degrees.
        pitch = pose[0] * 180 / np.pi
        yaw = pose[1] * 180 / np.pi
        roll = pose[2] * 180 / np.pi

        if self.training:
            # Flip?
            rnd = np.random.random_sample()
            if rnd < 0.5:
                yaw = -yaw
                roll = -roll
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            # Blur?
            rnd = np.random.random_sample()
            if rnd < 0.05:
                img = img.filter(ImageFilter.BLUR)

        # Bin values
        bins = np.array(range(-99, 102, 3))

        # why -1 ???
        binned_pose = np.digitize([yaw, pitch, roll], bins) - 1

        # Get target tensors
        labels = binned_pose
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)

        target = {FA.HEAD_YAW_BIN: int(binned_pose[0]), FA.HEAD_PITCH_BIN: int(binned_pose[1]),
                  FA.HEAD_ROLL_BIN: int(binned_pose[2]), FA.HEAD_YAW: yaw, FA.HEAD_PITCH: pitch, FA.HEAD_ROLL: roll}

        if self.target_transform is not None:
            target = self.target_transform(target)

        # return img, labels, cont_labels, self.X_train[index]
        return (img, target)

    def __len__(self):
        # 122,450
        return self.length

    @classmethod
    def list_attributes(cls):
        return cls._attrs
