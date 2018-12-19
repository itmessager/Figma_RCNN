import logging
import torch
import numpy as np
from math import sin, cos, asin, sqrt
from torch.utils.data.dataloader import default_collate
from gaze.model import generate_model
from gaze.transforms import get_transform
from gaze.utils.coordinates import cam2screen
from ignite._utils import convert_tensor


class GazeEstimator(object):
    # w_screen_cm, h_screen_cm, w_screen_pixels, h_screen_pixels, dx_cm, dy_cm
    device_params = (30.9, 17.3, 2560, 1440, 15.45, 0.87)

    def __init__(self, opt):
        model, parameters = generate_model(opt)

        logging.info('loading checkpoint {}'.format(opt.checkpoint))
        checkpoint = torch.load(opt.checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        self.model = model

        self.transform = get_transform(mode='inference')

    def __call__(self, img, faces):
        if len(faces) == 0:
            return []
        # crop, face, target = self.transform((img, [face], img.size, target))
        # return ((*crop, *face), target)
        t = [self.transform((img, [f], img.size)) for f in faces]
        # t = self.transform((img, faces, img.size))
        t = [(*x[0], *x[1]) for x in t]
        batch = default_collate(t)
        batch = convert_tensor(batch, device='cuda')
        preds = self.model(batch).data.cpu()

        gaze_targets = [cam2screen(p[0], p[1], self.device_params) for p in preds]

        return gaze_targets


def estimate_gaze_from_headpose(yaw, pitch, roll):
    """
    Estimate whether a person is gazing the screen by his/her head pose
    :param yaw: Yaw of head pose in angular degree
    :param pitch: Pitch of head pose in angular degree
    :param roll: Roll of head pose in angular degree
    """
    # Convert from angular to radian degree
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    # Angular between Z-Axis (out of the screen) and
    x_projected = sin(yaw)
    y_projected = -cos(yaw) * sin(pitch)

    #what the z_angle present???
    z_angle = asin(sqrt(x_projected ** 2+y_projected ** 2))

    # Using a fixed angular threshold for now
    if z_angle * 180 / np.pi > 15:
        return False
    else:
        return True

