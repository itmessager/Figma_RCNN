import os
import os.path
import ujson
from glob import glob

import torch.utils.data as data
from torchvision.datasets.folder import pil_loader


class GazeCapture(data.Dataset):

    def __init__(self, root, subset, transform=None, use_eye=False):
        self.data = self._make_dataset(root, subset)
        self.use_eye = use_eye

        self.transform = transform
        self.img_loader = pil_loader

    @staticmethod
    def _make_dataset(root, subset):
        assert subset in ['train', 'val', 'test.py']

        if not root.endswith('/'):
            root = root + '/'

        dataset = []
        for subject_folder in glob(root + '*/'):
            with open(os.path.join(subject_folder, 'info.json')) as json_file:
                info = ujson.load(json_file)
            if info['Dataset'] != subset:
                continue
            n_frames = info['TotalFrames']

            with open(os.path.join(subject_folder, 'appleFace.json')) as json_file:
                faces = ujson.load(json_file)

            with open(os.path.join(subject_folder, 'appleLeftEye.json')) as json_file:
                leyes = ujson.load(json_file)
            with open(os.path.join(subject_folder, 'appleRightEye.json')) as json_file:
                reyes = ujson.load(json_file)

            with open(os.path.join(subject_folder, 'screen.json')) as json_file:
                dots = ujson.load(json_file)

            with open(os.path.join(subject_folder, 'screen.json')) as json_file:
                screens = ujson.load(json_file)

            for i in range(n_frames):
                if faces['IsValid'][i] != 1 or leyes['IsValid'][i] != 1 or reyes['IsValid'][i] != 1:
                    continue

                sample = {
                    'image': os.path.join(subject_folder, 'frames', "{0:0>5}.jpg".format(i)),
                    'face': (int(round(faces['X'][i])), int(round(faces['Y'][i])), int(round(faces['W'][i])),
                             int(round(faces['H'][i]))),
                    'leye': (int(round(leyes['X'][i])) + int(round(faces['X'][i])),
                             int(round(leyes['Y'][i])) + int(round(faces['Y'][i])), int(round(leyes['W'][i])),
                             int(round(leyes['H'][i]))),
                    'reye': (int(round(reyes['X'][i])) + int(round(faces['X'][i])),
                             int(round(reyes['Y'][i])) + int(round(faces['Y'][i])), int(round(reyes['W'][i])),
                             int(round(reyes['H'][i]))),
                    'target': (dots['XCam'][i], dots['YCam'][i]),
                    'orientation': screens['Orientation'][i]
                }
                dataset.append(sample)

        return dataset

    def __getitem__(self, index):
        item = self.data[index]
        img_path = item['image']
        img = self.img_loader(img_path)

        face = item['face']
        target = item['target']
        orientation = item['orientation']

        if orientation == 1 or orientation == 2:
            fov = (2.610, 1.9626)
        else:
            fov = (1.9626, 2.610)

        if self.use_eye:
            # TODO Deprecated
            leye = item['leye']
            reye = item['reye']

            if self.transform is not None:
                crops, boxes, target = self.transform((img, [face, leye, reye], img.size, target))
                return ((crops, boxes), target)

            return ((img, face, leye, reye), target)
        else:
            if self.transform is not None:
                crop, loc, size, fov, target = self.transform((img, [face], img.size, fov, target))
                return ((*crop, (*loc, *size, fov)), target)

            return ((img, face), target)

    def __len__(self):
        return len(self.data)
