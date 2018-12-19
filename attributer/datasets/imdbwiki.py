from scipy.io import loadmat
import torch.utils.data as data
from datetime import datetime
import os
import numpy as np
from attributer.attributes import FaceAttributes as FA, AttributeType as AT, Attribute
from torchvision.datasets.folder import pil_loader # use to read img form the path


def calc_age(taken, dob):
    birth = datetime.fromordinal(max(int(dob) - 366, 1))

    # assume the photo was taken in the middle of the year
    if birth.month < 7:
        return taken - birth.year
    else:
        return taken - birth.year - 1


def get_meta(mat_path, db):
    meta = loadmat(mat_path)
    full_path = meta[db][0, 0]["full_path"][0]
    dob = meta[db][0, 0]["dob"][0]  # Matlab serial date number
    gender = meta[db][0, 0]["gender"][0]
    photo_taken = meta[db][0, 0]["photo_taken"][0]  # year
    face_score = meta[db][0, 0]["face_score"][0]
    second_face_score = meta[db][0, 0]["second_face_score"][0]
    age = [calc_age(photo_taken[i], dob[i]) for i in range(len(dob))]

    return full_path, dob, gender, photo_taken, face_score, second_face_score, age


class IMDBWIKI(data.Dataset):
    _attrs = [Attribute(FA.AGE, AT.NUMERICAL), Attribute(FA.GENDER, AT.BINARY)]

    def __init__(self, root, min_score=1.0, transform=None, target_transform=None):
        self.data = self._make_dataset(root, min_score)
        self.transform = transform
        self.target_transform = target_transform
        self.img_loader = pil_loader

    @staticmethod
    def _make_dataset(root, min_score):
        dataset = []
        dbs = ['imdb', 'wiki']
        for db in dbs:
            root_path = os.path.join(root, "{}_crop/".format(db))
            mat_path = root_path + "{}.mat".format(db)
            full_path, dob, gender, photo_taken, face_score, second_face_score, age = get_meta(mat_path, db)

            for i in range(len(face_score)):
                if face_score[i] < min_score:
                    continue

                if (~np.isnan(second_face_score[i])) and second_face_score[i] > 0.0:
                    continue

                if ~(0 <= age[i] <= 100):
                    continue

                if np.isnan(gender[i]):
                    continue

                sample = {'img': os.path.join(root_path, str(full_path[i][0])), FA.AGE: age[i], FA.GENDER: int(gender[i])}
                dataset.append(sample)

        return dataset

    def __getitem__(self, index):
        sample = self.data[index]
        img_path = sample['img']
        img = self.img_loader(img_path)

        sample.pop('img')

        if self.transform is not None:
            img = self.transform(img)

        target = sample
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (img, target)

    def __len__(self):
        return len(self.data)

    @classmethod
    def list_attributes(cls):
        return cls._attrs

