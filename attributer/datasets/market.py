import os

from torch.utils.data import Dataset
from torchvision.datasets.folder import pil_loader
from attributer.datasets import market1501, market_util
from training.market_utils import create_cmc_probe_and_gallery
import numpy as np


class Market1501(object):

    def __init__(self, dataset_dir, num_validation_y=0.1, seed=1234):
        self._dataset_dir = dataset_dir
        self._num_validation_y = num_validation_y
        self._seed = seed

    def read_train(self):
        filenames, ids, camera_indices = market1501.read_train_split_to_str(
            self._dataset_dir)
        train_indices, _ = market_util.create_validation_split(
            np.asarray(ids, np.int64), self._num_validation_y, self._seed)

        filenames = [filenames[i] for i in train_indices]
        ids = [ids[i] for i in train_indices]
        camera_indices = [camera_indices[i] for i in train_indices]
        return filenames, ids, camera_indices

    def read_validation(self):
        filenames, ids, camera_indices = market1501.read_train_split_to_str(
            self._dataset_dir)
        _, valid_indices = market_util.create_validation_split(
            np.asarray(ids, np.int64), self._num_validation_y, self._seed)

        filenames = [filenames[i] for i in valid_indices]
        ids = [ids[i] for i in valid_indices]
        camera_indices = [camera_indices[i] for i in valid_indices]
        return filenames, ids, camera_indices

    def read_test(self):
        return market1501.read_test_split_to_str(self._dataset_dir)


class Market(Dataset):

    def __init__(self, root, subset, transform=None, target_transform=None):
        self.data_x, self.data_y = self._make_dataset(root, subset)
        self.transform = transform
        self.target_transform = target_transform
        self.img_loader = pil_loader

    @staticmethod
    def _make_dataset(root, subset):
        assert subset in ['train', 'val', 'test.py']
        dataset = Market1501(os.path.join(root, 'Market-1501'), num_validation_y=0.1, seed=1234)
        data_x, data_y = [], []

        if subset == 'train':
            data_x, data_y, _ = dataset.read_train()

        if subset == 'val':
            total_data_x, total_data_y, camera_indices = dataset.read_validation()

            probe_indices, gallery_indices = create_cmc_probe_and_gallery(total_data_y, camera_indices)
            probe_x, probe_y = np.asarray(total_data_x)[probe_indices], np.asarray(total_data_y)[probe_indices]
            gallery_x, gallery_y = np.asarray(total_data_x)[gallery_indices], np.asarray(total_data_y)[gallery_indices]

            data_x = np.concatenate([probe_x, gallery_x], axis=0)
            data_y = np.concatenate([probe_y, gallery_y], axis=0)

        if subset == 'test.py':
            data_x, data_y, _ = dataset.read_test()

        return data_x, data_y

    def __getitem__(self, index):
        img_path = self.data_x[index]
        img = self.img_loader(img_path)

        # crop picture to get person according to the bbox to
        if self.transform is not None:
            img = self.transform(img)

        target = self.data_y[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (img, target)

    def __len__(self):
        return len(self.data_x)
