from torch.utils.data import Dataset
import os

from torchvision.datasets.folder import pil_loader
import json


class WiderAttr(Dataset):
    def __init__(self, root, subset, cropping_transform, img_transform=None, target_transform=None,
                 output_recognizable=False):
        self.output_recognizable = output_recognizable
        self._attrs = WiderAttr.list_attributes(output_recognizable)
        self.data = self._make_dataset(root, subset)

        self.cropping_transform = cropping_transform
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.img_loader = pil_loader

    def _make_dataset(self, root, subset):
        assert subset in ['train', 'val', 'test.py']

        data = []

        # Assuming dataset directory is layout as
        # Wider Attribute
        #   -- Anno
        #     -- wider_attribute_trainval.json
        #     -- wider_attribute_test.json
        #   -- Image
        #     -- train
        #       -- 0--Parade
        #         -- ...
        #     -- val
        #     -- test.py
        if subset in ['train', 'val']:
            anno_file = os.path.join(root, 'Anno', 'wider_attribute_trainval.json')
        else:
            anno_file = os.path.join(root, 'Anno', 'wider_attribute_test.json')

        with open(anno_file, 'r') as f:
            dataset = json.load(f)
        a = 0
        for im in dataset['images']:
            if im['file_name'].startswith(subset):
                #and a < 128:
                for person in im['targets']:
                    sample = {}
                    sample['img'] = os.path.join(root, 'Image', im['file_name'])
                    sample['bbox'] = person['bbox']
                    recognizability = {}
                    for i, attr in enumerate(self._attrs):
                        attr = attr.key  # Use the enum value as key instead
                        if person['attribute'][i] != 0:  #
                            # -1 => 0  1=> 1
                            sample[attr] = int((person['attribute'][i] + 1) / 2)
                            recognizability[attr] = 1
                        else:  # Attribute is unrecognizable
                            # Treat attribute is available only if recognizability is considered
                            if self.output_recognizable:
                                sample[attr] = -10  # Dummy value
                                recognizability[attr] = 0
                    if self.output_recognizable:
                        # In this dataset every attribute may be unrecognizable
                        sample['recognizability'] = recognizability
                    data.append(sample)
                    a += 1
        return data

    def __getitem__(self, index):
        sample = self.data[index]
        img_path = sample['img']
        bbox = sample['bbox']
        img = self.img_loader(img_path)

        crop = self.cropping_transform((img, bbox))

        # Transform image crop
        if self.img_transform is not None:
            crop = self.img_transform(crop)

        # Transform target
        target = sample.copy()  # Copy sample so that the original one won't be modified
        target.pop('img')
        target.pop('bbox')
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (crop, target)

    def __len__(self):
        return len(self.data)