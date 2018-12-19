from torch.utils.data import Dataset
import os
from attributer.attributes import WiderAttributes as WA, Attribute, AttributeType as AT
from torchvision.datasets.folder import pil_loader
from attributer.transforms import ToMaskedTargetTensor, get_inference_transform_person, square_no_elastic
import json
import numpy as np

from detection.tensorpacks.coco import COCODetection


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
                # and a < 128:
                for person in im['targets']:
                    sample = {}
                    sample['img'] = os.path.join(root, 'Image', im['file_name'])
                    sample['bbox'] = person['bbox']
                    recognizability = {}
                    for i, attr in enumerate(self._attrs):
                        attr = attr.key  # Use the enum value as key instead
                        print(attr)
                        sample[attr] = int(person['attribute'][i])
                        if person['attribute'][i] != 1:
                            # -1 => 0  1=> 1  0=>-1
                            sample[attr] = np.abs(person['attribute'][i]) - 1
                        # if person['attribute'][i] != 0:  #
                        #     # -1 => 0  1=> 1
                        #     sample[attr] = int((person['attribute'][i] + 1) / 2)
                        #     recognizability[attr] = 1
                        # else:  # Attribute is unrecognizable
                        #     # Treat attribute is available only if recognizability is considered
                        #     if self.output_recognizable:
                        #         sample[attr] = -10  # Dummy value
                        #         recognizability[attr] = 0
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

    @staticmethod
    def list_attributes(output_recognizable=False):
        return [Attribute(attr, AT.BINARY, maybe_unrecognizable=output_recognizable) for attr in WA]


def load_many(basedir, names):
    cropping_transform = get_inference_transform_person
    train_data = WiderAttr(basedir, names, cropping_transform, img_transform=None,
                           target_transform=None, output_recognizable=False)
    train_data_list = train_data.data
    img_attr_dict = {}
    for roi_attr in train_data_list:
        if roi_attr['img'] in img_attr_dict:
            for key in img_attr_dict[roi_attr['img']].keys():
                if type(img_attr_dict[roi_attr['img']][key]) == type([]):
                    pass
                else:
                    img_attr_dict[roi_attr['img']][key] = [img_attr_dict[roi_attr['img']][key]]
                img_attr_dict[roi_attr['img']][key].append(roi_attr[key])
        else:
            img_attr_dict[roi_attr['img']] = roi_attr

    img_attr_list = []
    for img_attr in img_attr_dict.values():
        for key in img_attr.keys():
            if key == 'img':
                img_attr[key] = set(img_attr[key]).pop()
            elif key == 'bbox':
                temp_list = [img_attr[key][0:4]]
                temp_list.extend(img_attr[key][4:])  # hava problem
                temp_list = [np.array(t) for t in temp_list]
                img_attr[key] = np.array(temp_list)
            else:
                img_attr[key] = np.array(img_attr[key])
        img_attr_list.append(img_attr)

    return img_attr_list




if __name__ == '__main__':
    roidbs = load_many('/root/datasets/wider attribute', 'train')
    roidbs2 = COCODetection.load_many(
        '/root/datasets/COCO/DIR', 'train2014', add_gt=True)
    print("OK")