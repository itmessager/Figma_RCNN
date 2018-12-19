from torch.utils.data import Dataset
import os
from attributer.attributes import ErisedAttributes as EA, Attribute, AttributeType as AT
from torchvision.datasets.folder import pil_loader
import json


class ErisedAll(Dataset):
    # if indicator = 'person', get the person image. if indicator = 'face', get the face image
    def __init__(self, root, subset, indicator='person', img_transform=None, target_transform=None,
                 output_recognizable=False, specified_attrs=[]):
        self.output_recognizable = output_recognizable
        self._attrs = ErisedAll.list_attributes(output_recognizable, specified_attrs)
        self.miss_indicate_list = self.list_end(specified_attrs)
        self.indicator = indicator
        self.name_list = self.generate_list_name(root, subset)

        self.data = self._make_dataset(root)
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.img_loader = pil_loader

    def _make_dataset(self, root):
        data = []

        # because the train label and val label are in same file, distinct them according to file_name head
        # train or val set
        anno_file = []
        if self.indicator == 'person':
            anno_file = os.path.join(root, 'json')
        elif self.indicator == 'face':
            anno_file = os.path.join(root, 'face_json')

        #a = 0
        anno_list = os.listdir(anno_file)
        for i in range(len(anno_list)):
            #if a < 128:
                path = os.path.join(anno_file, anno_list[i])
                if anno_list[i].startswith('MOV'):
                    img_file_name = 'MOV_' + anno_list[i].split('_')[1]
                else:
                    img_file_name = anno_list[i].split('_')[0]
                if img_file_name in self.name_list and os.path.isfile(path):
                    with open(path, 'r') as f:
                        datasample = json.load(f)
                        sample = {}
                        recognizability = {}
                        sample['img'] = os.path.join(root, img_file_name, 'crops', datasample['filename'])
                        if self.indicator == 'face':
                            sample['bbox'] = datasample['bbox']
                        #for i, attr in enumerate(EA):
                        for i, attr in enumerate(self._attrs):
                            attr = attr.key
                            # for the attribute just have a value
                            if str(attr) in datasample:
                                if datasample[str(attr)] != self.miss_indicate_list[i]:
                                    # for the binary classification, the map is 1 => 1, 2 => 0
                                    if str(attr) in ["glasses", "pregnant", "tottoo", "carry", "sex"]:
                                        sample[attr] = 2 - datasample[str(attr)]
                                        if self.output_recognizable:
                                            recognizability[attr] = 1
                                    # for multi-classification, the map is value-1
                                    else:
                                        sample[attr] = datasample[str(attr)] - 1
                                        if self.output_recognizable:
                                            recognizability[attr] = 1
                                elif self.output_recognizable:
                                    # sample[attr] = -1
                                    sample[attr] = -10
                                    recognizability[attr] = 0
                            # for the attributes could have more than one value
                            # datasample['hairstyle'] is a list including one or more value of [1, 2, 3, 4]
                            # 1 =>  hairstyleundercut, 2 = > hairstylegreasy, 3 => normal, 4 => miss
                            elif str(attr) == 'undercut':
                                if 1 in datasample['hairstyle']:
                                    sample[attr] = 1
                                    if self.output_recognizable:
                                        recognizability[attr] = 1
                                elif 2 in datasample['hairstyle'] or 3 in datasample['hairstyle']:
                                    sample[attr] = 0
                                    if self.output_recognizable:
                                        recognizability[attr] = 1
                                elif self.output_recognizable:
                                    # sample[attr] = -1
                                    sample[attr] = -10
                                    recognizability[attr] = 0
                            elif attr.name.lower() == 'greasy':
                                if 2 in datasample['hairstyle']:
                                    sample[attr] = 1
                                    if self.output_recognizable:
                                        recognizability[attr] = 1
                                elif 1 in datasample['hairstyle'] or 3 in datasample['hairstyle']:
                                    sample[attr] = 0
                                    if self.output_recognizable:
                                        recognizability[attr] = 1
                                elif self.output_recognizable:
                                    # sample[attr] = -1
                                    sample[attr] = -10
                                    recognizability[attr] = 0
                        if self.output_recognizable:
                            sample['recognizability'] = recognizability
                        data.append(sample)
                        #a += 1
            #else:
                #break

        return data

    def __getitem__(self, index):
        sample = self.data[index]
        img_path = sample['img']
        img = self.img_loader(img_path)
        if self.indicator == 'face':
            bbox = sample['bbox']
            img = (img, bbox)

        # crop picture to get person according to the bbox to
        if self.img_transform is not None:
            img = self.img_transform(img)

        # Transform target
        target = sample.copy()  # Copy sample so that the original one won't be modified
        target.pop('img')
        if self.indicator == 'face':
            target.pop('bbox')

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (img, target)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def list_attributes(output_recognizable=False, specified_attrs=[]):
        attrs = []
        if not specified_attrs:
            specified_attrs = EA.names()
        # if specified_attrs:
        #     for attr in EA:
        #         if str(attr) in specified_attrs:
        #
        #     return [Attribute(attr, AT.MULTICLASS, maybe_unrecognizable=output_recognizable) for attr in EA if str(attr) in specified_attrs]
        # else:
        #     return [Attribute(attr, AT.MULTICLASS, maybe_unrecognizable=output_recognizable) for attr in EA]
        for attr in EA:
            attr_name = str(attr)
            if attr_name in specified_attrs:
                # get type of the attr
                if attr_name in ["glasses", "pregnant", "tottoo", "carry", "sex"]:
                    attr_type = AT.BINARY
                else:
                    attr_type = AT.MULTICLASS
                attrs.append(Attribute(attr, attr_type, maybe_unrecognizable=output_recognizable))
        return attrs

    def list_end(self, specified_attrs):
        miss_indicate_list = [4, 3, 4, 4, 3, 11, 4, 4, 5, 3, 3, 3]
        list = []
        if not specified_attrs:
            specified_attrs = EA.names()
        for i, item in enumerate(EA):
            if str(item) in specified_attrs:
                list.append(miss_indicate_list[i])
        return list

    def generate_list_name(self, root, subset):
        val_list_name = ['YDXJ0021', 'YDXJ0043', 'MOV_5359', 'MOV_5360', 'MOV_5361', 'MOV_5365']
        if subset == 'val':
            return val_list_name
        else:
            train_list_name = []
            list = os.listdir(root)
            for name in list:
                if not name.endswith('json') and os.path.isdir(os.path.join(root, name)):
                    if name not in val_list_name:
                        train_list_name.append(name)
            return train_list_name
