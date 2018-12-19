import torch.utils.data as data
import os
from attributer.attributes import FaceAttributes as FA, AttributeType as AT, Attribute
from torchvision.datasets.folder import pil_loader


class CelebA(data.Dataset):
    attributes = ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips",
                  "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby",
                  "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male",
                  "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose",
                  "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair",
                  "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"]

    attr_map = {15: FA.EYEGLASSES, 20: FA.GENDER, 28: FA.RECEDING_HAIRLINES, 31: FA.SMILING}
    _attrs = [Attribute(FA.EYEGLASSES, AT.BINARY), Attribute(FA.GENDER, AT.BINARY),
              Attribute(FA.RECEDING_HAIRLINES, AT.BINARY), Attribute(FA.SMILING, AT.BINARY)]

    def __init__(self, root, subset, transform=None, target_transform=None):
        self.data = self._make_dataset(root, subset)
        self.transform = transform
        self.target_transform = target_transform
        self.img_loader = pil_loader

    @staticmethod
    def _make_dataset(root, subset):
        assert subset in ['train', 'val', 'test.py']
        split = {'train': 0, 'val': 1, 'test.py': 2}[subset]

        # Only keep images belong to the given subset split
        imgs_to_keep = []
        split_file = os.path.join(root, 'Eval', 'list_eval_partition.txt')
        with open(split_file, 'r') as f:
            for line in f.readlines():
                img, split_id = line.split()
                if int(split_id.strip()) != split:
                    continue
                imgs_to_keep.append(img)
        imgs_to_keep = set(imgs_to_keep)

        data = []
        anno_file = os.path.join(root, 'Anno', 'list_attr_celeba.txt')
        with open(anno_file, 'r') as f:
            for line in f.readlines()[2:]:
                tokens = line.split()
                if tokens[0] not in imgs_to_keep:
                    continue

                sample = {"img": os.path.join(root, "img_align_celeba", tokens[0])}
                # add in other attr to sample
                for idx, attr in CelebA.attr_map.items():  # Include only the specified attributes in the sample data
                    val = int(tokens[idx + 1].strip())  # 1 or -1
                    sample[attr] = 1 if val == 1 else 0
                data.append(sample)

        return data

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
