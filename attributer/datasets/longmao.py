import argparse
import codecs

import cv2
import json
import os
import random

from tensorpack.utils.viz import interactive_imshow
from torch.utils.data import Dataset
from torchvision.datasets.folder import pil_loader
from object_detection.utils.visualization_utils import draw_bounding_box_on_image

from attributer.attributes import ErisedAttributes as EA, Gender, AgeGroup, CarryKids, Pregnancy
from utils.folder import opencv_loader
from utils.viz_utils import pil_to_cv_image


class LongmaoMomKids(Dataset):
    # Category 2 means stroller, 16 means foreigner and thus no age group specified.
    category_map = {3: Gender.MALE, 4: Gender.FEMALE, 5: Gender.UNCERTAIN, 6: AgeGroup.UNDER_TWO,
                    7: AgeGroup.THREE_TO_FIVE,
                    8: AgeGroup.SIX_TO_TWELVE, 9: AgeGroup.THIRTEEN_TO_EIGHTEEN, 10: AgeGroup.NINETEEN_TO_TWENTY_THREE,
                    11: AgeGroup.TWENTY_FOUR_TO_TWENTY_NINE, 12: AgeGroup.THIRTY_TO_THIRTY_FIVE,
                    13: AgeGroup.THIRTY_SIX_TO_FORTY_FIVE, 14: AgeGroup.FORTY_SIX_TO_SIXTY, 15: AgeGroup.ABOVE_SIXTY,
                    17: AgeGroup.UNCERTAIN, 18: Pregnancy.PREGNANT, 19: Pregnancy.NON_PREGNANT, 20: Pregnancy.UNCERTAIN,
                    21: CarryKids.CARRYING, 22: CarryKids.NON_CARRYING, 23: CarryKids.UNCERTAIN}

    def __init__(self, root, visualize=False, img_loader=pil_loader, crop_transform=None, img_transform=None,
                 target_transform=None):
        """

        :param root:
        :param visualize: If true, each item will be an image which may includes multiple person instances.
        Otherwise, each item will be a person instance cropped from the original image.
        """
        self.data = self._make_dataset(root, visualize)
        self.image_as_item = visualize
        self.img_loader = img_loader
        self.crop_transform = crop_transform
        self.img_transform = img_transform
        self.target_transform = target_transform

    def _make_dataset(self, root, image_as_item):
        data = []
        n_categories = {id: 0 for id in self.category_map.keys()}
        n_people = 0
        n_no_age_data = 0

        image_folder = os.path.join(root, 'Images')
        anno_folder = os.path.join(root, 'annotations')

        for filename in os.listdir(anno_folder):
            anno_file = os.path.join(anno_folder, filename)
            img_file = os.path.join(image_folder, filename.replace('.json', '.jpg'))

            # with codecs.open(anno_file, 'r', 'utf-8') as f:
            with open(anno_file, 'r', encoding='ISO-8859-1') as f:
                anno = json.load(f)

                people = []
                for item in anno["annotation"]:
                    person = {}

                    cat_ids = [int(cat_id) for cat_id in item["category_id"]]
                    # Ignore non-person item, i.e. stroller
                    if 2 in cat_ids:
                        continue

                    # Update data statistics
                    n_people += 1
                    for cat_id in cat_ids:
                        if cat_id == 16:
                            n_no_age_data += 1
                        elif cat_id != 1:
                            n_categories[cat_id] += 1

                    person["bbox"] = tuple([float(i) for i in item["bbox"]])
                    person["attrs"] = {}
                    for cat_id in cat_ids:
                        if cat_id != 1 and cat_id != 16:
                            cat = self.category_map[cat_id]
                            person["attrs"][cat.get_attr_type()] = cat
                    people.append(person)

                if image_as_item:
                    # Each sample will be an image with multiple people
                    data.append({'img': img_file, 'people': people})
                else:
                    for person in people:
                        person['img'] = img_file  # Include path to the image where person patch to be cropped from
                        data.append(person)

        # Print statistics
        print("# of images: {}".format(len(data)))
        print("# of people instances: {}".format(n_people))
        for k, v in n_categories.items():
            print("# of {}/{}: {}".format(self.category_map[k].get_attr_type(), self.category_map[k], v))
        print("# of foreigner (without age group annotations): {}".format(n_no_age_data))

        return data

    def __getitem__(self, index):
        sample = self.data[index]
        img_path = sample['img']
        img = self.img_loader(img_path)

        if self.image_as_item:
            return img, sample
        else:
            if self.crop_transform is not None:
                img = self.crop_transform(img, sample["bbox"])

            if self.img_transform is not None:
                img = self.img_transform(img)

            target = sample["attrs"]
            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target

    def __len__(self):
        return len(self.data)


class LongmaoAgeGender(Dataset):
    def __init__(self, root, crop_transform=None, img_transform=None,
                 target_transform=None, visualize=False):
        self.data = self._make_dataset(root)
        # Has to use opencv to load the image because some images are manually rotated for correction
        # but PIL cannot recognize that......
        self.img_loader = opencv_loader
        self.crop_transform = crop_transform
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.visualize = visualize

    def _make_dataset(self, root):
        # Keep track of label statistics
        n_labels = {EA.GENDER: {gender: 0 for gender in Gender}, EA.AGE_GROUP: {age_group: 0 for age_group in AgeGroup}}
        n_switches = 0

        image_folder = os.path.join(root, 'Images')
        anno_file = os.path.join(root, 'Annotations', 'bbox_gt.txt')
        data = []
        with open(anno_file, 'r') as f:
            print("Parsing annotation file of Longmao Age-Gender dataset")
            while True:
                sample = {}
                # Sample of annotations:
                # 20-F/938-2-20.jpg  (subpath under image folder)
                # 1 (number of bounding boxes/faces in each image. Will always be 1 in this dataset)
                # 204.1 234.2 869.9 1019.6 (bounding box coordinate: xmin ymin xmax ymax)
                img_path = f.readline().strip()
                if not img_path:  # To end of file
                    break
                n_bboxs = int(f.readline().strip())
                assert n_bboxs == 1  # Each image in this dataset only contains one major person
                xmin, ymin, xmax, ymax = [float(val) for val in f.readline().strip().split()]  # Bounding box coordinate

                sample['img'] = os.path.join(image_folder, img_path)
                sample['bbox'] = (xmin, ymin, xmax - xmin, ymax - ymin)

                # Extract attributes from image name following 'id-gender-age.jpg' format
                id, gender, age = img_path.split('/')[1].split('.')[0].split('-')
                # However, it is also likely that gender and age label is mistakenly switched,
                # so we need to deal with that with folder information,
                # where the gender info in the folder is supposed to be accurate. Maybe...
                gender_folder = img_path.split('/')[0][-1]
                if (gender_folder == 'F' and int(gender) != 2) or (gender_folder == 'M' and int(gender) != 1):
                    # Switch the order of gender and age label
                    tmp = gender
                    gender = age
                    age = tmp
                    n_switches += 1
                # Convert to standard attribute map format
                gender = Gender.MALE if int(gender) == 1 else Gender.FEMALE
                age_group = AgeGroup.age_to_group(int(age))
                sample['attrs'] = {EA.GENDER: gender, EA.AGE: int(age), EA.AGE_GROUP: age_group}

                # Update statistics
                n_labels[EA.GENDER][gender] += 1
                n_labels[EA.AGE_GROUP][age_group] += 1

                data.append(sample)

        # Print statistics
        print("# of images: {}".format(len(data)))
        print("# of people instances: {}".format(len(data)))
        print("# of switched age/gender labels: {}".format(n_switches))
        for attr, submap in n_labels.items():
            print("# of instances for each label of {}:".format(attr))
            for k, v in submap.items():
                print("{}: {}".format(k, v))

        return data

    def __getitem__(self, index):
        sample = self.data[index]
        img_path = sample['img']
        img = self.img_loader(img_path)

        if self.visualize:
            return img, sample
        else:
            if self.crop_transform is not None:
                img = self.crop_transform(img, sample["bbox"])

            if self.img_transform is not None:
                img = self.img_transform(img)

            target = sample["attrs"]
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img, target

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        required=True,
        type=str,
        help="Dataset to visualize: age-gender, mom-kids"
    )
    parser.add_argument(
        '--root',
        required=True,
        type=str,
        help="Root directory to dataset")
    args = parser.parse_args()

    if args.dataset == 'mom-kids':
        ds = LongmaoMomKids(args.root, visualize=True)
        multi_bbox_per_image = True
    elif args.dataset == 'age-gender':
        ds = LongmaoAgeGender(args.root, visualize=True)
        multi_bbox_per_image = False
    else:
        raise Exception("Unsupported dataset: {}".format(args.dataset))

    print('Start to visualize dataset: {}'.format(args.dataset))
    print("Type 'x' to stop.")
    while True:
        ind = random.randint(0, len(ds) - 1)
        img, sample = ds[ind]

        print("Visualizing image {}".format(sample["img"].split("/")[-1]))
        if multi_bbox_per_image:
            people = sample["people"]
        else:
            people = [sample]

        for person in people:
            x, y, w, h = person["bbox"]
            display_str = ["{}/{}".format(k, v) for k, v in person["attrs"].items()]
            draw_bounding_box_on_image(img, y, x, (y + h), (x + w), display_str_list=display_str, thickness=3,
                                       use_normalized_coordinates=False)

        # Convert pil image to opencv format for display control
        image_bgr = pil_to_cv_image(img)
        interactive_imshow(image_bgr)
