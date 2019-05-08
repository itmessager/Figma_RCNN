"""
-1  unspecified
 0  negative
 1  positive
 bbox: xmin, ymin, width, height
"""

import os
import json
import numpy as np


def make_dataset(root, subset):
    assert subset in ['train', 'val', 'test']

    data = []
    attr_names = ['male', 'longhair', 'sunglass', 'hat', 'tshirt', 'longsleeve', 'formal', 'shorts', 'jeans',
                  'longpants',
                  'skirt', 'facemask', 'logo', 'stripe']
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
                sample['img'] = os.path.join(root, im['file_name'])
                #   print(sample['img'])
                sample['bbox'] = person['bbox']
                for i, attr in enumerate(attr_names):
                    sample[attr] = int(person['attribute'][i])
                    if person['attribute'][i] != 1:
                        # -1 => 0  1=> 1  0=>-1
                        sample[attr] = np.abs(person['attribute'][i]) - 1
                data.append(sample)
                a += 1
    return data


def load_many(basedir, names, is_augment=False):
    attr_names = ['male', 'longhair', 'sunglass', 'hat', 'tshirt', 'longsleeve', 'formal', 'shorts', 'jeans',
                  'longpants','skirt', 'facemask', 'logo', 'stripe']
    train_data_list = make_dataset(basedir, names)
    # a list contain  16 attributes of each roi
    img_attr_dict = {}
    # get a dictionary that contain 16 attributes of each roi of each image, the key is the image name
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

    # convert dict to list
    img_attr_list = []
    id = 10000
    for img_attr in img_attr_dict.values():
        for key in img_attr.keys():
            if key == 'img':
                img_attr[key] = set(img_attr[key]).pop() if type(img_attr[key]) == type([]) else img_attr[key]
            elif key == 'bbox':
                temp_list = [img_attr[key][0:4]]
                temp_list.extend(img_attr[key][4:])  # hava problem
                temp_list = [np.array(t) for t in temp_list]
                img_attr[key] = np.array(temp_list).astype(np.float32)
            else:
                img_attr[key] = np.array(img_attr[key])
                if np.array(img_attr[key]).shape == ():  # convert shape () to (1,)
                    img_attr[key].resize((1,))

        if is_augment:
            img_attr['bbox'] = box_augment(img_attr['bbox']).astype(np.float32)
            for attr_name in attr_names:
                img_attr[attr_name] = attr_augment(img_attr[attr_name])

        img_attr['id'] =id
        id += 1
        img_attr_list.append(img_attr)
    return img_attr_list


def attr_augment(attribute):
    attribute_aug = attribute
    for attr in attribute:
        attr_aug = np.tile(attr, 5)
        attribute_aug = np.concatenate((attribute_aug, attr_aug), axis=0)
    return attribute_aug


def box_augment(bboxes):
    bboxes_aug = bboxes
    for box in bboxes:
        #temp = [np.random.normal(box_i, 0.04*(abs(box_i)+box_i),size=5) for box_i in box]
        temp = [np.random.normal(box_i, 0.01*abs(box[2]+box[3]), size=5) for box_i in box]
        b_aug = np.array(list(zip(temp[0], temp[1], temp[2], temp[3])))
        bboxes_aug = np.concatenate((bboxes_aug, b_aug), axis=0)
    return bboxes_aug


if __name__ == '__main__':
    roidbs = load_many('/root/datasets/WiderAttribute', 'train', False)
    #bbb = box_augment(roidb['bbox'])
    print("OK")
