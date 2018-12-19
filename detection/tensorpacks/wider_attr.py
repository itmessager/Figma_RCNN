import os
import json
import numpy as np


def make_dataset(root, subset):
    assert subset in ['train', 'val', 'test.py']

    data = []
    attrs = ['male', 'longhair', 'sunglass', 'hat', 'tshirt', 'longsleeve', 'formal', 'shorts', 'jeans', 'longpants',
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
                for i, attr in enumerate(attrs):
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

                data.append(sample)
                a += 1
    return data


def load_many(basedir, names):
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

        img_attr_list.append(img_attr)

    return img_attr_list

if __name__ == '__main__':
    roidbs = load_many('/root/datasets/wider attribute', 'train')
    print("OK")
