import numpy as np
import torch
from pytorch2keras.converter import pytorch_to_keras
from deploy.keras_to_tensorflow import keras_to_tensorflow, parse_args

import argparse
from attributer.model import generate_model
import os


def convert_pytorch_to_keras(model, img_size, output_file):
    if isinstance(img_size, int):
        img_size = (img_size, img_size)

    n_channels = 3
    input_np = np.random.uniform(0, 1, (1, n_channels, img_size[0], img_size[1]))
    input_var = torch.FloatTensor(input_np)

    k_model = pytorch_to_keras(model, input_var, [(n_channels, img_size[0], img_size[1])], verbose=True)
    k_model.save(output_file + ".h5")


def convert_pytorch_to_tensorflow(model, img_size, output_len, output_file):
    convert_pytorch_to_keras(model, img_size, output_file)

    from argparse import Namespace
    args = parse_args()
    args.input_model_file = output_file + ".h5"
    args.output_fld = os.path.dirname(output_file)
    args.num_outputs = output_len
    # args = Namespace(input_model_file=output_file+".h5", num_output=output_len, )
    keras_to_tensorflow(args)


parser = argparse.ArgumentParser()
parser.add_argument(
    '--attribute_model',
    default='all_in_one',
    type=str,
    help='all_in_one')
parser.add_argument(
    '--attribute_conv',
    default='resnet18',
    type=str)
parser.add_argument(
    '--attribute_checkpoint',
    default='',
    type=str,
    help='Save data (.pth) of previous training')

opt = parser.parse_args()

attr_opt = argparse.Namespace()
attr_opt.model = opt.attribute_model
attr_opt.conv = opt.attribute_conv
# attr_opt.checkpoint = opt.attribute_checkpoint
attr_opt.checkpoint = "../models/save_36.pth"
attr_opt.pretrain = False

model, parameters, mean, std = generate_model(attr_opt)
# model = AllInOne(resnet18()).cuda()
# model = nn.DataParallel(model, device_ids=None)
checkpoint = torch.load(attr_opt.checkpoint)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
model = model.module.cpu()

convert_pytorch_to_tensorflow(model, (224, 224), 10, "../models/all_resnet18")
