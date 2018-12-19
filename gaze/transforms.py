import random

import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms import Normalize
from gaze.dataset import get_mean_and_std

Compose = transforms.Compose


def get_transform(mean, std, face_size=224, eye_size=None, mode='training'):
    assert mode in ['training', 'validation', 'inference']

    tfs = []
    # if mode == 'training':
    #     tfs.append(RandomJitterBox())
    if eye_size is None:
        tfs.append(ResizedCrop([face_size], to_square=True))
    else:
        tfs.append(ResizedCrop([face_size, eye_size, eye_size], to_square=True))
    tfs.append(NormalizeBox())

    if mode == 'training':
        tfs.append(RandomHorizontalFlip())

    tfs.append(ToTensor(inference=(mode == 'inference')))
    tfs.append(ImageTransformWrapper(Normalize(mean, std)))

    return Compose(tfs)


class ImageTransformWrapper(object):
    """Wrapper for Image Transform so that input images are processed by the underlying transform while other inputs are untouched

    Args:
        transform: Underlying image transform

    """

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, data):
        crops, *rest = data
        crops = [self.transform(c) for c in crops]
        return (crops, *rest)


class ToTensor(object):
    """Convert every elements of the data into tensors

    """

    def __init__(self, inference=False):
        self.inference = inference

    def __call__(self, data):
        if self.inference:
            crops, boxes, fov = data
        else:
            crops, boxes, fov, target = data

        crop_tensors = [F.to_tensor(c) for c in crops]
        loc_tensors = [torch.FloatTensor((b[0], b[1])) for b in boxes]
        size_tensors = [torch.FloatTensor((b[2], b[3])) for b in boxes]
        fov_tensors = torch.FloatTensor(fov)
        if self.inference:
            return (crop_tensors, loc_tensors, size_tensors, fov_tensors)
        else:
            return (crop_tensors, loc_tensors, size_tensors, fov_tensors, torch.FloatTensor(target))


class RandomHorizontalFlip(object):
    """Horizontally flip the given gaze input randomly with a probability of 0.5."""

    def __call__(self, data):
        #crops, boxes, img_size, target = data
        crops, boxes, fov, target = data
        if random.random() < 0.5:
            for i in range(len(crops)):
                crops[i] = F.hflip(crops[i])
                (xc, yc, w, h) = boxes[i]
                boxes[i] = (
                -xc, yc, w, h)  # Assuming the x-coordinate is relative to image center and corresponds to crop center
            target = (-target[0], target[1])

        return (crops, boxes, fov, target)


class RandomJitterBox(object):
    """Randomly shift the bounding box by at most the given number of pixels in both directions

    Args:
        shift: Integer, max number of pixels to be shift in both directions
    """

    def __init__(self, shift=0.05, scale=(1.0, 1.1)):
        self.shift = shift
        self.scale = scale

    def __call__(self, data):
        img, boxes, (w_img, h_img), *rest = data

        for i, (x, y, w, h) in enumerate(boxes):
            for attempt in range(10):
                scale = random.uniform(*self.scale)
                w_scaled = int(round(w * scale))
                h_scaled = int(round(h * scale))

                x_shift = int(round(random.uniform(-self.shift, self.shift) * w_scaled + x))
                y_shift = int(round(random.uniform(-self.shift, self.shift) * h_scaled + y))

                if 0 <= x_shift <= w_img - w and 0 <= y_shift <= h_img - h:
                    boxes[i] = (x_shift, y_shift, w_scaled, h_scaled)
                    break

        return (img, boxes, (w_img, h_img), *rest)


class ResizedCrop(object):
    """Crop the input image by the given bounding box and resize the cropped image

    Args:
        size: Integer, size of the output square image
        resize_box: If true, enlarge the shorter edge of bounding box to match the longer edge before cropping so that the cropped image won't be warped
    """

    def __init__(self, size, to_square=False):
        assert isinstance(size, list)
        # self.resize = transforms.Resize((size, size)
        self.resize = [transforms.Resize((s, s)) for s in size]
        self.to_square = to_square

    def __call__(self, data):
        img, boxes, *rest = data
        crops = []
        for i, (x, y, w, h) in enumerate(boxes):
            if self.to_square:
                if w > h:
                    y = y - (w - h) // 2
                    h = w
                elif h > w:
                    x = x - (h - w) // 2
                    w = h
                boxes[i] = (x, y, w, h)
            crop = img.crop((x, y, x + w, y + h))
            crops.append(self.resize[i](crop))

        return (crops, boxes, *rest)


class NormalizeBox(object):
    """Normalize bounding box coordinates to be relative to image center and the given scaled size of the original image
    """

    def __call__(self, data):
        crops, boxes, (w_img, h_img), *rest = data

        # Assuming every image is taken by camera with the same FOV
        # longer_edge_size = max(w_img, h_img)
        #
        # for i, (x, y, w, h) in enumerate(boxes):
        #     x = (x - w_img / 2 + w / 2) / longer_edge_size
        #     y = (y - h_img / 2 + h / 2) / longer_edge_size
        #     w = w / longer_edge_size
        #     h = h / longer_edge_size
        #     boxes[i] = (x, y, w, h)

        for i, (x, y, w, h) in enumerate(boxes):
            # Use face width as screen size v.s actual object size reference
            ref_size = w
            xc = (x - w_img / 2 + w / 2) / ref_size
            yc = (y - h_img / 2 + h / 2) / ref_size
            w = w_img / ref_size
            h = h_img / ref_size
            boxes[i] = (xc, yc, w, h)

        #return (crops, boxes, (w_img, h_img), *rest)
        return (crops, boxes, *rest)

# class CropUpperHalf(object):
#     def __call__(self, data):
#         img, (x, y, w, h), ori_img, *rest = data
#
#         img_w, img_h = img.size
#         img = img.crop((0, 0, img_w, img_h // 2))
#         return (img, (x, y, w, h // 2), *rest)
