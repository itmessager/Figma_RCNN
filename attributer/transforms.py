import torch
from torchvision import transforms
from torchvision.transforms import functional as F

Compose = transforms.Compose

from attributer.attributes import FaceAttributes as A, AttributeType as AT, Attribute


# return a func?????
def get_inference_transform(mean, std, face_size=224):
    def inference_transform(input):
        img, bbox = input
        w, h = img.size

        xc = (bbox[0] + bbox[2]) / 2
        yc = (bbox[1] + bbox[3]) / 2
        wbox = (bbox[2] - bbox[0]) / 2
        hbox = (bbox[3] - bbox[1]) / 2

        # Crop a square patch with added margin
        box_size = min(w - xc, h - yc, xc, yc, wbox * 1.4, hbox * 1.4)
        crop = img.crop((xc - box_size, yc - box_size, xc + box_size, yc + box_size))

        # Convert to normalized Pytorch Tensor
        return F.normalize(F.to_tensor(F.resize(crop, (face_size, face_size))), mean, std)

    return inference_transform


def inference_transform(input):
    img, bbox = input
    w, h = img.size

    xc = (bbox[0] + bbox[2]) / 2
    yc = (bbox[1] + bbox[3]) / 2
    wbox = (bbox[2] - bbox[0]) / 2
    hbox = (bbox[3] - bbox[1]) / 2

    # Crop a square patch with added margin
    box_size = min(w - xc, h - yc, xc, yc, wbox * 1.4, hbox * 1.4)
    crop = img.crop((xc - box_size, yc - box_size, xc + box_size, yc + box_size))

    return crop


def get_inference_transform_person(input):
    img, bbox = input

    # bbox = [xmin, ymin, w, h]
    return img.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))


def square_no_elastic(img):
    w, h = img.size
    size = max(w, h)
    return F.center_crop(img, size)


class NormalizeAge(object):
    def __init__(self, mean=50.0, std=50.0):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        if A.AGE in sample:
            age = sample[A.AGE]
            sample[A.AGE] = (age - self.mean) / self.std
        return sample


# Convert target of a sample to Tensor(s). Support both multi-dataset setting as well as recognizability.
# Assuming each sample is a dict whose keys include Attributes as well as (optional) recognizability of each attribute
class ToMaskedTargetTensor(object):
    _tensor_types = {AT.NUMERICAL: torch.float, AT.BINARY: torch.long, AT.MULTICLASS: torch.long}

    def __init__(self, attributes):
        for attr in attributes:
            assert isinstance(attr, Attribute)

        self.attrs = attributes

    def __call__(self, sample):
        mask = []  # Mask indicating whether each attribute is available in this sample
        target = []
        dummy_val = -10  # Placeholder value for unavailable attribute, so that each sample has the same length
        rec_text = 'recognizability'
        for i, attr in enumerate(self.attrs):
            if attr.key in sample:
                # Handle recognizability only for potentially unrecognizable attributes
                if attr.maybe_unrecognizable:
                    # Some dataset may not have recognizability issue with this attribute or with any attributes at all
                    # So only read recognizability from valid samples
                    if rec_text in sample and attr.key in sample[rec_text]:
                        recognizability = sample[rec_text][attr.key]
                    else:
                        recognizability = 1  # Treat such dataset or sample as trivially recognizable

                    # Class label is valid(available) only when the attribute of this sample is recognizable
                    if recognizability == 1:
                        cls_available = 1
                        val = sample[attr.key]
                    else:
                        cls_available = 0
                        val = dummy_val
                    rec_available = 1  # Recognizability is available as long as the sample contains the attribute
                else:
                    # Since the attribute is always recognizable, its label must be available
                    cls_available = 1
                    val = sample[attr.key]
            else:
                cls_available = 0
                rec_available = 0
                val = dummy_val
                if attr.maybe_unrecognizable:
                    recognizability = dummy_val

            # Use a mask tensor to indicate which attribute is available on each sample
            mask.append(torch.tensor([cls_available], dtype=torch.uint8, requires_grad=False))
            if attr.maybe_unrecognizable:
                mask.append(torch.tensor([rec_available], dtype=torch.uint8, requires_grad=False))

            target_tensor = torch.tensor([val], dtype=self._tensor_types[attr.data_type], requires_grad=False)
            target.append(target_tensor)
            # When one attribute is potentially unrecognizable, we always return a tuple,
            # no matter this sample contains such info or not
            if attr.maybe_unrecognizable:
                # Add recognizability as another component of the target
                target.append(torch.tensor([recognizability], dtype=torch.long, requires_grad=False))

        return (target, mask)


class ToSimpleTargetTensor:
    _tensor_types = {AT.NUMERICAL: torch.float, AT.BINARY: torch.long, AT.MULTICLASS: torch.long}

    def __init__(self, datatype):
        assert isinstance(datatype, AT)
        self.datatype=datatype

    def __call__(self, target):
        return torch.tensor(target, dtype=self._tensor_types[self.datatype])