import os
from collections import OrderedDict
from functools import partial

from torchvision.transforms import Compose, CenterCrop, Resize, ToTensor, \
    RandomHorizontalFlip, Normalize, RandomCrop, RandomRotation
from torch.utils.data import DataLoader

from attributer.attributes import AttributeType as AT
from attributer.dataset_util import split_dataset_into_train_val, MultiDataset
from attributer.datasets.celeba import CelebA
from attributer.datasets.headpose import Pose_300W_LP
from attributer.datasets.imdbwiki import IMDBWIKI
from attributer.datasets.widerattr import WiderAttr
from attributer.datasets.erisedall import ErisedAll
from attributer.datasets.market import Market
from attributer.transforms import ToMaskedTargetTensor, NormalizeAge, get_inference_transform_person, square_no_elastic, \
    ToSimpleTargetTensor, inference_transform


def _get_imdb(opt, mean, std, attrs):
    root = os.path.join(opt.root_path, 'IMDB')
    train_transform = Compose(
        [Resize((opt.face_size, opt.face_size)), RandomHorizontalFlip(), ToTensor(), Normalize(mean, std)])
    # [Resize((256, 256)), RandomCrop(224), RandomHorizontalFlip(), ToTensor(), Normalize(mean, std)])
    val_transform = Compose([Resize((opt.face_size, opt.face_size)), ToTensor(), Normalize(mean, std)])
    target_transform = Compose([NormalizeAge(), ToMaskedTargetTensor(attrs)])

    train_data = IMDBWIKI(root, transform=train_transform, target_transform=target_transform)
    val_data = IMDBWIKI(root, transform=val_transform, target_transform=target_transform)

    return split_dataset_into_train_val(train_data, val_data, val_ratio=0.1)


def _get_celeba(opt, mean, std, attrs):
    root = os.path.join(opt.root_path, 'CelebA')
    train_transform = Compose(
        [CenterCrop(178), Resize((opt.face_size, opt.face_size)), RandomHorizontalFlip(), ToTensor(),
         Normalize(mean, std)])
    val_transform = Compose(
        [CenterCrop(178), Resize((opt.face_size, opt.face_size)), ToTensor(), Normalize(mean, std)])
    target_transform = ToMaskedTargetTensor(attrs)

    train_data = CelebA(root, subset='train', transform=train_transform, target_transform=target_transform)
    val_data = CelebA(root, subset='val', transform=val_transform, target_transform=target_transform)

    return train_data, val_data


def _get_headpose_dataset(dataset, opt, mean, std, attrs):
    train_transform = Compose([Resize(240), RandomCrop(224), ToTensor(), Normalize(mean, std)])
    val_transform = Compose([Resize((opt.face_size, opt.face_size)), ToTensor(), Normalize(mean, std)])
    target_transform = ToMaskedTargetTensor(attrs)

    if dataset == '300W_LP':
        root = os.path.join(opt.root_path, '300W_LP')
        train_data = Pose_300W_LP(root, transform=train_transform, target_transform=target_transform, training=True)
        val_data = Pose_300W_LP(root, transform=val_transform, target_transform=target_transform, training=False)
    else:
        raise Exception('Error: not a valid dataset name')

    return split_dataset_into_train_val(train_data, val_data, val_ratio=0.05)


def _get_widerattr(opt, mean, std, attrs):
    root = os.path.join(opt.root_path, 'Wider Attribute')
    cropping_transform = get_inference_transform_person
    train_img_transform = Compose(
        [square_no_elastic, RandomHorizontalFlip(), RandomRotation(10, expand=True),
         # [RandomHorizontalFlip(), RandomRotation(10, expand=True),
         Resize((opt.person_size, opt.person_size)), ToTensor(), Normalize(mean, std)])
    # [CenterCrop(178), Resize((256, 256)), RandomCrop(224), RandomHorizontalFlip(), ToTensor(), Normalize(mean, std)])
    val_img_transform = Compose(
        [square_no_elastic, Resize((opt.person_size, opt.person_size)), ToTensor(), Normalize(mean, std)])
    target_transform = ToMaskedTargetTensor(attrs)

    train_data = WiderAttr(root, 'train', cropping_transform, img_transform=train_img_transform,
                           target_transform=target_transform, output_recognizable=opt.output_recognizable)
    val_data = WiderAttr(root, 'val', cropping_transform,
                         img_transform=val_img_transform, target_transform=target_transform,
                         output_recognizable=opt.output_recognizable)

    return train_data, val_data


def _get_erisedattr(opt, mean, std, attrs):
    root = os.path.join(opt.root_path, 'Erised')
    train_transform = Compose(
        [square_no_elastic,
         RandomHorizontalFlip(), RandomRotation(10, expand=True),
         Resize((opt.person_size, opt.person_size)), ToTensor(), Normalize(mean, std)])
    # [CenterCrop(178), Resize((256, 256)), RandomCrop(224), RandomHorizontalFlip(), ToTensor(), Normalize(mean, std)])
    val_transform = Compose(
        [square_no_elastic,
         Resize((opt.person_size, opt.person_size)), ToTensor(), Normalize(mean, std)])
    target_transform = ToMaskedTargetTensor(attrs)

    train_data = ErisedAll(root, 'train', img_transform=train_transform, target_transform=target_transform,
                           output_recognizable=opt.output_recognizable, specified_attrs=opt.specified_attrs)
    val_data = ErisedAll(root, 'val', img_transform=val_transform, target_transform=target_transform,
                         output_recognizable=opt.output_recognizable, specified_attrs=opt.specified_attrs)

    return train_data, val_data


# it just a idea, hasn't been implemented
def _get_erised_faceattr(opt, mean, std, attrs):
    root = os.path.join(opt.root_path, 'Erised')
    train_transform = Compose(
        [inference_transform,
         RandomHorizontalFlip(), RandomRotation(10, expand=True),
         Resize((opt.person_size, opt.person_size)), ToTensor(), Normalize(mean, std)])
    # [CenterCrop(178), Resize((256, 256)), RandomCrop(224), RandomHorizontalFlip(), ToTensor(), Normalize(mean, std)])
    val_transform = Compose(
        [inference_transform,
         Resize((opt.face_size, opt.face_size)), ToTensor(), Normalize(mean, std)])
    target_transform = ToMaskedTargetTensor(attrs)

    train_data = ErisedAll(root, 'train', indicator='face', img_transform=train_transform, target_transform=target_transform,
                           output_recognizable=opt.output_recognizable, specified_attrs=opt.specified_attrs)
    val_data = ErisedAll(root, 'val', indicator='face', img_transform=val_transform, target_transform=target_transform,
                         output_recognizable=opt.output_recognizable, specified_attrs=opt.specified_attrs)

    return train_data, val_data


def _get_market(opt, mean, std):
    root = opt.root_path
    train_transform = Compose([RandomHorizontalFlip(), Resize((128, 64)), ToTensor(), Normalize(mean, std)])
    val_transform = Compose([Resize((128, 64)), ToTensor(), Normalize(mean, std)])
    target_transform = ToSimpleTargetTensor(AT.MULTICLASS)

    train_data = Market(root, subset='train', transform=train_transform, target_transform=target_transform)
    val_data = Market(root, subset='val', transform=val_transform, target_transform=target_transform)

    return train_data, val_data


_dataset_getters_market = {'Market': _get_market}

_dataset_getters_face = {'IMDB': _get_imdb, 'CelebA': _get_celeba,
                         '300W_LP': partial(_get_headpose_dataset, '300W_LP'),
                         'Erised': _get_erised_faceattr}

_dataset_getters_person = {'Wider': _get_widerattr, 'Erised': _get_erisedattr}

_dataset_attrs_availability = {'IMDB': IMDBWIKI, '300W_LP': Pose_300W_LP,
                               'CelebA': CelebA, 'Wider': WiderAttr, 'Erised': ErisedAll}


def get_tasks(opt):
    datasets = opt.dataset.split(",")

    # Get all available attributes from these datasets
    available_attrs = OrderedDict()
    for ds in datasets:
        # TODO Refactor to remove this if-else statement
        if ds == 'IMDB':
            attrs_ds = IMDBWIKI.list_attributes()
        elif ds == '300W_LP':
            attrs_ds = Pose_300W_LP.list_attributes()
        elif ds == 'CelebA':
            attrs_ds = CelebA.list_attributes()
        elif ds == 'Wider':
            attrs_ds = WiderAttr.list_attributes(opt.output_recognizable)
        elif ds == 'Erised':
            attrs_ds = ErisedAll.list_attributes(opt.output_recognizable)
        else:
            raise Exception("Not supported dataset {}".format(ds))

        for attr in attrs_ds:
            if attr.key not in available_attrs:
                available_attrs[attr.key] = attr
            else:
                # Merge attributes from different datasets
                available_attrs[attr.key] = available_attrs[attr.key].merge(attr)
    return list(available_attrs.values())


def get_multiset_data(datasets, opt, available_attrs, mean, std):
    names = opt.dataset.split(",")

    # Get and collect each dataset which will be combined later
    train_data = []
    val_data = []
    for name in names:
        assert name in datasets
        train, val = datasets[name](opt, mean, std, available_attrs)
        train_data.append(train)
        val_data.append(val)

    # Combine multiple datasets if necessary
    if len(names) > 1:
        train_data = MultiDataset(train_data)
        val_data = MultiDataset(val_data)
    else:
        train_data = train_data[0]
        val_data = val_data[0]

    train_loader = DataLoader(train_data, batch_size=opt.batch_size, num_workers=opt.n_threads, shuffle=True,
                              pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=opt.batch_size, num_workers=opt.n_threads,
                            pin_memory=True)

    return available_attrs, train_loader, val_loader, None


get_faceattr_data = partial(get_multiset_data, _dataset_getters_face)
get_personattr_data = partial(get_multiset_data, _dataset_getters_person)


def get_data_market(opt, mean, std):
    names = opt.dataset.split(",")

    # Get and collect each dataset which will be combined later
    train_data = []
    val_data = []
    for name in names:
        assert name in _dataset_getters_market
        train, val = _dataset_getters_market[name](opt, mean, std)
        train_data.append(train)
        val_data.append(val)

    # Combine multiple datasets if necessary
    if len(train_data) > 1:
        train_data = MultiDataset(train_data)
        val_data = MultiDataset(val_data)
    else:
        train_data = train_data[0]
        val_data = val_data[0]

    train_loader = DataLoader(train_data, batch_size=opt.batch_size, num_workers=opt.n_threads, shuffle=True,
                              pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=opt.batch_size, num_workers=opt.n_threads, shuffle=False,
                            pin_memory=True)

    return train_loader, val_loader, None
