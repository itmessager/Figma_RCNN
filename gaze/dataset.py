from gaze.datasets.gazecapture import GazeCapture


def get_training_set(opt, transform=None):
    assert opt.dataset in ['gazecapture']

    if opt.dataset == 'gazecapture':
        data = GazeCapture(opt.root_path, 'train', transform=transform, use_eye=False)

    return data


def get_validation_set(opt, transform=None):
    assert opt.dataset in ['gazecapture']

    if opt.dataset == 'gazecapture':
        data = GazeCapture(opt.root_path, 'val', transform=transform, use_eye=False)

    return data


def get_test_set(opt, transform=None):
    assert opt.dataset in ['gazecapture']

    if opt.dataset == 'gazecapture':
        data = GazeCapture(opt.root_path, 'test.py', transform=transform, use_eye=False)

    return data


def get_mean_and_std(dataset):
    assert dataset in ['imagenet']
    if dataset == 'imagenet':
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# def get_mean(norm_value=255, dataset='gazecapture'):
#     assert dataset in ['gazecapture']
#
#     if dataset == 'gazecapture':
#         return [
#             100.97 / norm_value, 112.33 / norm_value, 148.37 / norm_value
#         ]
#
#
