from torch import nn
from gaze.models import facegaze
from gaze.models import eyegaze
import pretrainedmodels


def model_names():
    return pretrainedmodels.model_names


def generate_model(opt, inference=False):
    assert opt.model in ['facegaze', 'eyegaze']
    if opt.pretrain:
        model = pretrainedmodels.__dict__[opt.conv](num_classes=1000)
    else:
        model = pretrainedmodels.__dict__[opt.conv](num_classes=1000, pretrained=None)

    if not hasattr(model, 'mean'):
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        mean, std = model.mean, model.std

    if opt.model == 'facegaze':
        model = facegaze.FaceGaze(model)
    elif opt.model == 'eyegaze':
        model = eyegaze.EyeGaze(model)
    model = model.cuda()
    parameters = model.parameters()
    # parameters = {**dict(parameters), **dict(models.parameters())}
    if not inference:
        model = nn.DataParallel(model, device_ids=None)

    return model, parameters, mean, std