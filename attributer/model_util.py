import pretrainedmodels
from torchvision.models import vgg


def modify_vgg(model):
    model._features = model.features
    del model.features
    del model.classifier  # Delete unused module to free memory

    def features(self, input):
        x = self._features(input)
        return x

    # TODO Based on pretrainedmodels, it modify instance method instead of class. Will need to test.py
    setattr(model.__class__, 'features', features)  # Must use setattr here instead of assignment

    return model


def get_backbone_network(conv, pretrained=True): # conv, pretrained
    if conv.startswith('vgg'):  # For VGG, use torchvision directly instead
        vgg_getter = getattr(vgg, conv)  # return a function of VGG,VGG is a class inherits touch.nn.Module
        backbone = vgg_getter(pretrained=pretrained) # return a object of VGG
        feature_map_depth = 512

        # Modify the VGG model to make it align with pretrainmodels' format
        backbone = modify_vgg(backbone)
    else:
        if pretrained:
            backbone = pretrainedmodels.__dict__[conv](num_classes=1000)
        else:
            backbone = pretrainedmodels.__dict__[conv](num_classes=1000, pretrained=None)
        feature_map_depth = backbone.last_linear.in_features

    # use mean and std to do what
    if not hasattr(backbone, 'mean'):
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        mean, std = backbone.mean, backbone.std

    return backbone, feature_map_depth, mean, std


def _generate_param_group(modules):
    for module in modules:
        for name, param in module.named_parameters():
            yield param


def get_param_groups(group_of_layers):
    assert isinstance(group_of_layers, list)
    param_groups = []
    for group in group_of_layers:
        param_groups.append({'params': _generate_param_group(group)})
    return param_groups

