import torch.nn as nn
from attributer.attributes import Attribute
from attributer.model_util import get_backbone_network, get_param_groups


class AllInOneErisedPerson(nn.Module):
    # why n_headpose_bins=66 , not 67 ????
    def __init__(self, conv, attributes, pretrained=True, img_size=224):
        assert isinstance(img_size, int) or (isinstance(img_size, tuple) and len(img_size) == 2)
        assert isinstance(attributes, list)
        for attr in attributes:
            assert isinstance(attr, Attribute)

        super(AllInOneErisedPerson, self).__init__()

        self.feature_extractor, feature_map_depth, self.mean, self.std = get_backbone_network(conv, pretrained)
        # TODO Test if using separate spatial attention pooling instead of global pooling for each branch works better
        self.avgpool = nn.AvgPool2d((7, 7), stride=1)
        # self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        fc_in_features = feature_map_depth

        # Define classifier for each attribute
        self.attributes = attributes
        self.numclass = [3, 2, 2, 2, 2, 10, 3, 3, 4, 2, 2, 2]

        # define respective layer for  classification, name format: fc_Male_1
        for attr, num in zip(self.attributes, self.numclass):
            name = attr.name
            setattr(self, 'fc_' + name + '_1', nn.Linear(fc_in_features, 512))
            setattr(self, 'fc_' + name + '_classifier', nn.Linear(512, num))
            if attr.maybe_unrecognizable:
                # for detectability
                setattr(self, 'fc_' + name + '_recognizable', nn.Linear(512, 2))

        self.relu = nn.ReLU(inplace=True)
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Need to override forward method as JIT/trace requires call to _slow_forward, which is called implicitly by
        # __call__ only. Simply calling forward() or features() will result in missing scope name in traced graph.
        # Override here so that in multi-gpu training case when model is replicated, the forward function of feature
        # extractor still gets overriden
        self.feature_extractor.forward = self.feature_extractor.features

        x = self.feature_extractor(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        result = []
        for attr in self.attributes:
            name = attr.name
            y = getattr(self, 'fc_' + name + '_1')(x)
            y = self.relu(y)
            cls = getattr(self, 'fc_' + name + '_classifier')(y)
            result.append(cls)
            if attr.maybe_unrecognizable:  # Also return the recognizable branch if necessary
                recognizable = getattr(self, 'fc_' + name + '_recognizable')(y)
                result.append(recognizable)

        return tuple(result)


    def get_parameter_groups(self):
        """
        Return model parameters in three groups: 1) First Conv layer of the CNN feature extractor; 2) Other Conv layers
        of the CNN feature extractor; 3) All other added layers of this model
        """
        if self.conv.startswith('resnet'):
            first_conv = [self.feature_extractor.conv1, self.feature_extractor.bn1]  # Only support Resnet for now
            other_convs = [self.feature_extractor.layer1, self.feature_extractor.layer2, self.feature_extractor.layer3,
                           self.feature_extractor.layer4]
            new_layers = [module for module_name, module in self.named_modules() if module_name.startswith('fc')]
            param_groups = [first_conv, other_convs, new_layers]
        else:
            first_conv = [self.feature_extractor._features]  # Only support VGG for now
            new_layers = [module for module_name, module in self.named_modules() if module_name.startswith('fc')]
            param_groups = [first_conv, new_layers]

        return get_param_groups(param_groups)

