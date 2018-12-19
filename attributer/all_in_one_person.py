import torch.nn as nn
from attributer.model_util import get_backbone_network, get_param_groups
from attributer.attributes import Attribute
import torch


class AllInOnePerson(nn.Module):
    def __init__(self, conv, attributes, pretrained=True, img_size=224, attention=None, norm=False):
        assert isinstance(img_size, int) or (isinstance(img_size, tuple) and len(img_size) == 2)
        assert isinstance(attributes, list)
        for attr in attributes:
            assert isinstance(attr, Attribute)

        super(AllInOnePerson, self).__init__()

        self.attention = attention
        self.norm = norm
        self.conv = conv
        self.feature_extractor, feature_map_depth, self.mean, self.std = get_backbone_network(conv, pretrained)
        fc_in_features = feature_map_depth

        # TODO Test if using separate spatial attention pooling instead of global pooling for each branch works better
        map_size = int(img_size / 224 * 7)
        self.global_pool = nn.AvgPool2d((map_size, map_size), stride=1)
        # self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Define classifier for each attribute
        self.attributes = attributes
        for attr in self.attributes:
            name = attr.name

            setattr(self, 'fc_' + name + '_classifier', nn.Linear(512, 2))
            # Also define a branch for classifying recognizability if necessary
            if attr.maybe_unrecognizable:
                setattr(self, 'fc_' + name + '_recognizable', nn.Linear(512, 2))

            # Also define attention layer for attribute if necessary
            if self.attention == 'base1':
                setattr(self, 'attention_' + name + '_xb', nn.Conv2d(fc_in_features, 1, (1, 1)))
                setattr(self, 'attention_' + name + '_xa', nn.Conv2d(fc_in_features, 512, (1, 1)))
                if self.norm:
                    setattr(self, 'softmax_attention', nn.Softmax(2))
            elif self.attention == 'base2':
                setattr(self, 'attention_' + name + '_conv1', nn.Conv2d(fc_in_features, 512, (1, 1)))
                setattr(self, 'attention_' + name + '_conv2', nn.Conv2d(512, 512, (1, 1)))
                setattr(self, 'attention_' + name + '_conv3', nn.Conv2d(512, 1, (1, 1)))
                #setattr(self, 'fc_' + name + '_attention', nn.Conv2d(512*7*7, 1, (1, 1)))
                setattr(self, 'GP', nn.AvgPool2d((14, 14), stride=1))
                setattr(self, 'fc_' + name + '_1', nn.Linear(fc_in_features, 512))
                if self.norm:
                    setattr(self, 'softmax_attention', nn.Softmax(2))
            else:
                setattr(self, 'fc_' + name + '_1', nn.Linear(fc_in_features, 512))
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Need to override forward method as JIT/trace requires call to _slow_forward, which is called implicitly by
        # __call__ only. Simply calling forward() or features() will result in missing scope name in traced graph.
        # Override here so that in multi-gpu training case when model is replicated, the forward function of feature
        # extractor still gets overriden
        self.feature_extractor.forward = self.feature_extractor.features

        x = self.feature_extractor(x)

        # Global pooling
        if not self.attention:
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)
        #else:
            #print(x.shape)
            #x = x.permute(2, 3, 1, 0)
            #print(x.shape)
        # Compute output for each attribute
        result = []
        for attr in self.attributes:
            name = attr.name
            if self.attention == 'base1':
                xb = getattr(self, 'attention_' + name + '_xb')(x)
                #print(xb.shape)
                xa = getattr(self, 'attention_' + name + '_xa')(x)
                #print(xa.shape)
                if self.norm:
                    feature_w, feature_h = xb.size(2), xb.size(3)
                    xb = getattr(self, 'softmax_attention')(xb.view(xb.size(0), xb.size(1), -1)).view(
                        xb.size(0), xb.size(1), feature_w, feature_h)
                y = torch.mul(xb, xa)
                #print(y.shape)
                #y = y.view(y.size(0), y.size(1), -1)
                y = torch.sum(y, (2, 3))
                #w_m_h = y.size(2) * y.size(3)
                #y = self.global_pool(y)
                #* w_m_h
                y = y.view(y.size(0), -1)
            elif self.attention == 'base2':
                conv1 = getattr(self, 'attention_' + name + '_conv1')(x)
                conv1_relu = self.relu(conv1)
                conv2 = getattr(self, 'attention_' + name + '_conv2')(conv1_relu)
                conv2_relu = self.relu(conv2)
                conv3 = getattr(self, 'attention_' + name + '_conv3')(conv2_relu)
                conv3_relu = self.relu(conv3)
                if self.norm:
                    feature_w, feature_h = conv3_relu.size(2), conv3_relu.size(3)
                    conv3_relu = getattr(self, 'softmax_attention')(conv3_relu.view(
                        conv3_relu.size(0), conv3_relu.size(1), -1)).view(
                        conv3_relu.size(0), conv3_relu.size(1), feature_w, feature_h)
                y = torch.mul(conv3_relu, x)
                y = getattr(self, 'GP')(y)
                y = y.view(y.size(0), -1)
                y = getattr(self, 'fc_' + name + '_1')(y)
            else:
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
            new_layers = [module for module_name, module in self.named_modules() if module_name.startswith('fc') or
                          module_name.startswith('attention_')]
            param_groups = [first_conv, new_layers]

        return get_param_groups(param_groups)
