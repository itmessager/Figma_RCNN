import torch.nn as nn

from attributer.model_util import get_backbone_network


class AllInOneMOT(nn.Module):
    def __init__(self, conv, pretrained=True, img_size=(128, 64), num_classes=1501):
        assert isinstance(img_size, int) or (isinstance(img_size, tuple) and len(img_size) == 2)

        super(AllInOneMOT, self).__init__()

        self.conv = conv
        self.feature_extractor, feature_map_depth, self.mean, self.std = get_backbone_network(conv, pretrained)
        fc_in_features = feature_map_depth

        # TODO Test if using separate spatial attention pooling instead of global pooling for each branch works better
        # self.global_pool = nn.AvgPool2d((7, 7), stride=1)

        # TODO Run a dummy data to decide the actual feature map size, so that it doesn't need to be hard-coded
        self.fc1 = nn.Linear(fc_in_features * 8, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes + 1)

        self.elu = nn.ELU(inplace=True)
        self.relu = nn.ReLU(inplace=True)
        # dim is just like axis in the tensorflow
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Need to override forward method as JIT/trace requires call to _slow_forward, which is called implicitly by
        # __call__ only. Simply calling forward() or features() will result in missing scope name in traced graph.
        # Override here so that in multi-gpu training case when model is replicated, the forward function of feature
        # extractor still gets overriden
        self.feature_extractor.forward = self.feature_extractor.features

        x = self.feature_extractor(x)

        # x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.elu(x)
        features = self.fc2(x)
        logits = self.fc3(features)
        # logits = self.softmax(logits)
        return features, logits