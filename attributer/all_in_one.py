import torch
import torch.nn as nn
from attributer.model_util import get_backbone_network, get_param_groups


class AllInOne(nn.Module):
    # why n_headpose_bins=66 , not 67 ????
    def __init__(self, conv, pretrained=True, img_size=224, n_headpose_bins=66, headpose_bin_interval=3,
                 headpose_bin_mean=99): # opt.conv, opt.pretrain
        assert isinstance(img_size, int) or (isinstance(img_size, tuple) and len(img_size) == 2)

        super(AllInOne, self).__init__()

        self.feature_extractor, feature_map_depth, self.mean, self.std = get_backbone_network(conv, pretrained)
        # object of Module, int, float list, float list
        # TODO Test if using separate spatial attention pooling instead of global pooling for each branch works better
        self.avgpool = nn.AvgPool2d((7, 7), stride=1)

        # Age branch
        # fc_in_features = feature_map_depth * 7 * 7
        fc_in_features = feature_map_depth
        #
        self.fc_a1 = nn.Linear(fc_in_features, 512)
        # self.fc_a2 = nn.Linear(512, 128)
        self.fc_a3 = nn.Linear(512, 1)

        # Gender branch
        self.fc_g1 = nn.Linear(fc_in_features, 512)
        # self.fc_g2 = nn.Linear(512, 128)
        self.fc_g3 = nn.Linear(512, 2)

        # Eyeglasses branch
        self.fc_e1 = nn.Linear(fc_in_features, 512)
        self.fc_e3 = nn.Linear(512, 2)

        # Receding_hairline branch
        self.fc_r1 = nn.Linear(fc_in_features, 512)
        self.fc_r3 = nn.Linear(512, 2)

        # Similing branch
        self.fc_s1 = nn.Linear(fc_in_features, 512)
        self.fc_s3 = nn.Linear(512, 2)

        # Head Pose branch
        self.fc_yaw = nn.Linear(fc_in_features, n_headpose_bins)
        self.fc_pitch = nn.Linear(fc_in_features, n_headpose_bins)
        self.fc_roll = nn.Linear(fc_in_features, n_headpose_bins)
        self.headpose_bin_idx = [idx for idx in range(n_headpose_bins)]
        self.headpose_bin_interval = headpose_bin_interval
        self.headpose_bin_mean = headpose_bin_mean

        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Need to override forward method as JIT/trace requires call to _slow_forward, which is called implicitly by
        # __call__ only. Simply calling forward() or features() will result in missing scope name in traced graph.
        # Override here so that in multi-gpu training case when model is replicated, the forward function of feature
        # extractor still gets overriden
        self.feature_extractor.forward = self.feature_extractor.features   # only reserve cnn

        x = self.feature_extractor(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        age = self.fc_a1(x)
        age = self.relu(age)
        age = self.fc_a3(age)

        gender = self.fc_g1(x)
        gender = self.relu(gender)
        gender = self.fc_g3(gender)

        eyeglasses = self.fc_e1(x)
        eyeglasses = self.relu(eyeglasses)
        eyeglasses = self.fc_e3(eyeglasses)

        receding_hairline = self.fc_r1(x)
        receding_hairline = self.relu(receding_hairline)
        receding_hairline = self.fc_r3(receding_hairline)

        smiling = self.fc_s1(x)
        smiling = self.relu(smiling)
        smiling = self.fc_s3(smiling)

        # Head Pose
        # TODO Move separate binning/un-binning logics to dedicated class or functions
        yaw_bin = self.fc_yaw(x)
        pitch_bin = self.fc_pitch(x)
        roll_bin = self.fc_roll(x)
        yaw = self._headpose_bin_to_value(yaw_bin, x.device)
        pitch = self._headpose_bin_to_value(pitch_bin, x.device)
        roll = self._headpose_bin_to_value(roll_bin, x.device)

        return (age, gender, eyeglasses, receding_hairline, smiling, yaw_bin, pitch_bin, roll_bin, yaw, pitch, roll)

    def _headpose_bin_to_value(self, bin, device):
        #compute the real age form dealed digit    why sum(, 1) ??????
        # Convert to tensor and place these constants onto the same device as the model and other parameters
        return torch.sum(self.softmax(bin) * torch.FloatTensor(self.headpose_bin_idx).to(device=device),
                         1) * torch.FloatTensor([self.headpose_bin_interval]).to(device=device) - \
               torch.FloatTensor([self.headpose_bin_mean]).to(device=device)

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
