import logging
import torch

import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
from ignite._utils import convert_tensor
from attributer.model import generate_model
from attributer.transforms import get_inference_transform
from attributer.attributes import FaceAttributes as A


# Simply a rewrite for AllInOneAttributer for now
class AllInOneFaceModel:
    # TODO Consolidate separate Attribute related definitions
    thresh = {A.GENDER: 0.5, A.RECEDING_HAIRLINES: 0.5, A.EYEGLASSES: 0.5, A.SMILING: 0.5}
    attr_ret = [A.AGE, A.GENDER, A.RECEDING_HAIRLINES, A.EYEGLASSES, A.SMILING, A.HEAD_YAW, A.HEAD_PITCH, A.HEAD_ROLL]
    age_idx = 0
    age_mean = 50
    age_std = 50

    @classmethod
    def unnormalize_age(cls, ages):
        return ages * cls.age_std + cls.age_mean

    def __init__(self, opt):
        model, parameters, mean, std = generate_model(opt)

        logging.info('loading checkpoint {}'.format(opt.checkpoint))
        checkpoint = torch.load(opt.checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        self.model = model
        self.transform = get_inference_transform(mean, std)

    def __call__(self, img, faces, rgb=True):
        if len(faces) == 0:
            return []

        tensor = self._preprocess(img, faces, rgb)
        with torch.no_grad():
            preds = self.model(tensor)
        results = self._postprocess(preds, len(faces))
        return results

    def _preprocess(self, img, faces):
        crops = [self.transform((img, face)) for face in faces]

        # Batch crops to Tensor
        with torch.no_grad():
            batch = default_collate(crops)
            tensor = convert_tensor(batch, device='cuda')
        return tensor

    def _postprocess(self, preds, n_samples):
        converted_preds = []  # [n_attributes, n_face_samples]
        for i, attr in enumerate(A):
            p = preds[i]
            if attr == A.AGE:
                p = self.unnormalize_age(p)  # Post-process age
            elif attr in self.thresh:
                p = F.softmax(p, dim=1)
                p = (p[:, 1] > self.thresh[attr])
            elif attr not in [A.HEAD_ROLL, A.HEAD_PITCH, A.HEAD_YAW]:
                continue
            converted_preds.append(p.data.cpu().numpy())

        # Transpose results so that output will have size [n_face_samples, n_attributes]
        ret = []
        for i in range(n_samples):
            ret.append([])
            # Only return desired output results. E.g ignoring HEAD_YAW_BIN
            for j, _ in enumerate(self.attr_ret):
                ret[i].append(converted_preds[j][i])
        return ret
