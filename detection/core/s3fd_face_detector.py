import os

import torch.nn.functional as F

from detection.core.s3fd import S3fd
from detection.utils.bbox import *
from detection.core.detector import PytorchDetector, DetectionResult
from detection.config.s3fd_config import config as cfg


class S3fdFaceDetector(PytorchDetector):
    def __init__(self, weight_file='models/s3fd_convert.pth'):
        super().__init__(S3fd(), weight_file)

    def preprocess(self, img, rgb):
        if rgb:
            img = self.rgb_to_bgr(img)
        img = img - np.array([104, 117, 123])
        img = img.transpose(2, 0, 1)
        img = img.reshape((1,) + img.shape)
        img = torch.from_numpy(img).float().cuda()
        return img

    def postprocess(self, olist, width, height):
        n_outputs = int(len(olist) / 2)
        bboxlist = []
        for i in range(cfg.START_LAYER, n_outputs):
            olist[i * 2] = F.softmax(olist[i * 2], dim=1)  # Softmax classification score

        for i in range(cfg.START_LAYER, n_outputs):
            ocls, oreg = olist[i * 2].data, olist[i * 2 + 1].data
            FB, FC, FH, FW = ocls.size()  # feature map size

            stride = 2 ** (i + 2)  # 4,8,16,32,64,128

            # Ignore detection results with low confidence
            scores = ocls[0, 1].view(-1)
            mask = scores >= 0.05
            ind_mask = torch.masked_select(torch.arange(FH * FW).long().cuda(), mask)
            if len(ind_mask.size()) == 0 or ind_mask.size()[0] == 0:
                continue

            xy = torch.index_select(torch.from_numpy(
                np.ascontiguousarray(np.indices((FH, FW), dtype=np.float32).reshape(2, -1).T[:, ::-1])).cuda(), 0,
                                    ind_mask)
            xy = xy * stride + stride / 2.0

            loc = torch.index_select(oreg[0, :].view(4, -1).transpose(0, 1), 0, ind_mask)
            scores = torch.masked_select(scores, mask)

            priors = torch.cat([xy, torch.ones_like(xy) * stride * 4], dim=1)
            variances = [0.1, 0.2]
            boxes = decode(loc, priors, variances)
            bboxlist.append(torch.cat([boxes, scores.view(-1, 1)], dim=1))

        bboxlist = torch.cat(bboxlist, dim=0).cpu().numpy()
        if 0 == len(bboxlist): bboxlist = np.zeros((1, 5))

        # nms
        keep = nms(bboxlist, 0.3)
        bboxlist = bboxlist[keep, :]

        # thresholding
        bboxlist = bboxlist[bboxlist[:, 4] > cfg.MIN_SCORE]

        # Further clip boxes
        clipped_boxes = clip_boxes(bboxlist[:, :4], (height, width))
        scores = bboxlist[:, 4]

        # Convert bboxlist to DetectionResult
        detection_results = [DetectionResult(box=tuple(bbox), score=score, class_id=1, mask=None) for bbox, score in zip(clipped_boxes, scores)]

        return detection_results

    def get_class_ids(self):
        return {1}  # Just support face class
