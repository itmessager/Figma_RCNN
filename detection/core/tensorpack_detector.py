from detection.core.detector import AbstractDetector
from tensorpack import *

from detection.tensorpacks.coco import COCODetection
from detection.config.tensorpack_config import finalize_configs, config as cfg
from detection.tensorpacks.eval import (
    detect_one_image)
from detection.tensorpacks.train import ResNetC4Model


class TensorPackDetector(AbstractDetector):
    def __init__(self, weight_file):
        MODEL = ResNetC4Model()

        finalize_configs(is_training=False)

        cfg.TEST.RESULT_SCORE_THRESH = cfg.TEST.RESULT_SCORE_THRESH_VIS

        # predict model
        self.pred = OfflinePredictor(PredictConfig(
            model=MODEL,
            session_init=get_model_loader(weight_file),
            input_names=MODEL.get_inference_tensor_names()[0],
            output_names=MODEL.get_inference_tensor_names()[1]))

        # Only to load the class names into caches
        COCODetection(cfg.DATA.BASEDIR, cfg.DATA.VAL)

    def detect(self, img, rgb=True):
        # Convert to bgr if necessary
        if rgb:
            img = self.rgb_to_bgr(img)

        return detect_one_image(img, self.pred)

    def get_class_ids(self):
        return set(range(1, cfg.DATA.NUM_CATEGORY + 1))
