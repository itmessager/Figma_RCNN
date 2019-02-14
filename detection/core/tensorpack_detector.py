import cv2
import numpy as np
import tensorpack.utils.viz as tpviz
from detection.tensorpacks.viz import draw_final_outputs
from detection.core.detector import AbstractDetector
from tensorpack import *
from detection.tensorpacks.coco import COCODetection
from detection.config.tensorpack_config import finalize_configs, config as cfg
from detection.tensorpacks.eval import detect_one_image
from detection.tensorpacks.attrs_predict import ResNetC4Model
import argparse
class TensorPackDetector(AbstractDetector):
    def __init__(self, weight_file):
        MODEL = ResNetC4Model()

        finalize_configs(is_training=False)

        cfg.TEST.RESULT_SCORE_THRESH = cfg.TEST.RESULT_SCORE_THRESH_VIS

        # predict model
        self.pred = OfflinePredictor(PredictConfig(
            model=MODEL,
            session_init=get_model_loader(weight_file),
            input_names=['image'],
            output_names=['person_boxes', 'person_scores', 'person_labels', 'person_masks',
                          'male_predict', 'longhair_predict', 'sunglass_predict',
                          'hat_predict', 'tshirt_predict', 'longsleeve_predict',
                          'formal_predict', 'shorts_predict', 'jeans_predict',
                          'skirt_predict', 'facemask_predict', 'logo_predict',
                          'stripe_predict', 'longpants_predict', 'person_boxes'
                          ]))

        # Only to load the class names into caches
        COCODetection(cfg.DATA.BASEDIR, cfg.DATA.VAL)

    def detect(self, img, rgb=True):
        # Convert to bgr if necessary
        if rgb:
            img = self.rgb_to_bgr(img)

        return detect_one_image(img, self.pred)

    def get_class_ids(self):
        return set(range(1, cfg.DATA.NUM_CATEGORY + 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='',
        type=str,
        help='Configurations of object detection model',
        nargs='+'
    )
    args = parser.parse_args()
    if args.config:
        cfg.update_args(args.config)

    obj_detector = TensorPackDetector('/home/Figma_RCNN/detection/tensorpacks/train_log/maskrcnn/checkpoint')
    img = cv2.imread('/root/Figma_RCNN/detection/test/celebrities.jpg', cv2.IMREAD_COLOR)
    results = obj_detector.detect(img, rgb=False)
    final = draw_final_outputs(img, results)  # image contain boxes,labels and scores
    viz = np.concatenate((img, final), axis=1)
    tpviz.interactive_imshow(viz)