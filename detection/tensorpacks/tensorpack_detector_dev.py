import cv2
import numpy as np
import tensorpack.utils.viz as tpviz
from skimage.util import crop

from detection.tensorpacks.viz import draw_final_outputs
from detection.core.detector import AbstractDetector
from tensorpack import *
from detection.tensorpacks.coco import COCODetection
from detection.config.tensorpack_config import finalize_configs, config as cfg
from detection.tensorpacks.eval import fill_full_mask
from detection.tensorpacks.attrs_predict import ResNetC4Model
import argparse

from detection.tensorpacks.common import CustomResize
from detection.utils.bbox import clip_boxes
from detection.core.detector import DetectionResult


def detect_one_image(img, model_func):
    """
    Run detection on one image, using the TF callable.
    This function should handle the preprocessing internally.

    Args:
        img: an image
        model_func: a callable from TF model,
            takes image and returns (boxes, probs, labels, [masks])

    Returns:
        [DetectionResult]
    """
    orig_shape = img.shape[:2]
    resizer = CustomResize(cfg.PREPROC.TEST_SHORT_EDGE_SIZE, cfg.PREPROC.MAX_SIZE)
    resized_img = resizer.augment(img)
    scale = np.sqrt(resized_img.shape[0] * 1.0 / img.shape[0] * resized_img.shape[1] / img.shape[1])
    boxes, probs, labels, masks, *attrs = model_func(resized_img)
    boxes = boxes / scale
    # boxes are already clipped inside the graph, but after the floating point scaling, this may not be true any more.
    boxes = clip_boxes(boxes, orig_shape)
    # if masks:
    # has mask
    full_masks = [fill_full_mask(box, mask, orig_shape)
                  for box, mask in zip(boxes, masks)]
    masks = full_masks
    results = [DetectionResult(*args) for args in zip(boxes, probs, labels, masks,
                                                      attrs[0], attrs[1], attrs[2], attrs[3],
                                                      attrs[4], attrs[5], attrs[6], attrs[7],
                                                      attrs[8], attrs[9], attrs[10], attrs[11],
                                                      attrs[12], attrs[13])]
    return results


def detect_person(img, model_func):
    """
        Run detection on one image, using the TF callable.
        This function should handle the preprocessing internally.

        Args:
            img: an image
            model_func: a callable from TF model,
                takes image and returns (boxes, probs, labels, [masks])

        Returns:
            [DetectionResult]
        """
    orig_shape = img.shape[:2]
    resizer = CustomResize(cfg.PREPROC.TEST_SHORT_EDGE_SIZE, cfg.PREPROC.MAX_SIZE)
    resized_img = resizer.augment(img)
    scale = np.sqrt(resized_img.shape[0] * 1.0 / img.shape[0] * resized_img.shape[1] / img.shape[1])
    boxes, *_ = model_func(resized_img)
    # crop_imgs = [resized_img[box[1]:box[3], box[0]:box[2]] for box in boxes.astype(np.int)]
    results = []
    i=0
    for box in boxes.astype(np.int):

        crop_img = resized_img[box[1]:box[3], box[0]:box[2]]
        resized_img_black = np.zeros_like(resized_img)
        resized_img_black[box[1]:box[3], box[0]:box[2]] = crop_img
        # cv2.imwrite('/root/datasets/img_{}.jpg'.format(i), resized_img_black)
        bbox, prob, label, mask, *attrs = model_func(resized_img_black)
        bbox = bbox / scale
        bbox = clip_boxes(bbox, orig_shape)
        mask = fill_full_mask(bbox[0], mask[0], orig_shape)
        result = DetectionResult(bbox[0], prob[0], label[0], mask,
                                 attrs[0][0], attrs[1][0], attrs[2][0], attrs[3][0],
                                 attrs[4][0], attrs[5][0], attrs[6][0], attrs[7][0],
                                 attrs[8][0], attrs[9][0], attrs[10][0], attrs[11][0],
                                 attrs[12][0], attrs[13][0])
        results.append(result)
        i+=1
    return results


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
                          'stripe_predict', 'longpants_predict'
                          ]))

        # Only to load the class names into caches
        COCODetection(cfg.DATA.BASEDIR, cfg.DATA.VAL)

    # def detect(self, img, rgb=True):
    #     # Convert to bgr if necessary
    #     if rgb:
    #         img = self.rgb_to_bgr(img)
    #
    #     return detect_one_image(img, self.pred)

    def detect(self, img, rgb=True):
        # Convert to bgr if necessary
        if rgb:
            img = self.rgb_to_bgr(img)

        return detect_person(img, self.pred)

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

    obj_detector = TensorPackDetector('/root/datasets/figmarcnn/checkpoint')
    img = cv2.imread('/root/datasets/img-folder/a.png', cv2.IMREAD_COLOR)

    results = obj_detector.detect(img, rgb=False)
    final = draw_final_outputs(img, results)  # image contain boxes,labels and scores
    viz = np.concatenate((img, final), axis=1)
    tpviz.interactive_imshow(viz)
