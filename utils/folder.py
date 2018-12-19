import cv2

from utils.viz_utils import cv_to_pil_image


def opencv_loader(path):
    img_bgr = cv2.imread(path)

    return cv_to_pil_image(img_bgr)