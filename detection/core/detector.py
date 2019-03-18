"""
Base class for all detector. Defines detector interface.
"""
from abc import abstractmethod
from collections import namedtuple

DetectionResult = namedtuple(
    'DetectionResult',
    ['box', 'score', 'class_id', 'male', 'longhair',
     'sunglass', 'hat', 'tshirt', 'longsleeve', 'formal', 'shorts',
     'jeans', 'skirt', 'facemask', 'logo', 'stripe', 'longpants'])
"""
box: (xmin, ymin, xmax, ymax) in image original space
score: float
class_id: int, 1~NUM_CLASS
mask: None, or a binary image of the original image shape
"""

class AbstractDetector(object):
    @abstractmethod
    def detect(self, img, rgb=True):
        """
        Perform object detection on the given image.
        :param img: Numpy array of size (W, H, 3) containing an image to be detected.
        :param rgb: If true, the color channel of the image is stored as RGB. If false,
        it is assumed to be BGR (OpenCV style)
        :return a list of DetectionResult.
        """
        pass

    __call__ = detect

    @abstractmethod
    def get_class_ids(self):
        """
        Get the set of potential class ids that can be detected by the detector.
        :return a set of integers containing class ids. In most case the list will start from 1,
        where 0 is supposed to be the background class and not returned.
        """
        pass

    @staticmethod
    def rgb_to_bgr(img):
        return img[..., ::-1]

    @staticmethod
    def bgr_to_rgb(img):
        return img[..., ::-1]
