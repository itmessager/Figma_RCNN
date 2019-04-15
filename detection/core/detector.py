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

