from detection.config.base_config import AttrDict

__all__ = ['config']

config = AttrDict()
_C = config  # short alias to reduce coding

_C.MIN_SCORE = 0.5
# Starting output layer to be used. Using 0 will include detections from all-18-4 layers/scales,
# while using higher value will ignore smaller detections but improve post-processing speed
_C.START_LAYER = 1
