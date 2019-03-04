from detection.core.s3fd_face_detector import S3fdFaceDetector
#from detection.core.tensorpack_detector import TensorPackDetector
from detection.tensorpacks.tensorpack_detector_dev import TensorPackDetector


def get_detector(model, weight_file, config):
    assert model in ['s3fd', 'tensorpack']
    if model == 's3fd':
        from detection.config.s3fd_config import config as cfg
        cfg.update_args(config)
        return S3fdFaceDetector(weight_file)
    elif model == 'tensorpack':
        from detection.config.tensorpack_config import config as cfg
        if config:
            cfg.update_args(config)
        return TensorPackDetector(weight_file)
