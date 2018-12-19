import torch.nn.functional as F
from ignite.metrics import CategoricalAccuracy, Loss, MeanAbsoluteError

from attributer.attributes import FaceAttributes
from training.metric_utils import ScaledError

_metrics = {FaceAttributes.AGE: ScaledError(MeanAbsoluteError(), 50),
            FaceAttributes.GENDER: CategoricalAccuracy(),
            FaceAttributes.EYEGLASSES: CategoricalAccuracy(),
            FaceAttributes.RECEDING_HAIRLINES: CategoricalAccuracy(),
            FaceAttributes.SMILING: CategoricalAccuracy(),
            FaceAttributes.HEAD_YAW_BIN: CategoricalAccuracy(),
            FaceAttributes.HEAD_PITCH_BIN: CategoricalAccuracy(),
            FaceAttributes.HEAD_ROLL_BIN: CategoricalAccuracy(),
            FaceAttributes.HEAD_YAW: MeanAbsoluteError(),
            FaceAttributes.HEAD_PITCH: MeanAbsoluteError(),
            FaceAttributes.HEAD_ROLL: MeanAbsoluteError(),
            }

_losses = {FaceAttributes.AGE: F.l1_loss,
           FaceAttributes.GENDER: F.cross_entropy,
           FaceAttributes.EYEGLASSES: F.cross_entropy,
           FaceAttributes.RECEDING_HAIRLINES: F.cross_entropy,
           FaceAttributes.SMILING: F.cross_entropy,
           FaceAttributes.HEAD_YAW_BIN: F.cross_entropy,
           FaceAttributes.HEAD_PITCH_BIN: F.cross_entropy,
           FaceAttributes.HEAD_ROLL_BIN: F.cross_entropy,
           FaceAttributes.HEAD_YAW: F.l1_loss,
           FaceAttributes.HEAD_PITCH: F.l1_loss,
           FaceAttributes.HEAD_ROLL: F.l1_loss,
           }

loss_fns = [_losses[attr] for attr in FaceAttributes]
# losses = {attr: Loss(_losses[attr]) for attr in Attributes}
losses = [Loss(_losses[attr]) for attr in FaceAttributes]
metrics = [_metrics[attr] for attr in FaceAttributes]
