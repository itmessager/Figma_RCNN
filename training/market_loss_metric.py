import torch.nn.functional as F
from ignite.metrics import Loss
from training.cmc_metric import CmcMetric

# to calculate the loss for each batch
training_loss = F.cross_entropy

# to define a class
testing_loss = Loss(F.cross_entropy)
cmc_metric = CmcMetric()
