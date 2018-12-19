from __future__ import division
from ignite.metrics.metric import Metric
import numpy as np
import heapq
import torch


def recognition_rate_at_k(probe_x, probe_y, gallery_x, gallery_y, k, measure):
    """Compute the recognition rate at a given level `k`"""

    # (75,75)
    label_eq_mat = np.equal(probe_y.reshape(-1, 1), gallery_y.reshape(1, -1)).astype(np.float)
    # (75,1)
    num_relevant = np.minimum(np.float(k), np.sum(label_eq_mat, axis=1))
    predictions = np.exp(-measure(probe_x, gallery_x))  # (75,75) Compute similarity.

    # top_k large numbers of each row
    prediction_indices = []
    data_of_row_range = range(len(predictions[0]))
    for row in range(len(predictions)):
        prediction_indices.append(heapq.nlargest(k, data_of_row_range, predictions[row].take))

    # (75, k)
    label_mat = []
    for row in range(len(predictions)):
        label_mat.append(gallery_y[prediction_indices[row]])

    # (75,k)
    label_eq_mat = np.equal(label_mat, probe_y.reshape(-1, 1)).astype(np.float)

    true_positives_at_k = np.sum(label_eq_mat, axis=1)  # (75,1)

    return true_positives_at_k / num_relevant


def cosine_distance(a, b=None):
    """Compute element-wise cosine distance between `a` and `b` """
    a_normed = a / np.sqrt(np.square(a).sum(axis=1)).reshape(-1, 1)  # to unit each row in matrix 'a'

    b_normed = b / np.sqrt(np.square(b).sum(axis=1)).reshape(-1, 1)

    return 1 - np.dot(a_normed, b_normed.T)


def cmc_metric_at_k(probe_x, probe_y, gallery_x, gallery_y, k=1):
    recognition_rate = recognition_rate_at_k(  # (75, 1) that is the recognition rate for each probe
        probe_x, probe_y, gallery_x, gallery_y, k, cosine_distance)  # total number of probes is 75

    return np.mean(recognition_rate)


class CmcMetric(Metric):
    """
    Calculates the categorical accuracy.

    - `update` must receive output of the form `(y_pred, y)`.
    - `y_pred` must be in the following shape (batch_size, num_categories, ...)
    - `y` must be in the following shape (batch_size, ...)
    """

    # pay attention to the torch.Tensor or numpy.ndarray
    _features, _targets = None, None

    def reset(self):
        self._features = torch.tensor([], dtype=torch.float)
        self._targets = torch.tensor([], dtype=torch.int)

    def update(self, output):
        features, targets = output
        # shape; [batch_size, 128]
        features = features.type_as(self._features)
        targets = targets.type_as(self._targets)

        self._features = torch.cat([self._features, features], dim=0)
        self._targets = torch.cat([self._targets, targets], dim=0)

    def compute(self):
        # change type to ndarray
        features = self._features.numpy()
        targets = self._targets.numpy()

        # get data_length

        data_len = len(targets)
        if data_len <= 0:
            return 0
        num_probes = np.int(data_len/2)  # num_probes = num_gallery_images

        # seperate the features between the probes and gallery
        probe_x = features[:num_probes]
        probe_y = targets[:num_probes]

        gallery_x = features[num_probes:data_len]
        gallery_y = targets[num_probes:data_len]

        return cmc_metric_at_k(probe_x, probe_y, gallery_x, gallery_y)
