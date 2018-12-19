"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016 Alex Bewley alex@dynamicdetection.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np

from filterpy.kalman import KalmanFilter

from sklearn.utils.linear_assignment_ import linear_assignment
from tracking.utils import iou


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., [score]]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """
    count = 0  # Class level variable summarize the total counts of objects

    def __init__(self, bbox, max_trajectory_len=10):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity models
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count + 1  # +1 as MOT benchmark requires positive
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        if len(bbox) > 4:
            self.score = bbox[4]
        self.trajectory = np.expand_dims(bbox[:4], axis=0)
        self.max_trajectory_len = max_trajectory_len

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
        if len(bbox) > 4:
            self.score = bbox[4]
        if len(self.trajectory) == self.max_trajectory_len:
            self.trajectory = np.concatenate([bbox[:4].reshape(1,4), self.trajectory[:-1]])
        else:
            self.trajectory = np.concatenate([bbox[:4].reshape(1,4), self.trajectory])

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x, self.score))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x, self.score)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if len(trackers) == 0:
        return np.empty((0, 2)), np.arange(len(detections)), np.empty((0))
    if len(detections) == 0 or detections.shape[0] == 0:
        return np.empty((0, 2)), np.empty(0), np.arange(len(trackers))
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)
    matched_indices = linear_assignment(-iou_matrix)

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=1, min_hits=3, max_trajectory_len=10):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0  # TODO What if the model needs to run continuously and integer got overflown?
        self.max_trajectory_len = max_trajectory_len
        self.ids_output_before = set()

    def update(self, dets):
        """
        Requires: this method must be called once for each frame even with empty detections.
        NOTE: The number of objects returned may differ from the number of detections provided.
        :param dets: a numpy array of detections in the format [[x1,y1,x2,y2,score],...] or [[x1,y1,x2,y2,score],...]
        :return: a tuple of (tracked_objects, associated_detections, remove_ids), where tracked_objects contains tracked
        objects, which is a [N, 6] (or [N, 5] if input doesn't have score)
         numpy array where the second dimension is in the format of [x1, y1, x2, y2, score, id], and associated_detections
         is a list containing index into the input detection list which corresponds to each tracked_object, and removed_ids
         contains a list of outdated tracked_object ids which are supposed to be removed by other dependent system.
        """
        n_cols = dets.shape[1]
        self.frame_count += 1

        # Get predicted locations from existing trackers and remove any invalid trackers
        valid_trackers = []
        predicted_positions = []
        for i, tracker in enumerate(self.trackers):
            predicted_pos = tracker.predict()[0][:n_cols]  # Use Karlman filter to predict next position of each tracker
            # Only include valid trackers with non-nan values
            if not np.any(np.isnan(predicted_pos)):
                valid_trackers.append(tracker)
                predicted_positions.append(predicted_pos)
        trks = np.array(predicted_positions)
        self.trackers = valid_trackers

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks)

        # Indicate which detection each tracker is associated to in current frame
        tracker_associated_det_ind = [None] * len(self.trackers)

        # Update matched trackers with assigned detections
        for det_ind, tracker_ind in matched:
            self.trackers[tracker_ind].update(dets[det_ind, :])
            tracker_associated_det_ind[tracker_ind] = det_ind

        # Create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            tracker = KalmanBoxTracker(dets[i, :], self.max_trajectory_len)
            self.trackers.append(tracker)
            # tracker_associated_det_ind[i] = tracker.id
            tracker_associated_det_ind.append(i)

        # Decide what tracked detection to returned and remove dead tracklet
        valid_trackers = []
        returned_trackers = []
        returned_associated_det_ind = []
        removed_ids = []
        for i, tracker in enumerate(self.trackers):
            # Remove dead tracklet and tell dependent module to remove them too, only if Sort has been output such
            # tracklet before, as there might be tracklet that are never output and thus no need to tell other modules
            # to remove
            if tracker.time_since_update > self.max_age and tracker.id in self.ids_output_before:
                removed_ids.append(tracker.id)
                self.ids_output_before.remove(tracker.id)
                continue

            # Append qualified trackers to the return list
            if (tracker.time_since_update < 1) and (tracker.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                d = tracker.get_state()[0]
                returned_trackers.append(np.concatenate((d, [tracker.id])))
                returned_associated_det_ind.append(tracker_associated_det_ind[i])
                self.ids_output_before.add(tracker.id)
            valid_trackers.append(tracker)
        self.trackers = valid_trackers

        if len(returned_trackers) > 0:
            return np.array(returned_trackers), returned_associated_det_ind, removed_ids
        return np.empty((0, n_cols + 1), dtype=np.float32), returned_associated_det_ind, removed_ids

    def reset(self):
        self.trackers = []
        self.frame_count = 0

    def get_valid_ids(self):
        return [trk.id for trk in self.trackers]