from abc import abstractmethod
from collections import namedtuple

import numpy as np

from detection.core.detector import DetectionResult, DetectionFaceResult
from tracking.matching_face_to_person import matching
from tracking.sort import Sort

TrackedPerson = namedtuple(
    'TrackedPerson',
    ['id', 'face_box', 'face_score', 'body_box', 'body_score', 'body_mask'])
"""
id: Person's id
face_box: (xmin, ymin, xmax, ymax) in image original space
face_score: float
body_box: (xmin, ymin, xmax, ymax) in image original space
body_score: float
body_mask: None, or a binary image of the original image shape
"""


class PersonTracker:
    def __init__(self):
        # self.sort = MaskSort(max_age=3, max_trajectory_len=20)
        self.sort = Sort(max_age=3, max_trajectory_len=20)

    # The current simple logic is to map face to body only on this frame and ignore any previous faces
    def update(self, face_results, body_results, img, rgb=True):
        """
        Requires: this method must be called once for each frame even with empty detections.
        NOTE: The number of objects returned may differ from the number of detections provided.
        :param img: original image
        :param face_results: a list of DetectionResult of face.
        :param body_results: a list of DetectionResult of body
        :return: (tracked_people, removed_ids) where tracked_people is a list of TrackedPerson and
        removed_ids is a list of ids of people which copmletely lost track
        """
        assert isinstance(face_results, list)
        assert isinstance(body_results, list)
        for r in face_results:
            assert isinstance(r, DetectionFaceResult)
        for r in body_results:
            assert isinstance(r, DetectionResult)

        # Sort expect bounding boxes to be passed in the form of [[xmin, ymin, xmax, ymax, score],...]
        body_boxes = np.array([[*r.box, r.score] for r in body_results])
        body_masks = [r.mask for r in body_results]

        tracked_people, associated_det_inds, removed_ids = self.sort.update(body_boxes)

        # Match face against tracked people
        # tracked_body_boxes = tracked_people[:, :4]  # tracked_people is [[xmin, ymin, xmax, ymax, score, id],...]
        tracked_body_masks = [body_results[det_ind].mask for det_ind in associated_det_inds]
        face_boxes = np.array([[*r.box, r.score] for r in face_results])
        matched, unmatched_faces, unmatched_bodies = matching(face_boxes, tracked_body_masks, 0.8)

        # Construct return structure
        results = []
        for face_ind, body_ind in matched:
            results.append(
                self._to_tracked_person(tracked_people[body_ind], face_boxes[face_ind],
                                        body_results[associated_det_inds[body_ind]].mask))
        for body_ind in unmatched_bodies:
            results.append(self._to_tracked_person(tracked_people[body_ind], None,
                                                   body_results[associated_det_inds[body_ind]].mask))

        # Just ignore any unmatched faces for now
        return results, removed_ids

    @staticmethod
    def _to_tracked_person(body, face, mask):
        if face is not None:
            return TrackedPerson(id=int(body[5]), face_box=face[:4], face_score=np.asscalar(face[4]), body_box=body[:4],
                                 body_score=np.asscalar(body[4]), body_mask=mask)
        else:
            return TrackedPerson(id=int(body[5]), face_box=None, face_score=None, body_box=body[:4],
                                 body_score=np.asscalar(body[4]), body_mask=mask)
