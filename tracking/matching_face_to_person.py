import numpy as np

from sklearn.utils.linear_assignment_ import linear_assignment
from tracking.utils import iof


def matching(face_boxes, person_masks, iof_threshold=0.9):
    iof_matrix = np.zeros((len(face_boxes), len(person_masks)), dtype=np.float32)
    for f, face in enumerate(face_boxes):
        for p, person in enumerate(person_masks):
            iof_matrix[f, p] = iof(face, person)
    matched_indices = linear_assignment(-iof_matrix)

    unmatched_faces = []

    for f, face in enumerate(face_boxes):
        if f not in matched_indices[:, 0]:
            unmatched_faces.append(f)

    unmatched_persons = []
    for p, person in enumerate(person_masks):
        if p not in matched_indices[:, 1]:
            unmatched_persons.append(p)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iof_matrix[m[0], m[1]] < iof_threshold:
            unmatched_faces.append(m[0])
            unmatched_persons.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_faces), np.array(unmatched_persons)
