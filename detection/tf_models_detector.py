import numpy as np
import tensorflow as tf
import cv2
import timeit

import utils.viz_utils as fd

from tracking.matching_face_to_person import matching
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


PATH_TO_FACE_CKPT = '/root/models/detection/models/frozen_inference_graph.pb'
PATH_TO_FACE_LABELS = '/root/models/detection/dataset/wider_label_map.pbtxt'

PATH_TO_BODY_CKPT = '/root/models/detection/models/person_frozen_inference_graph.pb'
PATH_TO_BODY_LABELS = '/root/models/detection/dataset/mscoco_label_map.pbtxt'

class TensorflowModelsDetector:
    def __init__(self, model_path, label_path, num_classes):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        self.detection_graph = detection_graph

        # Initialize session
        self.sess = tf.Session(graph=self.detection_graph)

        # Loading label map
        label_map = label_map_util.load_labelmap(label_path)
        categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                    max_num_classes=num_classes,
                                                                    use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

    def __call__(self, image_rgb):
        image_np_expanded = np.expand_dims(image_rgb, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        return boxes, scores, classes,num_detections, self.category_index



class FaceModelsDetector(TensorflowModelsDetector):
    def __init__(self, model_path=PATH_TO_FACE_CKPT, label_path=PATH_TO_FACE_LABELS, num_classes=1):
        super(FaceModelsDetector, self).__init__(model_path, label_path, num_classes)


class PersonModelsDetector(TensorflowModelsDetector):
    PERSON_IDX = 1

    def __init__(self, model_path=PATH_TO_BODY_CKPT, label_path=PATH_TO_BODY_LABELS, num_classes=90):
        super(PersonModelsDetector, self).__init__(model_path, label_path, num_classes)

    def __call__(self, image_rgb):
        boxes, scores, classes, num_detections, category_index =super(PersonModelsDetector, self).__call__(image_rgb)

        # Since we only have one image as input
        boxes, scores, classes, num_detections = boxes[0, :, :], scores[0], classes[0], int(num_detections[0])

        valid_person_indices = np.where(classes[:num_detections] == self.PERSON_IDX)
        person_boxes = np.squeeze(boxes[valid_person_indices, :], axis=0)  # Somehow the slice ops add an extra dimension at the beginning
        person_scores = scores[valid_person_indices]
        person_classes = classes[valid_person_indices]

        return person_boxes, person_scores, person_classes, category_index

fmd = FaceModelsDetector()
pmd = PersonModelsDetector()

cap = cv2.VideoCapture('output/2018-08-30-170107.webm')
width, height= cap.get(3),cap.get(4)
print(width, height)
frame_count = 0
while True:

    t0 = timeit.default_timer()

    grabbed, image_np = cap.read()
    if not grabbed:
        break
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    face_boxes, face_scores, face_classes, num_faces, face_category_index= fmd(image_rgb)
    person_boxes, person_scores, person_classes, person_category_index = pmd(image_rgb)


    threshold_face_indices = np.where(face_scores[0] >= 0.3)
    threshold_face_boxes = face_boxes[0][threshold_face_indices]

    threshold_person_indices = np.where(person_scores >= 0.3)
    threshold_person_boxes = person_boxes[threshold_person_indices]

    matched, unmatched_faces, unmatched_persons = matching(threshold_face_boxes,threshold_person_boxes,0.8)

    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(face_boxes),
        np.squeeze(face_classes).astype(np.int32),
        np.squeeze(face_scores),
        face_category_index,
        use_normalized_coordinates=True,
        line_thickness=3,
        min_score_thresh=0.5)


    if len(person_scores) > 0:
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            person_boxes,
            person_classes.astype(np.int32),
            person_scores,
            person_category_index,
            use_normalized_coordinates=True,
            line_thickness=3,
            min_score_thresh = 0.5
        )

    image_np = fd.draw_match_boxes_with_same_color(image_np,face_boxes,person_boxes,matched)

    frame_count += 1
    print("Processed {} frames".format(frame_count))

    elapsed = timeit.default_timer() - t0
    print(elapsed)
    cv2.imshow('video', image_np)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

