import os
from abc import abstractmethod

import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util

from detection.core.detector import AbstractDetector, DetectionResult


class TensorflowModelsDetector(AbstractDetector):
    def __init__(self, model_path, label_path, num_classes, min_score=.5):
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

        self.num_classes = num_classes
        self.min_score = min_score

    def detect(self, img, rgb=True):
        if not rgb:
            img = self.bgr_to_rgb(img)

        image_np_expanded = np.expand_dims(img, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        # results is a tuple containing (boxes, scores, classes, num_detections)
        results = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        return self.to_detection_results(*results)

    def to_detection_results(self, boxes, scores, classes, num_detections):
        # Since we only have one image as input
        boxes, scores, classes, num_detections = boxes[0, :, :], scores[0], classes[0], int(num_detections[0])
        valid_indices = self.valid_result_indices(classes, num_detections)
        # Somehow the slice ops add an extra dimension at the beginning
        boxes = np.squeeze(boxes[valid_indices, :], axis=0)
        scores = scores[valid_indices]
        classes = classes[valid_indices]

        results = []
        for box, cls, score in zip(boxes, scores, classes):
            results.append(DetectionResult(class_id=cls, score=score, box=tuple(box), mask=None))
        return results

    @abstractmethod
    def valid_result_indices(self, classes, num_detection):
        pass

    def get_class_ids(self):
        return set(range(1, self.num_classes + 1))


class TensorflowModelsFaceDetector(TensorflowModelsDetector):
    root = os.path.dirname(__file__)
    PATH_TO_FACE_CKPT = os.path.join(root, '../models/wider_frozen_inference_graph.pb')
    PATH_TO_FACE_LABELS = os.path.join(root, '../dataset/wider_label_map.pbtxt')

    def __init__(self, model_path=PATH_TO_FACE_CKPT):
        super(TensorflowModelsFaceDetector, self).__init__(model_path, label_path=self.PATH_TO_FACE_LABELS, num_classes=1)

    def valid_result_indices(self, classes, num_detection):
        return list(range(num_detection))


class TensorflowModelsCocoDetector(TensorflowModelsDetector):
    root = os.path.dirname(__file__)
    PATH_TO_CKPT = os.path.join(root, '../models/coco_frozen_inference_graph.pb')
    PATH_TO_LABELS = os.path.join(root, '../dataset/mscoco_label_map.pbtxt')

    def __init__(self, model_path=PATH_TO_CKPT):
        super(TensorflowModelsCocoDetector, self).__init__(model_path, label_path=self.PATH_TO_LABELS, num_classes=90)

    def valid_result_indices(self, classes, num_detection):
        return list(range(num_detection))


class TensorflowModelsPersonDetector(TensorflowModelsCocoDetector):
    PERSON_IDX = 1

    def valid_result_indices(self, classes, num_detection):
        return np.where(classes[:num_detection] == self.PERSON_IDX)
