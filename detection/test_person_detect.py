import numpy as np
import os
import tensorflow as tf
import cv2

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import argparse

parser = argparse.ArgumentParser()

args = parser.parse_args()

# Variables
# What model to download.
# MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = '/root/models/detection/models/person_frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
#PATH_TO_LABELS = os.path.join('dataset', 'face_label_map.pbtxt')
PATH_TO_LABELS = '/root/models/detection/dataset/mscoco_label_map.pbtxt'
NUM_CLASSES = 90

# Load the model
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map,
                                                            max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Detection

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test.py the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
# PATH_TO_TEST_IMAGES_DIR = 'test_images'
# TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

# Size, in inches, of the output images.
# IMAGE_SIZE = (12, 8)

cap = cv2.VideoCapture(0)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out_video = cv2.VideoWriter('video.avi', fourcc, 30, (1280,720))
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        frame_count = 0
        while True:
            grabbed, image_np = cap.read()
            if not grabbed:
                break
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            # image_np = np.asarray(image)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]

            image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)


            image_np_expanded = np.expand_dims(image_rgb, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # Actual detection.
            # TODO Output person detection results only
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.pip
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
            # out_video.write(image_np)
            frame_count += 1
            print("Processed {} frames".format(frame_count))
            cv2.imshow('video', image_np)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

#out_video.release()
cap.release()
