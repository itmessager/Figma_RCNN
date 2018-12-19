from gaze.datasets.gazecapture import GazeCapture
import cv2
import numpy
import random

dataset = GazeCapture('/root/fast-storage/GazeCapture', 'train', transform=None, use_eye=True)


for _ in range(10):
    i = random.randint(0, len(dataset))
    [img, face, leye, reye], target = dataset[i]

    open_cv_image = numpy.array(img)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    cv2.rectangle(open_cv_image, (face[0], face[1]), (face[0]+face[2], face[1]+face[3]), (255, 255, 0), 3)
    cv2.rectangle(open_cv_image, (leye[0], leye[1]), (leye[0]+leye[2], leye[1]+leye[3]), (255, 255, 0), 3)
    cv2.rectangle(open_cv_image, (reye[0], reye[1]), (reye[0]+reye[2], reye[1]+reye[3]), (255, 255, 0), 3)
    cv2.imshow('image', open_cv_image)
    cv2.waitKey(0)