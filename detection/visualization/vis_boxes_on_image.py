from PIL import Image, ImageDraw
import cv2
import numpy as np


def vis_one_image(image_bgr, boxes, resize=1, box_format='x1y1wh'):
    # Draw attribute results
    image_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    for box in boxes:
        draw_bounding_box_pil(box, draw, box_format, 'blue')

    image_disp = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    x, y = image_disp.shape[0:2]
    image_disp = cv2.resize(image_disp, (int(y * resize), int(x * resize)))
    cv2.imshow('vis boxes on image', image_disp)
    cv2.waitKey(0)


def draw_bounding_box_pil(box, draw, box_format, color):
    if box_format is 'x1y1wh':
        xmin = box[0]
        ymin = box[1]
        width = box[2]
        height = box[3]
        xmax = box[0] + width
        ymax = box[1] + height
    elif box_format is 'x1y1x2y2':
        xmin = box[0]
        ymin = box[1]
        xmax = box[2]
        ymax = box[3]
    else:
        print('the format is not support!')
    (left, right, top, bottom) = (xmin, xmax,
                                  ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom),
               (right, top), (left, top)], width=3, fill=color)


if __name__ == '__main__':
    img = cv2.imread('/root/datasets/wider attribute/train/1--Handshaking/1_Handshaking_Handshaking_1_765.jpg')

    boxes = np.array([(93.06605, 121.849365, 380.8992, 593.8957),
                      (477.803, 86.349945, 476.84357, 634.1923)])

    vis_one_image(img, boxes)
