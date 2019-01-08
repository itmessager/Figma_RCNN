import cv2
from PIL import Image, ImageDraw
import numpy as np
from object_detection.utils.visualization_utils import STANDARD_COLORS, draw_mask_on_image_array, \
    draw_bounding_box_on_image_array

from utils.drawing import draw_text_pil, draw_axis


def pil_to_cv_image(image_pil, output_bgr=True):
    image_cv = np.array(image_pil)
    if output_bgr:
        image_cv = image_cv[:, :, ::-1]
    return image_cv


def cv_to_pil_image(image_cv, output_rgb=True):
    if output_rgb:
        image_cv = image_cv[..., ::-1]  # Convert BGR opencv image to RGB first
    return Image.fromarray(image_cv)


def draw_bounding_box_pil(box, draw, color, image_np):
    ymin = box[0]
    xmin = box[1]
    ymax = box[2]
    xmax = box[3]
    im_width, im_height = image_np.shape[1], image_np.shape[0]
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
    draw.line([(left, top), (left, bottom), (right, bottom),
               (right, top), (left, top)], width=6, fill=color)


def draw_match_boxes_with_same_color(image_np, face_boxes, person_boxes, matched):
    image_pil = Image.fromarray(image_np)
    draw = ImageDraw.Draw(image_pil)

    i = 0
    for f, p in zip(face_boxes[0][matched[:, 0]], person_boxes[matched[:, 1]]):
        # bbox_color = (255, 0, 0) if people[id].gender == 0 else (0, 0, 255)
        i += 1
        bbox_color = STANDARD_COLORS[
            i % len(STANDARD_COLORS)]

        draw_bounding_box_pil(f, draw, bbox_color, image_np)
        draw_bounding_box_pil(p, draw, bbox_color, image_np)

    image_np = np.array(image_pil)
    return image_np


def draw_tracked_people(img_bgr, tracked_people):
    """
    Draw bounding box and mask of detected and tracked people, each with a different color based on their ids.
    :param img_bgr: Original image where people are detected and tracked.
    :param tracked_people: a list of TrackedPerson objects.
    :return A numpy
    """
    if len(tracked_people) == 0:
        return img_bgr

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # img_pil = Image.fromarray(img_rgb)

    for person in tracked_people:
        color = STANDARD_COLORS[person.id % len(STANDARD_COLORS)]
        if person.body_box is not None:
            xmin, ymin, xmax, ymax = person.body_box
            draw_bounding_box_on_image_array(img_rgb, ymin, xmin, ymax, xmax, color=color,
                                             display_str_list=['ID:{}  Score:{:.2f}'.format(person.id, person.body_score)],
                                             use_normalized_coordinates=False)
        if person.face_box is not None:
            xmin, ymin, xmax, ymax = person.face_box
            draw_bounding_box_on_image_array(img_rgb, ymin, xmin, ymax, xmax, color=color,
                                             display_str_list=['Score:{:.2f}'.format(person.face_score)],
                                             use_normalized_coordinates=False)
        if person.body_mask is not None:
            draw_mask_on_image_array(img_rgb, person.body_mask, color=color)

    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)



male_text = u"男"
female_text = u"女"
facemask_text = u"面罩"
formal_text = u"正装"
hat_text = u"帽子"
jeans_text = u"牛仔裤"
logo_text = u"标志"
longhair_text = u"长发"
longpants_text = u"长裤"
longsleeve_text = u"长袖"
shorts_text = u"短裤"
skirt_text = u"裙子"
stripe_text = u"条纹"
sunglass_text = u"太阳镜"
tshirt_text = u"T恤"


def draw_person_attributes(draw, person, face, body):
    # Draw (semi-)static attributes as text on body bounding boxes
    xmin, ymin, xmax, ymax = body
    text = [female_text if person.male == 0 else male_text]
    if person.facemask == 1:
        text.append(facemask_text)
    if person.formal == 1:
        text.append(formal_text)
    if person.hat == 1:
        text.append(hat_text)
    if person.jeans == 1:
        text.append(jeans_text)
    if person.logo == 1:
        text.append(logo_text)
    if person.longhair == 1:
        text.append(longhair_text)
    if person.longpants == 1:
        text.append(longpants_text)
    if person.longsleeve == 1:
        text.append(longsleeve_text)
    if person.shorts == 1:
        text.append(shorts_text)
    if person.skirt == 1:
        text.append(skirt_text)
    if person.stripe == 1:
        text.append(stripe_text)
    if person.sunglass == 1:
        text.append(sunglass_text)
    if person.tshirt == 1:
        text.append(tshirt_text)

    draw_text_pil(text, (xmax + 3, ymin), draw, "blue", width=1, background="white")
