import cv2
from PIL import Image, ImageDraw
import numpy as np
from object_detection.utils.visualization_utils import STANDARD_COLORS, \
    draw_bounding_box_on_image_array


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


male_text = u" Male;  "
female_text = u" Female;  "
unspecified_text = u" Unspecified;  "
facemask_text = u"Face mask;  "
formal_text = u"Formal;  "
hat_text = u"Hat;  "
jeans_text = u"Jeans;  "
logo_text = u"Logo;  "
longhair_text = u"Long Hair;  "
longpants_text = u"Long pants;  "
longsleeve_text = u"Long sleeve;  "
shorts_text = u"Shorts;  "
skirt_text = u"Skirt;  "
stripe_text = u"Stripe;  "
sunglass_text = u"Sunglasses;  "
tshirt_text = u"T-shirt;  "


def label_to_text(label):
    if label == 1:
        return male_text
    elif label == 0:
        return female_text
    else:
        return unspecified_text


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
        color = STANDARD_COLORS[np.random.randint(1, 100) % len(STANDARD_COLORS)]
        text = label_to_text(person.male)

        if person.facemask == 1:
            text += facemask_text
        if person.formal == 1:
            text += formal_text
        if person.hat == 1:
            text += hat_text
        if person.jeans == 1:
            text += jeans_text
        if person.logo == 1:
            text += logo_text
        if person.longhair == 1:
            text += longhair_text
        if person.longpants == 1:
            text += longpants_text
        if person.longsleeve == 1:
            text += longsleeve_text
        if person.shorts == 1:
            text += shorts_text
        if person.skirt == 1:
            text += skirt_text
        if person.stripe == 1:
            text += stripe_text
        if person.sunglass == 1:
            text += sunglass_text
        if person.tshirt == 1:
            text += tshirt_text

        if person.box is not None:
            xmin, ymin, xmax, ymax = person.box
            draw_bounding_box_on_image_array(img_rgb, ymin, xmin, ymax, xmax, color=color, thickness=2,
                                             display_str_list=[text[:-3]],
                                             use_normalized_coordinates=False)

    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
