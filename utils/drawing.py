from math import cos, sin

import numpy as np
from PIL import ImageFont

FONT = "wqy-zenhei.ttc"
DEFAULT_FONT_SIZE = 16
TITLE_FONT_SIZE = 48
LINE_THICKNESS = 4


def draw_text_pil(text, xy, draw, rgb, font_size=DEFAULT_FONT_SIZE, width=1, background=None):
    font = ImageFont.truetype(FONT, size=font_size)

    def draw_text_with_width(text, x, y, width, rgb, font):
        for x_off in range(0, width):
            for y_off in range(0, width):
                draw.text((x + x_off, y + y_off), text, fill=rgb, font=font)

    x = xy[0]
    y = xy[1]
    if isinstance(text, list):
        text_margin = 2

        for t in text:
            text_w, text_h = draw.textsize(t, font)
            if background is not None:
                draw.rectangle(((x, y), (x + text_w, y + text_h)), fill=background)
            draw_text_with_width(t, x, y, width, rgb, font)
            y = y + text_h + text_margin
    else:
        draw_text_with_width(text, x, y, width, rgb, font)


def draw_bounding_box_pil(box, draw, rgb):
    draw.line([(box[0], box[1]),
               (box[0], box[3]), (box[2], box[3]), (box[2], box[1]), (box[0], box[1])], fill=rgb,
              width=LINE_THICKNESS)


def draw_gaze_target_pil(xy, draw, rgb):
    radius = 5
    draw.ellipse((xy[0] - radius, xy[1] - radius, xy[0] + radius, xy[1] + radius), fill=rgb)


def draw_axis(draw, yaw, pitch, roll, tdx, tdy, size=100):
    # Convert from angular to radian degree
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    draw.line([(tdx, tdy), (x1, y1)], fill=(255, 0, 0), width=LINE_THICKNESS)
    draw.line([(tdx, tdy), (x2, y2)], fill=(0, 255, 0), width=LINE_THICKNESS)
    draw.line([(tdx, tdy), (x3, y3)], fill=(0, 0, 255), width=LINE_THICKNESS)
