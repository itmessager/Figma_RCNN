import collections


def crop_and_resize(image, bboxes, size):
    assert isinstance(size, int) or (isinstance(size, tuple) and len(size) == 2)
    # TODO