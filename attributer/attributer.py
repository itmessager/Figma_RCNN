import numpy as np


class PersonAttrs:
    def __init__(self, attribute):
        self.facemask = attribute.facemask
        self.formal = attribute.formal
        self.hat = attribute.hat
        self.jeans = attribute.jeans
        self.logo = attribute.logo
        self.longhair = attribute.longhair
        self.longpants = attribute.longpants
        self.longsleeve = attribute.longsleeve
        self.male = attribute.male
        self.shorts = attribute.shorts
        self.skirt = attribute.skirt
        self.stripe = attribute.stripe
        self.sunglass = attribute.sunglass
        self.tshirt = attribute.tshirt


class PersonBoxes:
    def __init__(self, box):
        self.body_box = box.box
        self.body_mask = box.mask
        self.body_score = box.score
        self.id = np.random.randint(1, 20)

