from enum import Enum


class FaceAttributes(Enum):
    AGE = 0
    GENDER = 1
    EYEGLASSES = 2
    RECEDING_HAIRLINES = 3
    SMILING = 4
    HEAD_YAW_BIN = 5
    HEAD_PITCH_BIN = 6
    HEAD_ROLL_BIN = 7
    HEAD_YAW = 8
    HEAD_PITCH = 9
    HEAD_ROLL = 10

    def __str__(self):
        return self.name.lower()


# Enum class for attributes included in Wider Attribute dataset only
class WiderAttributes(Enum):
    # { 0”：“男性”，“1”：“长发”，“2”：“太阳镜”“3”：“帽子”，“4”：“T-shirt”，“5”：“长袖”，“6”：“正装”,
    # “7”：“短裤”，“8”：“牛仔裤”“9”：“长裤”“10”：“裙子”，“11”：“面罩”，“12”：“标志”“13”：“条纹”}
    MALE = 0
    LONGHAIR = 1
    SUNGLASS = 2
    HAT = 3
    TSHIRT = 4
    LONGSLEEVE = 5
    FORMAL = 6
    SHORTS = 7
    JEANS = 8
    LONGPANTS = 9
    SKIRT = 10
    FACEMASK = 11
    LOGO = 12
    STRIPE = 13

    def __str__(self):
        return self.name.lower()


class Gender(Enum):
    MALE = 0
    FEMALE = 1
    UNCERTAIN = 2

    def get_attr_type(self):
        return ErisedAttributes.GENDER

    def __str__(self):
        return self.name.lower()


class AgeGroup(Enum):
    UNDER_TWO = 0
    THREE_TO_FIVE = 1
    SIX_TO_TWELVE = 2
    THIRTEEN_TO_EIGHTEEN = 3
    NINETEEN_TO_TWENTY_THREE = 4
    TWENTY_FOUR_TO_TWENTY_NINE = 5
    THIRTY_TO_THIRTY_FIVE = 6
    THIRTY_SIX_TO_FORTY_FIVE = 7
    FORTY_SIX_TO_SIXTY = 8
    ABOVE_SIXTY = 9
    UNCERTAIN = 10

    def get_attr_type(self):
        return ErisedAttributes.AGE

    @classmethod
    def age_to_group(cls, age):
        assert isinstance(age, int) or isinstance(age, float)

        if age < 3:
            return cls.UNDER_TWO
        if age < 6:
            return cls.THREE_TO_FIVE
        if age < 13:
            return cls.SIX_TO_TWELVE
        if age < 19:
            return cls.THIRTEEN_TO_EIGHTEEN
        if age < 24:
            return cls.NINETEEN_TO_TWENTY_THREE
        if age < 30:
            return cls.TWENTY_FOUR_TO_TWENTY_NINE
        if age < 36:
            return cls.THIRTY_TO_THIRTY_FIVE
        if age < 46:
            return cls.THIRTY_SIX_TO_FORTY_FIVE
        if age < 61:
            return cls.FORTY_SIX_TO_SIXTY
        return cls.ABOVE_SIXTY

    def __str__(self):
        return self.name.lower()


class Pregnancy(Enum):
    NON_PREGNANT = 0
    PREGNANT = 1
    UNCERTAIN = 2

    def get_attr_type(self):
        return ErisedAttributes.PREGNANT

    def __str__(self):
        return self.name.lower()


class CarryKids(Enum):
    NON_CARRYING = 0
    CARRYING = 1
    UNCERTAIN = 2

    def get_attr_type(self):
        return ErisedAttributes.CARRY_KIDS

    def __str__(self):
        return self.name.lower()


class ErisedAttributes(Enum):
    GENDER = 0
    AGE = 1
    AGE_GROUP = 2
    # DRESS = 0
    # GLASSES = 1
    # UNDERCUT = 2
    # GREASY = 3
    # PREGNANT = 4
    # AGE = 5
    # FIGURE = 6
    # HAIRCOLOR = 7
    # ALOPECIA = 8
    # TOTTOO = 9
    # CARRY_KIDS = 10
    # GENDER = 11

    def __str__(self):
        return self.name.lower()

    @staticmethod
    def names():
        return [a.name.lower() for a in ErisedAttributes]


class AttributeType(Enum):
    BINARY = 0
    MULTICLASS = 1
    NUMERICAL = 2


class Attribute:
    def __init__(self, key, type, maybe_unrecognizable=False):
        assert isinstance(key, Enum)
        assert isinstance(type, AttributeType)
        self.key = key
        self.name = str(key)
        self.data_type = type
        self.maybe_unrecognizable = maybe_unrecognizable

    # Merge two attribute of same key to make them compatible
    def merge(self, other_attribute):
        assert isinstance(other_attribute, Attribute)
        assert self.key == other_attribute.key
        assert self.data_type == other_attribute.data_type

        return Attribute(self.key, self.data_type, self.maybe_unrecognizable or other_attribute.maybe_unrecognizable)

    def __str__(self):
        return self.name


def get_attribute_names(attributes, specified_attrs=[]):
    assert isinstance(attributes, list)
    assert isinstance(specified_attrs, list)

    names = []
    for attr in attributes:
        assert isinstance(attr, Attribute)

        if not specified_attrs:
            names.append(attr.name)
            if attr.maybe_unrecognizable:
                names.append(attr.name + '/recognizable')
        else:
            if attr.name in specified_attrs:
                names.append(attr.name)
                if attr.maybe_unrecognizable:
                    names.append(attr.name + '/recognizable')
    return names
