from enum import Enum, unique

@unique
class Shape(Enum):
    circle = 0
    semicircle = 1
    quarter_circle = 2
    triangle = 3
    square = 4
    rectangle = 5
    trapezoid = 6
    pentagon = 7
    hexagon = 8
    heptagon = 9
    octagon = 10
    star = 11
    cross = 12

@unique
class Color(Enum):
    red = 0
    orange = 1
    yellow = 2
    green = 3
    blue = 4
    purple = 5
    brown = 6
    black = 7
    gray = 8
    white = 9

def alphanumeric_to_num(name):
    if(ord(name)<65):
        return ord(name) - 48
    else:
        return ord(name) - 55

def num_to_alphanumeric(num):
    return {
    0:'0',
    1:'1',
    2:'2',
    3:'3',
    4:'4',
    5:'5',
    6:'6',
    7:'7',
    8:'8',
    9:'9',
    10:'A',
    11:'B',
    12:'C',
    13:'D',
    14:'E',
    15:'F',
    16:'G',
    17:'H',
    18:'I',
    19:'J',
    20:'K',
    21:'L',
    22:'M',
    23:'N',
    24:'O',
    25:'P',
    26:'Q',
    27:'R',
    28:'S',
    29:'T',
    30:'U',
    31:'V',
    32:'W',
    33:'X',
    34:'Y',
    35:'Z'}.get(num)
