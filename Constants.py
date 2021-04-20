import os
import json
import copy
import random
import pathlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np


# directory constants
ROOT_DIR = pathlib.Path('/home/weixin/neuralPoolTest/')

#  specials = list(OrderedDict.fromkeys(
            # tok for tok in [self.unk_token, self.pad_token, self.init_token,
            #                 self.eos_token] + kwargs.pop('specials', [])
PAD = 1
EOS = 3
UNK = 0
SOS = 2
# T = 1
NUM_BBOX = 36
VISUAL_FEAT = 2048
BIAS = 1000
EMB_DIM = 100


OBJECT_FUNCS = ['relate', 'relate_inv', 'relate_name', 'relate_inv_name', 'select', 'relate_attr', 'filter', 'filter_not', 'filter_h']
STRING_FUNCS = ['query_n', 'query_h', 'query', 'query_f', 'choose_n', 'choose_f', 'choose', 'choose_attr', 'choose_h', 'choose_v', 'choose_rel_inv', 'choose_subj', 'common']
BINARY_FUNCS = ['verify', 'verify_f', 'verify_h', 'verify_v', 'verify_rel', 'verify_rel_inv', 'exist', 'or', 'and', 'different', 'same', 'same_attr', 'different_attr']

BBOX_ONTOLOGY = {'darkness': ['dark', 'bright'],
                 'dryness': ['wet', 'dry'],
                 'colorful': ['colorful', 'shiny'],
                 'leaf': ['leafy', 'bare'],
                 'emotion': ['happy', 'calm'],
                 'sports': ['baseball', 'tennis'],
                 'flatness': ['flat', 'curved'],
                 'lightness': ['light', 'heavy'],
                 'gender': ['male', 'female'],
                 'width': ['wide', 'narrow'],
                 'depth': ['deep', 'shallow'],
                 'hardness': ['hard', 'soft'],
                 'cleanliness': ['clean', 'dirty'],
                 'switch': ['on', 'off'],
                 'thickness': ['thin', 'thick'],
                 'openness': ['open', 'closed'],
                 'height': ['tall', 'short'],
                 'length': ['long', 'short'],
                 'fullness': ['full', 'empty'],
                 'age': ['young', 'old'],
                 'size': ['large', 'small'],
                 'pattern': ['checkered', 'striped', 'dress', 'dotted'],
                 'shape': ['round', 'rectangular', 'triangular', 'square'],
                 'activity': ['waiting', 'staring', 'drinking', 'playing', 'eating', 'cooking', 'resting', 'sleeping', 'posing', 'talking',
                              'looking down', 'looking up', 'driving', 'reading', 'brushing teeth', 'flying', 'surfing', 'skiing', 'hanging'],
                 'pose': ['walking', 'standing', 'lying', 'sitting', 'running', 'jumping', 'crouching', 'bending', 'smiling', 'grazing'],
                 'material': ['wood', 'plastic', 'metal', 'glass', 'leather', 'leather', 'porcelain', 'concrete', 'paper', 'stone', 'brick'],
                 'color': ['white', 'red', 'black', 'green', 'silver', 'gold', 'khaki', 'gray', 'dark', 'pink', 'dark blue', 'dark brown',
                           'blue', 'yellow', 'tan', 'brown', 'orange', 'purple', 'beige', 'blond', 'brunette', 'maroon', 'light blue', 'light brown']}

SCENE_ONTOLOGY = {'location': ['indoors', 'outdoors'],
                 'weather': ['clear', 'overcast', 'cloudless', 'cloudy', 'sunny', 'foggy', 'rainy'],
                 'room': ['bedroom', 'kitchen', 'bathroom', 'living room'],
                 'place': ['road', 'sidewalk', 'field', 'beach', 'park', 'grass', 'farm', 'ocean', 'pavement',
                           'lake', 'street', 'train station', 'hotel room', 'church', 'restaurant', 'forest', 'path',
                           'display', 'store', 'river', 'sea', 'yard', 'airport', 'parking lot']}

ONTOLOGY = copy.deepcopy(BBOX_ONTOLOGY)
ONTOLOGY.update(SCENE_ONTOLOGY)

BBOX_ATTR = list(BBOX_ONTOLOGY.keys())

SCENE_ATTR = list(SCENE_ONTOLOGY.keys())

BBOX_ATTRIBUTES = {}
for k, v in BBOX_ONTOLOGY.items():
    for i, item in enumerate(v):
        if item in BBOX_ATTRIBUTES:
            BBOX_ATTRIBUTES[item].append((BBOX_ATTR.index(k), i))
        else:
            BBOX_ATTRIBUTES[item] = [(BBOX_ATTR.index(k), i)]

SCENE_ATTRIBUTES = {}
for k, v in SCENE_ONTOLOGY.items():
    for i, item in enumerate(v):
        if item in SCENE_ATTRIBUTES:
            SCENE_ATTRIBUTES[item].append((SCENE_ATTR.index(k), i))
        else:
            SCENE_ATTRIBUTES[item] = [(SCENE_ATTR.index(k), i)]

# with open(ROOT_DIR / 'GraphVQA/meta_info/GQA_hypernym.json') as f:
#     hypernym = json.load(f)

with open(ROOT_DIR / 'GraphVQA/meta_info/objects.json') as f:
    OBJECTS_INV = json.load(f)
    OBJECTS = {k: i for i, k in enumerate(OBJECTS_INV)}

with open(ROOT_DIR / 'GraphVQA/meta_info/predicates.json') as f:
    RELATIONS_INV = json.load(f)
    RELATIONS = {k: i for i, k in enumerate(RELATIONS_INV)}

with open(ROOT_DIR / 'GraphVQA/meta_info/attributes.json') as f:
    ATTRIBUTES_INV = json.load(f)
    ATTRIBUTES = {k: i for i, k in enumerate(ATTRIBUTES_INV)}

with open(ROOT_DIR / 'GraphVQA/meta_info/obj2attribute.json') as f:
    mapping = json.load(f)


OBJ2ATTRIBUTES = {}
for k, vs in mapping.items():
    tmp = set()
    for v in vs:
        if v in BBOX_ATTRIBUTES:
            for attr_k, _ in BBOX_ATTRIBUTES[v]:
                tmp.add(attr_k)
    OBJ2ATTRIBUTES[k] = list(tmp)


def show_im(k, x, y, w, h, title):
    im = np.array(Image.open(ROOT_DIR / "Downloads/allImages/images/{}.jpg".format(k)), dtype=np.uint8)
    # Create figure and axes
    fig, ax = plt.subplots(1)
    ax.imshow(im)
    # Create a Rectangle patch
    rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
    # Add the patch to the Axes
    ax.set_title(title)
    ax.add_patch(rect)

    plt.show()


def show_im_bboxes(k, coordinates):
    im = np.array(Image.open(ROOT_DIR / "Downloads/allImages/images/{}.jpg".format(k)), dtype=np.uint8)
    height = im.shape[0]
    width = im.shape[1]
    # Create figure and axes
    fig, ax = plt.subplots(1)
    ax.imshow(im)
    colors = ['red', 'yellow', 'black', 'blue', 'orange', 'grey', 'cyan', 'green', 'purple']
    # Create a Rectangle patch
    for coordinate in coordinates:
        x, y = coordinate[0] * width, coordinate[1] * height
        w, h = (coordinate[2] - coordinate[0]) * width, (coordinate[3] - coordinate[1]) * height
        color = random.choice(colors)
        rect1 = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor=color, facecolor='none')
        ax.add_patch(rect1)

    plt.show()


def intersect(bbox1, bbox2, contained=False, option="xywh"):
    if option == 'xywh':
        x_inter = max(bbox1[0], bbox2[0])
        y_inter = max(bbox1[1], bbox2[1])
        x_p_inter = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
        y_p_inter = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
        intersect_area = max(x_p_inter - x_inter, 0) * max(y_p_inter - y_inter, 0)
        whole = bbox1[2] * bbox1[3] + bbox2[2] * bbox2[3] - intersect_area
    elif option == 'x1y1x2y2':
        x_inter = max(bbox1[0], bbox2[0])
        y_inter = max(bbox1[1], bbox2[1])
        x_p_inter = min(bbox1[2], bbox2[2])
        y_p_inter = min(bbox1[3], bbox2[3])
        intersect_area = max(x_p_inter - x_inter, 0) * max(y_p_inter - y_inter, 0)
        whole = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]) + (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]) - intersect_area
    else:
        raise NotImplementedError

    if contained:
        return intersect_area / (whole + 0.01), intersect_area / (bbox1[2] * bbox1[3] + 0.01)
    else:
        return intersect_area / (whole + 0.01)

def parse_program(string):
    if '=' in string:
        result, function = string.split('=')
    else:
        function = string
        result = "?"

    func, arguments = function.split('(')
    if len(arguments) == 1:
        return result, func, []
    else:
        arguments = list(map(lambda x: x.strip(), arguments[:-1].split(',')))
        return result, func, arguments
