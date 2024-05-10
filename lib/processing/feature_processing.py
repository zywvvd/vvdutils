import cv2
import numpy as np
from .image_processing import to_gray_image


def get_hu_fea(img):
    img = to_gray_image(img)
    fea_dict = cv2.moments(img)
    keys = list(fea_dict)
    keys.sort()
    fea = list()
    for key in keys:
        fea.append(fea_dict[key])
    return np.array(fea)