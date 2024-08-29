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


def get_SIFT_fea(image, fea_num=None):
    sift = cv2.SIFT_create(fea_num)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 将彩色图片转换成灰度图

    (kps, des) = sift.detectAndCompute(gray, None)

    # descriptor = cv2.ORB_create()  # 建立ORB生成器
    # # 检测ORB特征点，并计算描述符
    # (kps, des) = descriptor.detectAndCompute(image, None)

    return kps, des  # 返回特征点集，及对应的描述特征


def get_ORB_fea(image, fea_num=None):
    orb = cv2.ORB_create(fea_num)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 将彩色图片转换成灰度图
    (kps, des) = orb.detectAndCompute(gray, None)

    # descriptor = cv2.ORB_create()  # 建立ORB生成器
    # # 检测ORB特征点，并计算描述符
    # (kps, des) = descriptor.detectAndCompute(image, None)

    return kps, des  # 返回特征点集，及对应的描述特征