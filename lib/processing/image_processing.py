import cv2
import math
import cv2 as cv
import numpy as np
import PIL.Image as Image
import io
import matplotlib
import matplotlib.pyplot as plt

from ..utils import is_number, vvd_round
from ..utils import is_path_obj
from ..utils import current_system
from ..utils import is_integer
from ..utils import OS_exists
from ..utils import time_reduce
from ..utils import glob_recursively
from ..utils import glob_images
from ..utils import has_chinese_char
from ..utils import vvd_floor, vvd_ceil
from ..utils import change_file_name_for_path
from ..utils import encode_path
from ..utils import cal_distance
from ..utils import remove_file
from ..utils import path_insert_content

from tqdm import tqdm
from pathlib import Path
from pathlib2 import Path as Path2

from numpy.lib.function_base import iterable
from matplotlib.backends.backend_agg import FigureCanvasAgg
from .array_processing import get_Several_MinMax_Array

popular_image_suffixes = ['png', 'jpg', 'jpeg', 'bmp']


def get_image_size(file_path, fallback_on_error=True):
    """
    Return (width, height) for a given img file
    """
    try:
        with Image.open(file_path) as img:
            width, height = img.size
    except Exception as e:
        if fallback_on_error:
            img = cv2.imread(file_path)
            width = img.shape[1]
            height = img.shape[0]
        else:
            raise e

    return (width, height)

def erode(mat, iterations=1, kernel_size=3, kernel=None, bool=True, float=False):
    """ erode 2D binary matrix by one pixel """
    assert isinstance(mat, np.ndarray)
    if kernel is None:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
    else:
        kernel = np.array(kernel, np.uint8)
    if float:
        mat = mat.astype(np.float32)
    else:
        mat = mat.astype(np.uint8)
    mat_eroded = cv.erode(mat, kernel, iterations=iterations)
    if bool:
        return mat_eroded > 0
    return mat_eroded

def dilate(mat, iterations=1, kernel_size=3, kernel=None, bool=True, float=False):
    """ dilate 2D binary matrix by one pixel """
    assert isinstance(mat, np.ndarray)
    if kernel is None:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
    else:
        kernel = np.array(kernel, np.uint8)
    if float:
        mat = mat.astype(np.float32)
    else:
        mat = mat.astype(np.uint8)
    mat_dilated = cv.dilate(mat, kernel, iterations=iterations)
    if bool:
        return mat_dilated > 0
    return mat_dilated

def open(mat, iterations=1, kernel_size=3, kernel=None, bool=True, float=False):
    """ dilate 2D binary matrix by one pixel """
    assert isinstance(mat, np.ndarray)
    if kernel is None:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
    else:
        kernel = np.array(kernel, np.uint8)
    if float:
        mat = mat.astype(np.float32)
    else:
        mat = mat.astype(np.uint8)

    mat_dilated = cv.erode(mat, kernel, iterations=iterations)
    mat_dilated = cv.dilate(mat_dilated, kernel, iterations=iterations)

    if bool:
        return mat_dilated > 0
    return mat_dilated

def close(mat, iterations=1, kernel_size=3, kernel=None, bool=True, float=False):
    """ close 2D binary matrix by one pixel """
    assert isinstance(mat, np.ndarray)
    if kernel is None:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
    else:
        kernel = np.array(kernel, np.uint8)
    if float:
        mat = mat.astype(np.float32)
    else:
        mat = mat.astype(np.uint8)

    mat_dilated = cv.dilate(mat, kernel, iterations=iterations)
    mat_dilated = cv.erode(mat_dilated, kernel, iterations=iterations)
    
    if bool:
        return mat_dilated > 0
    return mat_dilated

def get_bool_gravity_center(mask):
    num = np.sum(np.abs(mask))
    gra_cen_x = np.sum(np.abs(mask) * np.arange(mask.shape[1]) / max(1, num))
    gra_cen_y = np.sum(np.abs(mask).T * np.arange(mask.shape[0]) / max(1, num))
    return gra_cen_x, gra_cen_y

def get_grey_level_gravity_center(image, mask=None):
    image = image.astype('uint8')
    if mask:
        img = image[mask!=0]
    else:
        img = image
    value, _ = np.histogram(img, bins=256, range=[0,256])
    image_gravity_center = np.sum(value * np.arange(256)) / max(np.sum(value), 1)
    return image_gravity_center

def gravity_center_xy(image):
    bool_mask = to_gray_image(image) !=0

    if not bool_mask.any():
        return None, None

    H, W = bool_mask.shape[:2]

    X_array = np.arange(W)
    Y_array = np.arange(H)

    C_X = np.mean((bool_mask * X_array)[bool_mask])
    C_Y = np.mean((bool_mask.T * Y_array)[bool_mask.T])

    return C_X, C_Y

def contour_center(mask):
    contours, hierarchy = cv2.findContours((mask != 0).astype('uint8'), 2, 1)
    points = np.vstack(contours).squeeze()
    center_x, center_y = np.mean(points, axis=0)
    return center_x, center_y

def ring_degree(mask):
    center_x, center_y = contour_center(mask)
    Ys, Xs = np.nonzero(mask)
    dis = ((Ys - center_y) ** 2 + (Xs - center_x) ** 2) ** 0.5
    max_radius = np.percentile(dis, 98)
    min_radius = np.percentile(dis, 2)
    ring_area = np.pi * (max_radius ** 2 - min_radius ** 2)
    degree = len(Ys) / max(1, ring_area)
    return degree

def show_hist(img):
    rows, cols = img.shape
    hist = img.reshape(rows * cols)
    histogram, bins, patch = plt.hist(hist, 256, facecolor="green", histtype="bar")  # histogram即为统计出的灰度值分布
    plt.xlabel("gray level")
    plt.ylabel("number of pixels")
    plt.axis([0, 255, 0, np.max(histogram)])
    plt.show()
    return histogram


def get_center_bbox(input_image, width, height):
    assert width > 0 and height > 0
    H, W = input_image.shape[:2]
    center_x, center_y = W / 2, H / 2
    x_min = vvd_round(center_x - width / 2)
    x_max = vvd_round(x_min + width)
    y_min = vvd_round(center_y - height / 2)
    y_max = vvd_round(y_min + height)
    crop_box = [x_min, y_min, x_max, y_max]
    return crop_box


def center_crop(input_image, width, height):
    crop_box = get_center_bbox(input_image, width, height)
    return crop_data_around_boxes(input_image, crop_box)


def image_resize(img_source, shape=None, factor=None, unique_check=False, interpolation=None, uint8=True):
    if str(img_source.dtype).lower() != 'uint8' and uint8:
        print(f"Waring from mtutils image_resize: input source type {str(img_source.dtype)} is not uint8, we've made a transfer automatically.")
        img_source = img_source.astype('uint8')
    image_H, image_W = img_source.shape[:2]
    if shape is not None:
        if is_number(shape):
            shape = [shape] * 2
        shape = vvd_round(shape)

        resized_image = cv2.resize(img_source, tuple(shape), interpolation=interpolation)

    elif factor is not None:
        if iterable(factor):
            assert len(factor) == 2
            factor_x, factor_y = factor
        else:
            factor_x, factor_y = factor, factor

        resized_H = int(round(image_H * factor_y))
        resized_W = int(round(image_W * factor_x))

        resized_image = cv2.resize(img_source, (resized_W, resized_H), interpolation=interpolation)

    elif shape is None and factor is None:
        resized_image = img_source
    else:
        raise RuntimeError

    if unique_check:
        pixel_list = np.unique(img_source).tolist()
        if len(pixel_list) == 2 and 0 in pixel_list:
            resized_image[resized_image > 0] = np.max(pixel_list)

    return resized_image

def image_merge(*image_weight_pair_list, gamma=0):
    """
    :param image_weight_pair: [(image, weight), ...]
    :param gamma: value offset
    wi' = wi / sum(wi)
    I = I1*w1' + I2*w2' + ... + gamma
    :return: uint8 image
    """
    if len(image_weight_pair_list) == 0:
        raise RuntimeError("image_weight_pair_list is empty")
    
    shape_temp = None
    total_weight = 0
    for image, weight in image_weight_pair_list:
        if weight <= 0:
            raise RuntimeError("weight is not positive")
        else:
            total_weight += weight
        if shape_temp is None:
            shape_temp = image.shape
        else:
            assert shape_temp == image.shape, "image shape is not same"
    
    puzzle = np.zeros(shape_temp, dtype='float32')
    for image, weight in image_weight_pair_list:
        weight_temp = weight / total_weight
        puzzle += weight_temp * image
    
    if gamma != 0:
        puzzle += gamma
    
    puzzle = np.clip(puzzle, 0, 255)
    return puzzle.astype('uint8')


def to_gray_image(img):
    if isinstance(img, Path) or isinstance(img, Path2):
        if img.exists():
            gray_img = cv_rgb_imread(img, gray=True)
            return gray_img
        else:
            raise FileNotFoundError(f"file not found {str(img)}")
    elif isinstance(img, str):
        if OS_exists(img):
            gray_img = cv_rgb_imread(img, gray=True)
            return gray_img
        else:
            raise FileNotFoundError(f"file not found {str(img)}")
    elif isinstance(img, np.ndarray):
        if img.ndim == 2:
            return img.copy()
        elif img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            raise RuntimeError(f"ndim of img is not 2 or 3 {img.shape}")
    else:
        raise RuntimeError(f"unknown data type {type(img)}")

def percentile_normalize(image, low_value, low_p, high_value, high_p, except_zero=True):
    assert low_value < high_value
    assert 0 <= low_p < high_p <= 100

    if str(image.dtype) == 'uint8':
        if except_zero:
            min_value = np.percentile(image[image!=0], low_p)
            max_value = np.percentile(image[image!=0], high_p)
        else:
            min_value = np.percentile(image, low_p)
            max_value = np.percentile(image, high_p)

        if max_value <= min_value:
            return image

        a = (high_value - low_value) / (max_value - min_value)
        b = low_value - a * min_value

        normed_image = np.clip(image.astype('float32') * a + b, 0, 255).astype('uint8')
        return normed_image
    else:
        print(f"percentile_normalize neet uint8 image as input img.")
        return image

def gaussian_mask_2d(size, sigma):
    gaussian_mask_1d = cv2.getGaussianKernel(size, sigma)
    gaussian_mask_2d = gaussian_mask_1d * gaussian_mask_1d.T
    normalized_gaussian_mask_2d = min_max_normalize(gaussian_mask_2d)
    return normalized_gaussian_mask_2d

def min_max_normalize(image, min=0, max=1, min_p=0, max_p=100):
    assert 0 <= min_p < max_p <= 100, f"0 <= min_p {min_p} < max_p {max_p}<= 100 should be true." 
    assert min < max, f"min {min} < max {max} should be true."

    min_value = np.percentile(image, min_p)
    max_value = np.percentile(image, max_p)

    if max_value == min_value:
        res_image = image - max_value + min
    else:
        assert min_value < max_value
        clip_image = np.clip(image, min_value, max_value)
        res_image = (clip_image - min_value) / (max_value - min_value) * (max - min) + min
    return res_image


def median_transfer(image, gray=False, target_median=120):
    if gray:
        image = to_gray_image(image)
    median = max(1, np.median(image))
    mapped_image = (np.clip(image * (target_median / median), 0, 255)).astype('uint8')
    return mapped_image


def to_colorful_image(image):
    """
    make a gray image to an image with 3 channels
    """
    if image.ndim == 2:
        if str(image.dtype) == 'bool':
            image = image.astype('uint8') * 255
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image


def img_normalize(img, mean, std, to_rgb=False):
    """Inplace normalize an image with mean and std.

    Args:
        img (ndarray): Image to be normalized.
        mean (ndarray): The mean to be used for normalize.
        std (ndarray): The std to be used for normalize.
        to_rgb (bool): Whether to convert to rgb.

    Returns:
        ndarray: The normalized image.
    """
    # cv2 inplace normalization does not accept uint8
    img = img.astype('float32')

    mean = np.array(mean, dtype='float32')
    std = np.array(std, dtype='float32')

    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace
    return img


def gamma_transform(image, theta):
    # gamma transform
    data = np.arange(256)
    data = data / 255
    data = data ** theta
    data = (data * 255).astype('uint8')
    
    gamma_image = data[image]
    return gamma_image

def paint_words(img, display_str, left=0, top=0, fontsize=20, color=tuple([0, 0, 255]), font_path=None):
    from PIL import ImageFont, ImageDraw, Image
    import matplotlib.font_manager as fm
    try:
        if font_path is None:
            if current_system() == 'Windows':
                font = ImageFont.truetype('arial.ttf', fontsize)
            else:
                font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')),fontsize)
        else:
            font = ImageFont.truetype(font_path, fontsize)
    except Exception as e:
        print(f"Failed to load font: {e}, fallback to default font.")
        font = ImageFont.load_default()

    img = to_colorful_image(img.copy())
    pil_image = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_image)

    draw.text((left, top), str(display_str), fill=tuple(color), font=font)
    res_img = np.asarray(pil_image)
    return res_img

def img_rotate(img,
             angle,
             center=None,
             interpolation=cv2.INTER_LINEAR,
             border_mode=cv2.BORDER_CONSTANT,
             border_value=0,
             auto_bound=False,
             ):
    """Rotate an image.

    Args:
        img (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees, positive values mean
            clockwise rotation.
        center (tuple[float], optional): Center point (w, h) of the rotation in
            the source image. If not specified, the center of the image will be
            used.
        border_value (int): Border value.
        auto_bound (bool): Whether to adjust the image size to cover the whole
            rotated image.

    Returns:
        ndarray: The rotated image.
    """
    type = img.dtype
    img = img.astype('uint8')
    scale=1.0
    if center is not None and auto_bound:
        raise ValueError('`auto_bound` conflicts with `center`')
    h, w = img.shape[:2]
    if center is None:
        center = ((w - 1) * 0.5, (h - 1) * 0.5)
    assert isinstance(center, tuple)

    matrix = cv2.getRotationMatrix2D(center, -angle, scale)
    if auto_bound:
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        new_w = h * sin + w * cos
        new_h = h * cos + w * sin
        matrix[0, 2] += (new_w - w) * 0.5
        matrix[1, 2] += (new_h - h) * 0.5
        w = int(np.round(new_w))
        h = int(np.round(new_h))
    rotated = cv2.warpAffine(
        img,
        matrix, (w, h),
        flags=interpolation,
        borderValue=border_value,
        borderMode=border_mode)
    rotated.astype(type)
    return rotated

image_rotate = img_rotate

def image_center_rotate(image, angle, center=None, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, border_value=0):
    type = image.dtype
    image = image.astype('uint8')
    h, w = image.shape[:2]

    if center is None:
        center = ((w - 1) * 0.5, (h - 1) * 0.5)
    assert isinstance(center, tuple)

    matrix = cv2.getRotationMatrix2D((0, 0), -angle, 1)

    corner_points = np.array([
                        [0, 0],
                        [w, 0],
                        [w, h],
                        [0, h]
                        ] + [[center[0], center[1]]])

    new_corner_points = corner_points @ matrix.T[:2, :]

    x_min = np.min(new_corner_points[:, 0])
    y_min = np.min(new_corner_points[:, 1])
    x_max = np.max(new_corner_points[:, 0])
    y_max = np.max(new_corner_points[:, 1])

    matrix[0, 2] -= x_min
    matrix[1, 2] -= y_min

    new_w = np.ceil(x_max - x_min).astype('int32')
    new_h = np.ceil(y_max - y_min).astype('int32')

    new_corner_points -= [x_min, y_min]

    rotated = cv2.warpAffine(
        image,
        matrix, (new_w, new_h),
        flags=interpolation,
        borderValue=border_value,
        borderMode=border_mode)

    rotated.astype(type)
    center_point = new_corner_points[-1, :]

    return rotated, center_point

def image_flip_lr(image, contiguousarray=True):
    fliped = np.fliplr(image)
    if contiguousarray:
        fliped = np.ascontiguousarray(fliped)
    return fliped


def image_flip_ud(image, contiguousarray=True):
    fliped = np.flipud(image)
    if contiguousarray:
        fliped = np.ascontiguousarray(fliped)
    return fliped


def image_rotate_90(image, contiguousarray=True):
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h/2), 90, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    if contiguousarray:
        rotated = np.ascontiguousarray(rotated)
    return rotated

def image_rotate_180(image, contiguousarray=True):
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h/2), 180, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    if contiguousarray:
        rotated = np.ascontiguousarray(rotated)
    return rotated

def image_rotate_270(image, contiguousarray=True):
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h/2), 270, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    if contiguousarray:
        rotated = np.ascontiguousarray(rotated)
    return rotated


def cv_rgb_imread(image_path, gray=False):
    """
    按照RGB顺序使用cv读取图像
    """
    image_path = str(image_path)
    image = image_read(image_path)
    if gray:
        if image.ndim > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        b, g, r = cv.split(image)
        image = cv.merge([r, g, b])

    return image


def cv_rgb_bgr_convert(image):
    """[convert rgb to bgr or bgr ro rgb]

    Args:
        image ([np.array(uint8)]): [uint8 image]

    Returns:
        [image]: [r and b swapped]
    """
    b, g, r = cv.split(image.astype('uint8'))
    image = cv.merge([r, g, b])

    return image


def image_show(image, window_name='image show'):
    '''
    更加鲁棒地显示图像包括二维图像,第三维度为1的图像
    '''
    temp_image = extend_image_channel(image)
    cv_image_show(image=temp_image, window_name=window_name)


def image_read(image_path, channel=3):
    """
    读取图像, 可包含中文路径
    Args:
        image_path ([str]): [图像路径]
        channel (int, optional): [图像通道数, -1为默认, 0为灰度]. Defaults to -1.
    """
    image_path = str(image_path)
    return cv.imdecode(np.fromfile(image_path, dtype=np.uint8), channel)


def cv_image_show(image, window_name='image show'):
    '''
    show image (for debug)
    press anykey to destory the window 

    image: image in numpy 
    window_name: name of the window

    image color - bgr
    '''
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.imshow(window_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def extend_image_channel(input_image):
    '''
    cv显示三通道图像, 本函数将原始图像扩展到三通道
    '''
    image = input_image.copy()

    shape = image.shape

    max_value = np.max(image)
    if not is_integer(max_value) and max_value > 1:
        image /= np.max(image)

    if 0 < np.max(image) <= 1:
        image = (255*image).astype('uint8')


    if len(shape) == 3:
        if shape[2] == 3:
            return image
        elif shape[2] == 1:
            temp_image = np.zeros([shape[0], shape[1], 3])
            for i in range(3):
                temp_image[:, :, i] = image[:, :, 0]
            return temp_image
        else:
            raise TypeError('image type error')
    elif len(shape) == 2:
        temp_image = np.zeros([shape[0], shape[1], 3], dtype=type(image[0][0]))
        for i in range(3):
            temp_image[:, :, i] = image
        return temp_image
    else:
        raise TypeError('image type error')

def cv_bgr_imwrite(bgr_image, image_save_path, para=None, with_suffix=None, factor=1):
    rgb_image = cv_rgb_bgr_convert(bgr_image)
    cv_rgb_imwrite(rgb_image, image_save_path, para, with_suffix, factor)

def cv_rgb_imwrite(rgb_image, image_save_path, para=None, with_suffix=None, factor=1):
    """
    [cv2 save a rgb image]
    Args:
        rgb_image ([np.array]): [rgb image]
        image_save_path ([str/Path]): [image save path]
    """
    image = rgb_image
    if rgb_image.ndim == 3:
        if rgb_image.shape[2] == 3:
            bgr_image = cv_rgb_bgr_convert(rgb_image)
            image = bgr_image

    image_save_path = Path(image_save_path)
    image_save_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = image_save_path.suffix.lower()
    quality_para = None
    if para is not None:
        if suffix == '.jpg' or suffix == '.jpeg':
            # para in 0-100, the bigger the image quality higher and the file size larger
            quality_para = [cv2.IMWRITE_JPEG_QUALITY, para]
        elif suffix == '.png':
            # para in 0-9, the bigger the image file size smaller
            quality_para = [cv2.IMWRITE_PNG_COMPRESSION, para]
    if with_suffix is not None:
        image_save_path = Path(image_save_path).with_suffix('.' + with_suffix)
    
    if factor != 1:
        image = image_resize(image, factor=factor)

    save_path = str(image_save_path)
    if has_chinese_char(save_path):
        cv2.imencode(suffix, image, quality_para)[1].tofile(save_path)
    else:
        cv.imwrite(save_path, image, quality_para)

def jpeg_noise(image, para):
    encoded_img = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, para])[1]
    decoded_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    return decoded_img

def pil_rgb_imwrite(rgb_image, image_save_path):
    """
    [pil save a rgb image]
    Args:
        rgb_image ([np.array]): [rgb image]
        image_save_path ([str/Path]): [image save path]
    """
    from PIL.ImageFile import ImageFile
    if isinstance(rgb_image, ImageFile):
        pil_image = rgb_image
    elif isinstance(rgb_image, np.ndarray):
        pil_image = Image.fromarray(rgb_image)
    else:
        raise NotImplementedError(f"unknown data type {rgb_image}")

    image_save_path = Path(image_save_path)
    image_save_path.parent.mkdir(parents=True, exist_ok=True)
    pil_image.save(str(image_save_path))


def image_show_from_path(file_path):
    """[show image from image file path]

    Args:
        file_path ([str or Path]): [path of image file]
    """
    assert is_path_obj(file_path) or isinstance(file_path, str)
    file_path = str(file_path)
    if not OS_exists(file_path):
        print('file: ', file_path, 'does not exist.')
    else:
        image = cv_rgb_imread(file_path)
        plt_image_show(image)


def plt_image_show(*image, window_name='', array_res=False, full_screen=True, cmap=None, position=[30, 30], share_xy=False, axis_off=False, col_num=None, row_num=None, norm_float=True, bgr=False):
    '''
    更加鲁棒地显示图像包括二维图像,第三维度为1的图像
    '''
    image_list = list(image)
    if len(image_list) == 0:
        return
    # temp_image = extend_image_channel(image)
    image_num = len(image_list)
    if col_num is None and row_num is None:
        col_num = int(np.ceil(image_num**0.5))
        row_num = int(np.ceil(image_num/col_num))
    elif row_num is not None:
        row_num = vvd_round(row_num)
        assert row_num > 0
        col_num = int(np.ceil(image_num/row_num))
    else:
        col_num = vvd_round(col_num)
        assert col_num > 0
        row_num = int(np.ceil(image_num/col_num))

    image_list = image_list + [None] * (col_num * row_num - image_num)

    assert len(image_list) == col_num * row_num

    if full_screen:
        if current_system() == 'Windows':
            figsize=(18.5, 9.4)
        else:
            figsize=(18.5, 9.4)

    fig, ax = plt.subplots(row_num, col_num, figsize=figsize, sharex=share_xy, sharey=share_xy)

    backend = matplotlib.get_backend()

    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.2, wspace=0.05)

    for index, image_item in enumerate(image_list):
        if image_item is None:
            continue
        print_name = window_name
        if isinstance(image_item, tuple) or isinstance(image_item, list):
            if isinstance(image_item[1], str):
                current_name = image_item[1]
                print_name = current_name
                cur_image = image_item[0]
                image = np.array(cur_image)
            else:
                image = np.array(image_item)
        else:
            image = np.array(image_item)

        if image.ndim == 3:
            if image.shape[0] == 3 and image.shape[-1] != 3:
                # guess it is a tensor image
                image = np.transpose(image, [1, 2 ,0])
            if image.shape[-1] == 3 and bgr:
                image = cv_rgb_bgr_convert(image)

        if iterable(ax):
            if ax.ndim == 1:
                cur_ax = ax[index]
            elif ax.ndim == 2:
                row_index = index // col_num
                col_index = index % col_num
                cur_ax = ax[row_index][col_index]
            else:
                raise RuntimeError(f'bad ax ndim num {ax}')
        else:
            cur_ax = ax

        if axis_off:
            cur_ax.axis('off')

        if image is None:
            cur_ax.axis('off')
            continue

        elif image.ndim == 1:
            cur_ax.plot(image)

        else:
            if 'uint8' == image.dtype.__str__():
                cur_ax.imshow(image, cmap=cmap, vmax=np.max(image), vmin=np.min(image))
            elif 'int' in image.dtype.__str__():
                cur_ax.imshow(image, cmap=cmap, vmax=np.max(image), vmin=np.min(image))
            elif 'bool' in image.dtype.__str__():
                cur_ax.imshow(image.astype('float32'), cmap=cmap, vmax=np.max(image), vmin=np.min(image))
            elif 'float' in image.dtype.__str__():
                image = np.squeeze(image)
                if norm_float:
                    cur_ax.imshow((image - np.min(image)) / (max(1, np.max(image)) - np.min(image)), cmap=cmap)
                else:
                    cur_ax.imshow(image, cmap=cmap)
            else:
                cur_ax.imshow(image.astype('uint8'), cmap=cmap, vmax=np.max(image), vmin=np.min(image))

        cur_ax.margins(0, 0)
        cur_ax.set_title(print_name)

    if not array_res:
        try:
            mngr = plt.get_current_fig_manager()
            mngr.window.wm_geometry(f"+{position[0]}+{position[1]}")
        except Exception:
            pass
        plt.show()
    else:
        image = convert_plt_to_rgb_image(plt)
        plt.close()
        return image

PIS = plt_image_show

def scatter3d(xyz_data_list, label_list, xyz_label=None, title='3d Scatter plot'):
    if not xyz_data_list:
        return
    xyz_data = np.array(xyz_data_list)
    assert xyz_data.ndim == 2
    assert xyz_data.shape[1] == 3

    label_set = set(label_list)
    label_2_color_dict = dict()
    label_data_dict = dict()
    for index, label in enumerate(label_set):
        color = [np.random.rand(), np.random.rand(), np.random.rand()]
        label_2_color_dict[label] = color
        info = dict(
            x=[],
            y=[],
            z=[]
        )
        label_data_dict[label] = info

    for data, label in zip(xyz_data_list, label_list):
        x, y, z = data
        label_data_dict[label]['x'].append(x)
        label_data_dict[label]['y'].append(y)
        label_data_dict[label]['z'].append(z)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for label, data_set in label_data_dict.items():
        ax.scatter3D(np.array(data_set['x']), np.array(data_set['y']), np.array(data_set['z']), c=[label_2_color_dict[label]] * len(data_set['x']), label=label)

    ax.set_title(title)
    try:
        ax.set_xlabel(xyz_label[0])
        ax.set_ylabel(xyz_label[1])
        ax.set_zlabel(xyz_label[2])
    except:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    plt.legend()
    plt.show()
    pass


def scatter2d(xy_data_list, label_list=None, xy_label=None, title='2d Scatter plot'):
    if len(xy_data_list) == 0:
        return
    xyz_data = np.array(xy_data_list)
    assert xyz_data.ndim == 2
    assert xyz_data.shape[1] == 2
    
    if label_list is None:
        label_list = [0] * len(xy_data_list)

    label_set = set(label_list)
    label_2_color_dict = dict()
    label_data_dict = dict()
    for index, label in enumerate(label_set):
        color = [np.random.rand(), np.random.rand(), np.random.rand()]
        label_2_color_dict[label] = color
        info = dict(
            x=[],
            y=[]
        )
        label_data_dict[label] = info

    for data, label in zip(xy_data_list, label_list):
        x, y = data
        label_data_dict[label]['x'].append(x)
        label_data_dict[label]['y'].append(y)

    fig = plt.figure()
    ax = plt.axes()
    for label, data_set in label_data_dict.items():
        ax.scatter(np.array(data_set['x']), np.array(data_set['y']), c=[label_2_color_dict[label]] * len(data_set['x']), label=label)

    ax.set_title(title)
    try:
        ax.set_xlabel(xy_label[0])
        ax.set_ylabel(xy_label[1])
    except:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    plt.legend()
    plt.show()

def make_histgram(data_array_list, min_v=None, max_v=None, bin_num=None, gap=None, style_str_list=None, array_res=False):
    color_list=['', 'y', 'g', 'r', 'c', 'm', 'b', 'k', 'w']
    linestyle_list=['', '-', '--', '-.', ':']

    if type(data_array_list) == np.ndarray:
        data_array_list = list(data_array_list)

    hist_list = list()
    for index, sub_data in enumerate(data_array_list):
        data = np.array(sub_data).flatten()
        data_num = len(data)
        assert data_num > 0
        span = data.max() - data.min()
        if min_v is None:
            min_v = data.min() - span*0.1
        if max_v is None:
            max_v = data.max() + span*0.1
        assert max_v >= min_v
        
        if bin_num is None:
            if gap is not None:
                assert gap > 0
                bin_num = max(1, int((max_v - min_v) / gap))
            else:
                bin_num = max(int(data_num ** 0.75), 1)
        
        assert bin_num > 0
        
        hist, x_list = np.histogram(data, bin_num, [min_v, max_v])
        hist = hist / data_num
        color = color_list[index % len(color_list)]
        linestyle = linestyle_list[index // len(color_list) % len(linestyle_list)]
        style_real_str = color + linestyle
        
        if style_str_list is not None:
            style_real_str = style_str_list[index % len(style_str_list)]
        
        hist_list.append([x_list[:-1], hist, style_real_str])
    for hist in hist_list:
        plt.plot(*hist)

    if not array_res:
        plt.show()
    else:
        image = convert_plt_to_rgb_image(plt)
        plt.close()
        return image
    pass


def convert_plt_to_rgb_image(plt):
    # Convert a Matplotlib figure to a 3D numpy array with RGB channels and return it
    canvas = FigureCanvasAgg(plt.gcf())
    canvas.draw()
    w, h = canvas.get_width_height()
    buf = np.fromstring(canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    image = np.asarray(image)
    rgb_image = image[:, :, :3]
    return rgb_image


def image_format_transfer(origin_dir, tar_dir, origin_suffix, tar_suffix, recursively=True, keep_struct=True):

    if origin_suffix.lower() not in popular_image_suffixes:
        raise Warning(f'origin_suffix {origin_suffix} is not an usually image file suffix')

    if tar_suffix.lower() not in popular_image_suffixes:
        raise Warning(f'tar_suffix {tar_suffix} is not an usually image file suffix')

    origin_path = Path(origin_dir)
    target_path = Path(tar_dir)
    assert origin_path.is_dir(), f"origin_dir {origin_dir} does not exist"
    if not target_path.is_dir():
        target_path.mkdir()

    image_path_list = glob_recursively(origin_dir, origin_suffix, recursively=recursively)

    for image_path in tqdm(image_path_list, desc=f"converting suffix from {origin_suffix} to {tar_suffix}"):
        img = Image.open(image_path)
        if keep_struct:
            rel_file_path = Path(image_path).relative_to(origin_path)
            new_file_path = change_file_name_for_path(target_path / rel_file_path, new_suffix=tar_suffix)
        else:
            file_name = encode_path(image_path, with_suffix=tar_suffix)
            new_file_path = str(target_path / file_name)
        pil_rgb_imwrite(img, new_file_path)


def vvd_image_preprocess(image):
    """
    vvd 图像预处理
    """
    new_image = image / 127.5 - 1
    return new_image


def crop_data_around_boxes(image, crop_box, cut_box_back=False):
    """make a image crop from a image safely"""

    ndim = image.ndim
    data_type = image.dtype
    height, width = image.shape[:2]

    crop_box = np.array(crop_box).astype('int32').tolist()

    ori_left, ori_top, ori_right, ori_bottom = 0, 0, width, height

    crop_left, crop_top, crop_right, crop_bottom = crop_box

    assert crop_right > crop_left and crop_bottom > crop_top

    crop_width = crop_right - crop_left
    crop_height = crop_bottom - crop_top

    cut_left = max(crop_left, ori_left)
    cut_right = max(min(ori_right, crop_right), cut_left)
    cut_top = max(ori_top, crop_top)
    cut_bottom = max(min(ori_bottom, crop_bottom), cut_top)

    cut_box = [cut_left, cut_top, cut_right, cut_bottom]

    crop_ori = image[cut_top:cut_bottom, cut_left:cut_right, ...]

    if cut_right - cut_left != crop_width or cut_bottom - cut_top != crop_height:

        # out of boundary
        if ndim == 3:
            crop_ori_temp = np.zeros([crop_height, crop_width, image.shape[2]], dtype=data_type)
        elif ndim == 2:
            crop_ori_temp = np.zeros([crop_height, crop_width], dtype=data_type)
        else:
            raise RuntimeError(f"error image shape {image.shape} ndim {ndim}")

        win_left = cut_left - crop_left
        win_right = max(cut_right - crop_left, win_left)
        win_top = cut_top - crop_top
        win_bottom = max(cut_bottom - crop_top, win_top)

        crop_ori_temp[win_top:win_bottom, win_left:win_right, ...] = crop_ori
        crop_ori = crop_ori_temp

    if cut_box_back:
        return crop_ori, cut_box
    else:
        return crop_ori


def zero_padding(in_array, padding_size_1, padding_size_2, padding_size_3=None, padding_size_4=None):
    """
    四周补零, 以此避免边界判断(仅用于三通道图像)

    输入: 
    :in_array: 输入矩阵 np.array (rows, cols, 3)

    (padding_size_3-4 为 None 时)
    :padding_size_1:  上下补零行数
    :padding_size_2:  左右补零列数

    (padding_size_3-4 均不为 None 时)
    :padding_size_1:  上补零行数
    :padding_size_2:  下补零行数
    :padding_size_3:  左补零列数
    :padding_size_4:  右补零列数

    输出: 
    :padded_array: 补零后的图像（新建矩阵, 不修改原始输入）
    """

    assert np.ndim(in_array) == 3 or np.ndim(in_array) == 2

    if np.ndim(in_array) == 3:
        rows, cols, ndim = in_array.shape
    else:
        rows, cols = in_array.shape

    if (padding_size_3 is None) and (padding_size_4 is None):
        padding_size_1 = max(padding_size_1, 0)
        padding_size_2 = max(padding_size_2, 0)
        assert padding_size_1 >= 0 and padding_size_2 >= 0
        if np.ndim(in_array) == 3:
            padded_array = np.zeros([rows + 2 * padding_size_1, cols + 2 * padding_size_2, ndim], dtype=type(in_array[0][0][0]))
            padded_array[padding_size_1:rows + padding_size_1, padding_size_2:cols + padding_size_2, :] = in_array
        elif np.ndim(in_array) == 2:
            padded_array = np.zeros([rows + 2 * padding_size_1, cols + 2 * padding_size_2], dtype=type(in_array[0][0]))
            padded_array[padding_size_1:rows + padding_size_1, padding_size_2:cols + padding_size_2] = in_array
        else:
            raise ValueError("np.ndim error")

    else:
        assert (padding_size_3 is not None) and (padding_size_4 is not None), "padding_size_3 padding_size_4 必须都不是none"
        padding_size_1 = max(padding_size_1, 0)
        padding_size_2 = max(padding_size_2, 0)
        padding_size_3 = max(padding_size_3, 0)
        padding_size_4 = max(padding_size_4, 0)
        assert padding_size_1 >= 0 and padding_size_2 >= 0 and padding_size_3 >= 0 and padding_size_4 >= 0
        if np.ndim(in_array) == 3:
            padded_array = np.zeros([rows + padding_size_1 + padding_size_2, cols + padding_size_3 + padding_size_4, ndim], dtype=type(in_array[0][0][0]))
            padded_array[padding_size_1:rows + padding_size_1, padding_size_3:cols + padding_size_3, :] = in_array
        elif np.ndim(in_array) == 2:
            padded_array = np.zeros([rows + padding_size_1 + padding_size_2, cols + padding_size_3 + padding_size_4], dtype=type(in_array[0][0]))
            padded_array[padding_size_1:rows + padding_size_1, padding_size_3:cols + padding_size_3] = in_array
        else:
            raise ValueError("np.ndim error")

    return padded_array


def image_border_move(img, left, top=None, right=None, bottom=None, border_type='constant', value=0):
    """
    adjust image border reltively
    Args:
        img (np.array): image to be processed
        left (int): left value
        top (int): top value
        right (int): right value
        bottom (int): bottom value
        border_type (str, optional): type for additional border, should be one of ['constant', 'replicate', 'reflect']. Defaults to 'constant'.
        value (int, optional): if constant border is set, it means the padding pixel value. Defaults to 0.
    """
    if top is None:
        top = left
    if right is None:
        right = top
    if bottom is None:
        bottom = right

    assert border_type in ['constant', 'replicate', 'reflect']
    left, top, right, bottom = vvd_round([left, top, right, bottom])
    ori_H, ori_W = img.shape[:2]
    assert -left < ori_W + right
    assert -top < bottom + ori_H

    move_data = np.array([left, top, right, bottom])

    # padding
    padding_data = np.clip(move_data, 0, None)
    if padding_data.max() > 0:
        assert len(padding_data) == 4
        if border_type == 'constant':
            border_type_index = cv2.BORDER_CONSTANT
        elif border_type == 'replicate':
            border_type_index = cv2.BORDER_REPLICATE
        elif border_type == 'reflect':
            border_type_index = cv2.BORDER_REFLECT_101
        else:
            raise RuntimeError(f"unknown border type {border_type}.")
        padded_img = cv2.copyMakeBorder(img, padding_data[1], padding_data[3], padding_data[0], padding_data[2], borderType=border_type_index, value=[value] * 4)
    else:
        padded_img = img

    # croping
    H, W = padded_img.shape[:2]
    crop_data = np.clip(move_data, None, 0)
    if crop_data.min() < 0:
        bbox = [0 - crop_data[0], 0 - crop_data[1], W + crop_data[2], H + crop_data[3]]
        crop_img = crop_data_around_boxes(padded_img, bbox)
    else:
        crop_img = padded_img

    return crop_img


def polar_move(polar_image, source_center_phase, target_center_phase):
    """[height of polar_image is the origin circle side]

    Args:
        polar_image ([np.array]): [polar image]
        source_center_phase ([float]): [source center phase]
        target_center_phase ([float]): [target center phase]
    """
    height, width = polar_image.shape[:2]
    center_index = vvd_round(source_center_phase % 360 / 360 * height)
    target_index = vvd_round(target_center_phase % 360 / 360 * height)

    new_polar_image = np.zeros_like(polar_image)

    movement = target_index - center_index

    new_polar_image[:movement] = polar_image[-movement:]
    new_polar_image[movement:] = polar_image[:-movement]

    return new_polar_image


def cycle_move(data: np.ndarray, offset: int, axis=0):
    assert axis < data.ndim
    index_list = list(range(data.ndim))
    index_list[axis] = 0
    index_list[0] = axis
    new_data = np.transpose(data, index_list)
    offset = offset % new_data.shape[0]
    if data.ndim > 1:
        concate_fun = np.vstack
    else:
        concate_fun = np.concatenate
    moved_data = concate_fun([new_data[offset:,...], new_data[:offset,...]])
    moved_data = np.transpose(moved_data, index_list)
    return moved_data


def crop_by_cycle_y_min_max(image, y_min, y_max):
    height = image.shape[0]

    if image.ndim > 1:
        concate_fun = np.vstack
    else:
        concate_fun = np.concatenate

    if y_min >= 0 and y_max <= height:
        if y_min <= y_max:
            crop_image = image[y_min:y_max, ...]
        else:
            crop_image = concate_fun((image[y_min:, ...], image[:y_max, ...]))

    elif y_min < 0:
        crop_image = concate_fun((image[y_min % height:, ...], image[:y_max, ...]))
    elif y_max > height:
        crop_image = concate_fun((image[y_min:, ...], image[:y_max % height, ...]))
    return crop_image


def crop_by_cycle_x_min_max(image, x_min, x_max):
    width = image.shape[1]

    if x_min >= 0 and x_max <= width:
        if x_min <= x_max:
            crop_image = image[:, x_min:x_max, ...]
        else:
            crop_image = np.hstack((image[:, x_min:, ...], image[:, :x_max, ...]))
    elif x_min < 0:
        crop_image = np.hstack((image[:, x_min % width:, ...], image[:, :x_max, ...]))
    elif x_max > width:
        crop_image = np.hstack((image[:, x_min:, ...], image[:, :x_max % width, ...]))
    return crop_image


def fill_sector(image, center, radius_out, radius_in, start_radian, end_radian, color=[255, 255, 255]):
    """[fill sector]

    Args:
        image ([np.array]): [input image]
        center ([list]): [center X Y]
        radius_out ([number]): [outside radius]
        radius_in ([number]): [inside radius]
        start_radian ([float]): [start radian]
        end_radian ([float]): [end radian]
        color (list, optional): [fill color]. Defaults to [255, 255, 255].

    Returns:
        [np.array]: [output image]
    """
    image = image.copy()
    mask = np.zeros_like(image)
    start_angle = start_radian / np.pi * 180
    end_angle = end_radian / np.pi * 180
    mask = cv2.ellipse(mask, vvd_round(center), vvd_round([radius_out, radius_out]), 0, start_angle, end_angle, color, -1)
    mask = cv2.ellipse(mask, vvd_round(center), vvd_round([radius_in, radius_in]), 0, start_angle, end_angle, [0] * 3, -1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones([3,3]))

    image[mask > 0] = mask[mask > 0]
    return image


def local_normalization(image, k_size=5):
    gauss_image = cv2.GaussianBlur(image, (k_size, k_size), 0)
    reduce_mean_image = image - gauss_image.astype('float')
    square_image = (reduce_mean_image ** 2)
    gauss_square_image = cv2.GaussianBlur(square_image, (k_size, k_size), 0)
    sigma_image = gauss_square_image ** 0.5
    res_image = reduce_mean_image / (sigma_image + 1e-6)

    return res_image


def rectangle2polygon(rec):
    ((x1, y1), (x2, y2)) = rec
    polygon = xyxy2polygon(x1, y1, x2, y2)
    return polygon


def bbox2polygon(bbox):
    if not bbox:
        return []
    x1, y1, x2, y2 = bbox
    polygon = xyxy2polygon(x1, y1, x2, y2)
    return polygon


def polygon2bbox(polygon):
    array = np.array(polygon)
    if array.ndim == 3 and array.shape[1] == 1:
        array = array[:, 0, :]
    return np.min(array, axis=0).tolist() + np.max(array, axis=0).tolist()

def ceil_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return [vvd_floor(x1), vvd_floor(y1), vvd_ceil(x2), vvd_ceil(y2)]

def floor_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return [vvd_ceil(x1), vvd_ceil(y1), vvd_floor(x2), vvd_floor(y2)]

def xyxy2polygon(x1, y1=None, x2=None, y2=None):
    if y1 is None and x2 is None and y2 is None:
        bbox = x1
        x1, y1, x2, y2 = bbox

    points = []
    points.append([x1, y1])
    points.append([x1, y2])
    points.append([x2, y2])
    points.append([x2, y1])
    return points


def polygon_interpolation(polygon, max_distance, close=True):
    assert max_distance > 0, f'distance {max_distance} should > 0'
    if len(polygon) < 2:
        return polygon
    if len(polygon) > 2 and close:
        polygon.append(polygon[0])

    points_res = list()
    for index in range(len(polygon)-1):
        cur_point = polygon[index]
        next_point = polygon[index+1]
        assert len(cur_point) == len(next_point) == 2, 'point length should be 2'
        interpolate_num = int(cal_distance(cur_point, next_point) // max_distance)
        points_res.extend(np.linspace(cur_point, next_point, 2+interpolate_num).tolist()[:-1])
    points_res.append(polygon[-1])
    return points_res


def get_polygons_from_mask(mask, approx=False, epsilon=None):
    """ get polygons for all isolated areas on fg-mask
        return something like: [[x1,y1], [x2,y2], [x3,y3], ...]
    """
    mask = to_gray_image(mask)
    contours, _  = cv2.findContours((mask).astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if approx:
        new_contours = list()

        for contour in contours:
            if epsilon is None:
                epsilon = 0.005 * cv2.arcLength(contour, True)
            new_contour = cv2.approxPolyDP(contour, epsilon, True)
            if new_contour.shape[0] > 2:
                new_contours.append(new_contour)
        contours = new_contours
    return contours


def get_shapes_from_mask(mask, epsilon=None, drop_small_polygon=False):
    contours = get_polygons_from_mask(mask)
    if epsilon is not None:
        new_contours = list()
        for contour in contours:
            new_contour = cv2.approxPolyDP(contour, epsilon, True)
            if new_contour.shape[0] > 2:
                new_contours.append(new_contour)
            else:
                if not drop_small_polygon:
                    new_contours.append(contour)
        contours = new_contours
    shapes = list()
    for contour in contours:
        shapes.append(contour[:, 0, :].tolist())
    return shapes


def draw_polygons(mask, polygons, color=None, thickness=5, fill=False, draw_split=False):
    mask = mask.copy()
    if color is None:
        color = [255, 255, 255]
    try:
        if np.array(polygons).ndim == 2:
            polygons = [polygons]
    except:
        pass

    if fill:
        if not draw_split:
            mask = cv2.fillPoly(mask, [np.array(p).astype('int32') for p in polygons], color)
        else:
            for p in polygons:
                mask = cv2.fillPoly(mask, [np.array(p).astype('int32')], color)
    else:
        mask = cv2.polylines(mask, [np.array(p).astype('int32') for p in polygons], True, color=color, thickness=thickness)

    return mask

def xyxy_to_yolo_label(bbox, class_id, image_width, image_height):
    x1, y1, x2, y2 = bbox
    x_center = (x1 + x2) / (2 * image_width)
    y_center = (y1 + y2) / (2 * image_height)
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height
    return [class_id, x_center, y_center, width, height]

def draw_boxes(image, bboxes, color=None, thickness=1, fill=False):
    image = image.copy()
    if color is None:
        color = [255, 255, 255]
    if np.array(bboxes).ndim == 1 and len(bboxes) == 4:
        bboxes = [bboxes]
    if fill:
        thickness = -1
    for box in bboxes:
        box = vvd_round(box)
        box[2] = box[2] - box[0]
        box[3] = box[3] - box[1]
        image = cv2.rectangle(image, box, color, thickness=thickness)
    return image

def get_xyxys_from_mask(mask):
    """ get xyxys for all isolated areas on fg-mask """
    contours = get_polygons_from_mask(mask)
    xyxys = list()
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        xyxys.append([x,y,x+w,y+h])
    return xyxys

def get_xyxy_from_mask(mask):
    """ get one xyxy for all isolated areas on fg-mask """
    contours = get_polygons_from_mask(mask)
    contour = np.vstack(contours)
    x,y,w,h = cv2.boundingRect(contour)

    return [x,y,x+w,y+h]

def get_polar_ring(image, center, outer_radius, inner_radius=0, factor=1, interpolation=12):
    size_X = vvd_round(outer_radius)
    size_Y = vvd_round(outer_radius * 2 * np.pi)
    
    dsize = [size_X, size_Y]
    center = vvd_round(center)
    ori_polar = cv2.warpPolar(image, dsize, center, size_X, interpolation)
    band = ori_polar[:, vvd_round(inner_radius):, ...]
    if factor != 1:
        band = image_resize(band, factor=factor)
    return band


def apply_gimp_brightness_contrast(input_img, brightness = 0, contrast = 0):
    """
    Gimp style brightness and contrast adjusification
    bringhtness: [-127, 127] --> [dark, bright]
    contrast: [-127, 127]  --> [low contrast, high contrast]
    """
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

def connected_filter(image, min_area=0, w_h_ratio=None, largest_num=None, w_min=None, w_max=None, h_min=None, h_max=None, final_value=1):
    image = to_gray_image(image)
    image = (image != 0).astype('uint8')
    num, mark_img, info, _ = cv2.connectedComponentsWithStats(image)
    if 0 < min_area < 1:
        min_area = min_area * time_reduce(*image.shape[:2])
    if largest_num is not None and num > 1:
        largest_num = max(0, min(vvd_round(largest_num), num - 1))
        pixel_array = info[:, -1][1:]
        index_array, _ = get_Several_MinMax_Array(pixel_array, -largest_num)
        index_array += 1
    else:
        index_array = np.nonzero(info[:, -1] > min_area)[0]
    ratio_check = False
    if w_h_ratio is not None:
        min_ratio = np.min(w_h_ratio)
        max_ratio = np.max(w_h_ratio)
        ratio_check = True
    for index in index_array:
        if index > 0:
            sub_info = info[index]
            if ratio_check:
                if min_ratio <= sub_info[2] / max(1, sub_info[3]) <= max_ratio:
                    pass
                else:
                    continue
            if w_min is not None:
                if sub_info[2] < w_min:
                    continue
            if w_max is not None:
                if sub_info[2] > w_max:
                    continue
            if h_min is not None:
                if sub_info[3] < h_min:
                    continue
            if h_max is not None:
                if sub_info[3] > h_max:
                    continue
            mark_img[mark_img == index] = - index
    res_img = (mark_img < 0).astype('uint8') * final_value
    return res_img


def close_connected_area(image, convex=False):
    image = to_gray_image(image)
    image = (image != 0).astype('uint8')
    num, mark_img, info, _ = cv2.connectedComponentsWithStats(image)
    for index in range(num):
        if index > 0:
            contours, hierarchy = cv2.findContours((mark_img == index).astype('uint8'), 2, 1)
            if convex:
                contours = cv2.convexHull(np.vstack(contours), clockwise=True).squeeze()
                if contours.ndim > 1:
                    image = cv2.fillConvexPoly(image, np.vstack(contours), (1, 1, 1))
            else:
                for contour in contours:
                    image = cv2.fillConvexPoly(image, contour, (1, 1, 1))
    return image


def get_contours_from_img(img):
    contours, _ = cv2.findContours((img != 0).astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(contours)
    for index, contour in enumerate(contours):
        contour = cv2.approxPolyDP(contour, 1, True)
        contours[index] = contour.squeeze()
    return contours


def get_convexHull(mask):
    contours, _ = cv2.findContours((mask != 0).astype('uint8'), 2, 1)
    convexHull_points = cv2.convexHull(np.vstack(contours), clockwise=True).squeeze()
    return convexHull_points


def bounding_bbox(mask):
    x, y, w, h =  cv2.boundingRect(get_convexHull(mask))
    bbox = [x, y, x + w, y + h]
    return bbox


def get_kernal(ksize, shape='rec'):
    """
    ksize = [W, H]
    shape in ['rec', 'circle', 'cross']
    """

    if is_number(ksize):
        ksize = [ksize, ksize]

    assert len(ksize) == 2 
    ksize = vvd_round(ksize)
    assert np.min(ksize) > 0, f"kernal size should be > 0, {ksize}"

    if shape not in ['rec', 'circle', 'cross']:
        shape = 'rec'
    if shape == 'rec':
        flag = cv2.MORPH_RECT
    elif shape == 'circle':
        flag = cv2.MORPH_ELLIPSE
    elif shape == 'cross':
        flag = cv2.MORPH_CROSS
    else:
        raise RuntimeError('unknown shape {shape}')
    
    kernal = cv2.getStructuringElement(flag, ksize)
    return kernal


def morphology_op(image, op, kernal_size, kernal_type=None):
    """
    op in ['open', 'close', 'erode', 'dilate']
    ksize = [W, H]
    kernal_type in ['rec', 'circle', 'cross']
    """
    op_list = ['open', 'close', 'erode', 'dilate']
    assert op in op_list, f"op {op} not in {op_list}"
    image = (image != 0).astype('uint8')
    kernal = get_kernal(kernal_size, shape=kernal_type)
    if op == 'open':
        op_mark = cv2.MORPH_OPEN
    elif op == 'close':
        op_mark = cv2.MORPH_CLOSE
    elif op == 'erode':
        op_mark = cv2.MORPH_ERODE
    elif op == 'dilate':
        op_mark = cv2.MORPH_DILATE
    else:
        raise RuntimeError(f"unknown op mark {op_mark}")

    return cv2.morphologyEx(image, op_mark, kernal)
    


def erode_op(image, kernal_size, kernal_type=None):
    """
    ksize = [W, H]
    kernal_type in ['rec', 'circle', 'cross']
    """
    return morphology_op(image, 'erode', kernal_size, kernal_type)


def dilate_op(image, kernal_size, kernal_type=None):
    """
    ksize = [W, H]
    kernal_type in ['rec', 'circle', 'cross']
    """
    return morphology_op(image, 'dilate', kernal_size, kernal_type)


def open_op(image, kernal_size, kernal_type=None):
    """
    ksize = [W, H]
    kernal_type in ['rec', 'circle', 'cross']
    """
    return morphology_op(image, 'open', kernal_size, kernal_type)


def close_op(image, kernal_size, kernal_type=None):
    """
    ksize = [W, H]
    kernal_type in ['rec', 'circle', 'cross']
    """
    return morphology_op(image, 'close', kernal_size, kernal_type)


def image_center_paste(image, canvas):
    assert image.ndim == canvas.ndim
    I_H, I_W = image.shape[:2]
    C_H, C_W = canvas.shape[:2]
    
    assert C_H >= I_H and C_W >= I_W, f"image shape {image.shape} is larger than canvas' {canvas.shape}"
    
    H_gap = C_H - I_H
    W_gap = C_W - I_W
    
    top = vvd_floor(H_gap / 2)
    down = top + I_H
    left = vvd_floor(W_gap / 2)
    right = left + I_W
    
    canvas[top:down, left:right, ...] = image
    return canvas


def puzzle(image_list, col_num = None, row_num = None, margin = 20, white_edge=False):

    image_num = len(image_list)
    if col_num is not None and row_num is not None:
        raise RuntimeError(f"At most one of row_num and col_num could be set. col_num: {col_num}, row_num: {row_num}")
    else:
        if col_num is not None:
            col_num = max(1, col_num)
            assert row_num is None, "Bug of data num settings"
            row_num = int(np.ceil(image_num/col_num))
        elif row_num is not None:
            row_num = max(1, row_num)
            assert col_num is None, "Bug of data num settings"
            col_num = int(np.ceil(image_num/row_num))
        else:
            col_num = int(np.ceil(image_num**0.5))
            row_num = int(np.ceil(image_num/col_num))

    

    col_size = [ -1 for _ in range(col_num)]
    row_size = [ -1 for _ in range(row_num)]

    ndim = -1

    for row_index in range(row_num):
        base = row_index * col_num
        for col_index in range(col_num):
            image_index = col_index + base
            if image_index >= image_num:
                break
            H, W = image_list[image_index].shape[:2]
            row_size[row_index] = H if H > row_size[row_index] else row_size[row_index]
            col_size[col_index] = W if W > col_size[col_index] else col_size[col_index]
            ndim = image_list[image_index].ndim if image_list[image_index].ndim > ndim else ndim

    assert ndim in [2, 3]
    assert np.min(col_size) > 0 and np.min(row_size) > 0

    Image_H = np.sum(row_size) + (row_num - 1) * margin
    Image_W = np.sum(col_size) + (col_num - 1) * margin

    if ndim == 2:
        image = np.zeros([Image_H, Image_W], dtype='uint8')
    elif ndim == 3:
        image = np.zeros([Image_H, Image_W, 3], dtype='uint8')
    else:
        raise ValueError(f"bad ndim value {ndim}")

    for row_index in range(row_num):
        base = row_index * col_num
        for col_index in range(col_num):
            image_index = col_index + base
            if image_index >= image_num:
                break

            top = np.sum(row_size[:row_index]).astype('int') + row_index * margin
            down = top + row_size[row_index]
            left = np.sum(col_size[:col_index]).astype('int') + col_index * margin
            right = left + col_size[col_index]

            crop = image[top:down, left:right, ...]

            cur_image = image_list[image_index]
            if ndim == 3:
                cur_image = to_colorful_image(cur_image)

            if white_edge:
                cur_image = cur_image.copy()
                cur_image[0, :, ...] = 255
                cur_image[-1, :, ...] = 255
                cur_image[:, 0, ...] = 255
                cur_image[:, -1, ...] = 255

            image_center_paste(cur_image, crop)

    return image


def crop_tilted_rectangle(image, point_1, point_2, width):
    # 裁剪倾斜矩形
    # 原理是利用 meshgrid 乘以旋转矩阵，从原图中采样拼接得到

    # point -> col, row
    # 图像坐标系下，让 point1 在图像下方 即 row 更大
    if point_1[1] < point_2[1]:
        point_1, point_2 = point_2, point_1

    # 计算矩形的高度和宽度
    height = cal_distance(point_1, point_2)
    assert height > 0, f"两点距离必须大于0 {height}"

    x = np.linspace(-width/2, -width/2 + round(width) -1, round(width))
    y = np.linspace(-height/2, -height/2 + round(height) -1, round(height))

    X, Y = np.meshgrid(x, y)

    # 计算矩形的倾斜角度
    # 正上方为零度，逆时针为正角度
    alpha = math.asin((point_1[0] - point_2[0]) / height)

    # 平移距离
    move_x = (point_1[0] + point_2[0]) / 2
    move_y = (point_1[1] + point_2[1]) / 2

    # 计算旋转矩阵
    rotation_matrix = np.array([[np.cos(alpha), -np.sin(alpha)],
                                [np.sin(alpha), np.cos(alpha)]
    ])

    # 旋转 meshgrid
    X_rotated, Y_rotated = np.dot(rotation_matrix.T, np.array([X.flatten(), Y.flatten()]))

    # 变回之前的形状
    X_rotated = X_rotated.reshape(X.shape)
    Y_rotated = Y_rotated.reshape(Y.shape)

    # 平移
    X_rotated += move_x
    Y_rotated += move_y

    X_rotated = X_rotated.astype('int32')
    Y_rotated = Y_rotated.astype('int32')

    # 防止图像裁边溢出
    X_rotated = np.clip(X_rotated, 0, image.shape[1]-1)
    Y_rotated = np.clip(Y_rotated, 0, image.shape[0]-1)

    # 裁剪图像
    cropped_image = image[Y_rotated, X_rotated]
    return cropped_image


# 画一个图像各通道的直方图
def draw_hist(rgb_img):
    color = ('r', 'g', 'b')
    for i, col in enumerate(color):
        hist = cv.calcHist([rgb_img], [i], None, [256], [0, 256])
        # print(hist.shape)
        plt.plot(hist, color=col)
        plt.xlim([0, 256])
    plt.show()

def gray_map_to_color_map(gray_map: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Compute color heatmap.

    Args:
        gray_map (np.ndarray): Final gray map computed by the distance metric.
        normalize (bool, optional): Bool to normalize the gray map prior to applying
            the color map. Defaults to True.

    Returns:
        np.ndarray: [description]
    """
    if normalize:
        gray_map = (gray_map - gray_map.min()) / np.ptp(gray_map)
    gray_map = gray_map * 255
    gray_map = gray_map.astype(np.uint8)

    gray_map = cv2.applyColorMap(gray_map, cv2.COLORMAP_JET)
    gray_map = cv2.cvtColor(gray_map, cv2.COLOR_BGR2RGB)
    return gray_map

def superimpose_map(
    image: np.ndarray, gray_map: np.ndarray, alpha: float = 0.4, gamma: int = 0, normalize: bool = False) -> np.ndarray:
    """Superimpose gray map on top of in the input image.

    Args:
        gray_map (np.ndarray): gray map
        image (np.ndarray): Input image
        alpha (float, optional): Weight to overlay gray map
            on the input image. Defaults to 0.4.
        gamma (int, optional): Value to add to the blended image
            to smooth the processing. Defaults to 0. Overall,
            the formula to compute the blended image is
            I' = (alpha*I1 + (1-alpha)*I2) + gamma
        normalize: whether or not the gray maps should
            be normalized to image min-max

    Returns:
        np.ndarray: Image with gray map superimposed on top of it.
    """
    gray_map = to_gray_image(gray_map)
    if gray_map.shape[:2] != image.shape[:2]:
        print(f"Warning: GrayMap shape {gray_map.shape[:2]} does not equal to Image shape {image.shape[:2]}")
        H, W = image.shape[:2]
        gray_map = image_resize(gray_map, [W, H], uint8=False)
    gray_map = gray_map_to_color_map(gray_map.squeeze(), normalize=normalize)
    superimposed_map = cv2.addWeighted(gray_map, alpha, image, (1 - alpha), gamma)
    return superimposed_map

def get_connect_components(img):
    img = to_gray_image((img!= 0).astype('uint8'))
    com_num, labels, stats, _ = cv2.connectedComponentsWithStats(img)
    return com_num, labels, stats

def mask_on_mask(input_mask, standard_mask):
    input_mask_pro = to_gray_image(input_mask != 0)
    standard_mask_pro = to_gray_image(standard_mask != 0)
    assert input_mask_pro.shape == standard_mask_pro.shape
    com_num, labels, stats = get_connect_components(input_mask_pro)
    for index in range(com_num):
        if index > 0:
            if standard_mask_pro[labels == index].max() > 0:
                labels[labels == index] = -1
    final_mask = (labels < 0).astype('uint8')
    return final_mask

def img2video(image_list, output_video_name, fps, resize=None):
    # 图像转视频 基于 OpenCV 
    # 参数：image_list: rgb 图像列表，output_video_name: 输出视频文件名，fps: 视频帧率，resize: 图像尺寸调整，None 表示不调整

    if resize is not None:
        shape = (resize[0], resize[1])
    else:
        shape = None

    # 获取图像的尺寸
    image = image_list[0]
    image = to_colorful_image(image)
    if shape is not None:
        image = image_resize(image, shape)

    height, width, channels = image.shape
    shape = (width, height)

    # 创建视频 writer
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')  # 使用mp4视频编码
    out = cv2.VideoWriter(output_video_name, fourcc, fps, (width, height))

    # 遍历图像路径列表，并将图像写入视频文件
    for image in tqdm(image_list, desc=f' @@ image to video running: '):
        image = to_colorful_image(image)
        image = cv_rgb_bgr_convert(image)
        image = image_resize(image, shape)
        out.write(image)

    # 释放writer
    out.release()

def video_process(video_path, output_path=None, max_size = None, callback=None, remove_original=False):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if output_path is None:
        output_path = path_insert_content(video_path, '_processed')
    
    # 获取视频的宽度和高度
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # 获取帧率

    # 定义视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    if max_size is not None:
        cur_max_size = max(width, height)
        if cur_max_size > max_size:
            scale = max_size / cur_max_size
            width = int(width * scale)
            height = int(height * scale)
            

    # 创建VideoWriter对象，用于写入处理后的视频
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    progress_bar = tqdm(total=frame_count, desc=f'Video Processing {video_path}')

    success = False

    # 检查视频是否打开
    if not cap.isOpened():
        print(" !! Error opening video file")
    else:
        while True:
            # 逐帧读取视频
            ret, frame = cap.read()
            if not ret:
                break  # 视频读取完毕，退出循环

            # 对帧进行操作
            if callback is not None:
                h, w, c = frame.shape
                frame = callback(frame)
                h_p, w_p, c_p = frame.shape
                assert h_p == h and w_p == w and c_p == c,  " !! Callback function should not change the size of the frame"

            frame = image_resize(frame, shape=[width, height])

            # 将处理后的帧写入输出视频
            out.write(frame)
            progress_bar.update(1)
        
        success = True

    # 释放VideoCapture和VideoWriter对象
    cap.release()
    out.release()

    if remove_original and success:
        remove_file(video_path)

    # 关闭所有窗口
    cv2.destroyAllWindows()
    pass

def imgpath2video(image_path_list, output_video_name, fps, resize=None):
    # 图像转视频 基于 OpenCV 
    # 参数：image_path_list: 图像路径列表，output_video_name: 输出视频文件名，fps: 视频帧率，resize: 图像尺寸调整，None 表示不调整

    if resize is not None:
        shape = (resize[0], resize[1])
    else:
        shape = None

    # 获取图像的尺寸
    image = cv2.imread(image_path_list[0])
    image = to_colorful_image(image)
    if shape is not None:
        image = image_resize(image, shape)

    height, width, channels = image.shape
    shape = (width, height)

    # 创建视频 writer
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')  # 使用mp4视频编码
    out = cv2.VideoWriter(output_video_name, fourcc, fps, (width, height))

    # 遍历图像路径列表，并将图像写入视频文件
    for image_path in tqdm(image_path_list, desc=f' @@ image to video running: '):
        image = cv2.imread(image_path)
        image = to_colorful_image(image)
        image = image_resize(image, shape)
        out.write(image)

    # 释放writer
    out.release()


def distortion_calibration(image_dir_path, W_num, H_num, show=False):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,6,0)

    objp = np.zeros((W_num*H_num,3), np.float32)
    objp[:,:2] = np.mgrid[0:W_num,0:H_num].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    image_path_list = glob_images(image_dir_path)
    if len(image_path_list) == 0:
        raise Exception(f'no image found in {image_dir_path}.')

    valid_image_path_list = []
    image_shape = None

    for image_path in tqdm(image_path_list, desc=f' @@ calibration running: '):
        img = cv_rgb_imread(image_path, 1)
        if image_shape is None:
            image_shape = img.shape[:2]
        else:
            assert image_shape == img.shape[:2], f' !! image {image_path} shape not match: {image_shape} vs {img.shape[:2]}.'

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(img, (W_num, H_num), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            valid_image_path_list.append(image_path)

            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(img, corners, (11, 11), (-1,-1), criteria)
            imgpoints.append(corners2)

            if show:
                copy_img = img.copy()
                # Draw and display the corners
                cv2.drawChessboardCorners(copy_img, (W_num, H_num), corners2, ret)
                PIS(copy_img)
        else:
            print(f' !! {image_path} findChessboardCorners failed. ')

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)

    print("\n @@ Camera Distortion Calibration calculation done.\n")

    # 打印内参矩阵
    print(" @@ Camera Matrix :")
    print(mtx)

    # 打印畸变系数
    print("\n @@ dist :")
    print(dist)
    print("\n")

    # 计算重投影误差
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
        print(f'@ image {i+1} error: {error}')
        if error > 0.1:
            print(f'  image path: {valid_image_path_list[i]}')

    # 打印平均误差
    print(f' @@ total mean error: {mean_error/len(objpoints)}')

    # matrix dist W H
    result = dict()
    result['mtx'] = mtx
    result['dist'] = dist
    result['W'] = image_shape[1]
    result['H'] = image_shape[0]

    return result


def distortion_calibration_to_map(mtx, dist, W, H, **kwargs):
    result = dict()

    result['mtx'] = mtx
    result['dist'] = dist
    result['W'] = W
    result['H'] = H

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (W,H), 1, (W,H))
    result['newcameramtx'] = newcameramtx

    map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (W, H), cv2.CV_32FC1)
    distort_bbox = [*(roi[:2]),roi[2]+roi[0], roi[3]+roi[1]]
    result['map1'] = map1
    result['map2'] = map2
    result['roi_bbox'] = distort_bbox

    # frame = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
    return result


def undistort_with_map_result(img, map_result, crop=False, interpolation=cv2.INTER_LINEAR):
    assert 'map1' in map_result, f' !! map1 not in map_result: {map_result}.'
    assert 'map2' in map_result, f' !! map2 not in map_result: {map_result}.'
    
    map1 = map_result['map1']
    map2 = map_result['map2']

    assert map1.shape[:2] == img.shape[:2] == map2.shape[:2], f' !! shape not match: map1 {map1.shape[:2]}, map2 {map2.shape[:2]}, img {img.shape}.'

    undistorted = cv2.remap(img, map1, map2, interpolation)

    if crop:
        roi_bbox = map_result['roi_bbox']
        undistorted = undistorted[roi_bbox[1]:roi_bbox[3], roi_bbox[0]:roi_bbox[2]]

    return undistorted


def colorful_mask(ori_output, color_dict=None):
    # mask to colorful image
    if color_dict is None:
        color_dict = {
            0: (160, 160, 160),
            1: (0, 255, 255),
            2: (255, 255, 0),
            3: (170, 85, 85),
            4: (20, 0, 255),
            5: (255, 0, 20),
            6: (176, 58, 221),
            7: (88, 138, 165),
            8: (119, 176, 54),
            9: (230, 95, 54),
        }
    puzzle = np.zeros([ori_output.shape[0], ori_output.shape[1], 3], np.float16)
    for key in color_dict.keys():
        puzzle[ori_output==key] = color_dict[key]

    return puzzle.astype('uint8')


def huge_image_load(file_path):
    import imageio.v3 as iio
    with iio.imopen(file_path, 'r') as image_file:
        img = image_file.read()
    return img


def read_mongodb_image(mongo_img_file_obj, obj_type: str):
    import tifffile
    image_stream = io.BytesIO(mongo_img_file_obj.read())
    Image.MAX_IMAGE_PIXELS = None

    if obj_type == 'tif':
        with tifffile.TiffFile(image_stream) as tif:
            image = tif.asarray()

    else:
        pil_image = Image.open(image_stream)
        # pil_image = pil_image.convert('RGB')
        image = np.array(pil_image)                             # 5000 * 4000 * 3 uint8 图像耗时 0.17 s

    return image


def make_voronoi_grad_mask(width, height, voronoi_polygon, gradient_width):
    voronoi_img = np.zeros((height, width), dtype=np.float32)
    cv2.polylines(voronoi_img, voronoi_polygon[None, :, :].astype('int32'), isClosed=True, color=255, thickness=vvd_round(gradient_width))

    Ys, Xs = np.where(voronoi_img > 0)
    voronoi_img[...] = 0

    cv2.fillPoly(voronoi_img, voronoi_polygon[None, :, :].astype('int32'), color=1)
    cv2.erode(voronoi_img, np.ones((7, 7), dtype=np.uint8), voronoi_img)

    for x, y in zip(Xs, Ys):
        row, col = int(y), int(x)
        distance = cv2.pointPolygonTest(voronoi_polygon, (col, row), True)
        value = np.clip(0.5 + distance / gradient_width, 0, 1)
        voronoi_img[row, col] = value

    return voronoi_img


def make_voronoi_mask(width, height, voronoi_polygon):
    voronoi_img = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(voronoi_img, voronoi_polygon[None, :, :].astype('int32'), color=1)

    return voronoi_img


class UndistortProcessor:
    def __init__(self, fx, fy, cx, cy, k1, k2, p1, p2, k3, W, H):

        mtx = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        dist = np.array([k1, k2, p1, p2, k3])

        self.undistort_param_dict = distortion_calibration_to_map(mtx, dist, W, H)
        self.W = W
        self.H = H

    def __call__(self, img, interpolation=cv2.INTER_LINEAR):
        return self.process_img(img, interpolation=interpolation)

    def process_img(self, img, crop=False, interpolation=cv2.INTER_LINEAR):
        W, H = img.shape[1], img.shape[0]
        assert W == self.W and H == self.H, f"Image size {W}x{H} is not equal to {self.W}x{self.H}"

        undistort_image = undistort_with_map_result(img, self.undistort_param_dict, crop, interpolation)

        return undistort_image

    # data_processor = UndistortProcessor(3706.15, 3706.15, 2639.71, 1937.01, -0.110588, 0.0124022, -0.0001086, -0.000254551, -0.0262282, 5280, 3956)
    # for image_path in vv.tqdm(img_path_list):

    #     img = vv.cv_rgb_imread(image_path)
        
    #     processed_img = data_processor(img)
    #     vv.PIS([img, 'ori_img'], [processed_img, 'undistorted'])
