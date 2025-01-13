import numpy as np
import re
import hashlib


def get_Several_MinMax_Array(np_arr, several):
    """
    获取numpy数值中最大或最小的几个数
    :param np_arr:  numpy数组
    :param several: 最大或最小的个数（负数代表求最大, 正数代表求最小）
    :return:
        several_min_or_max: 结果数组
    """
    np_arr = np.array(np_arr)
    if several > 0:
        if several == len(np_arr):
            index_pos = np.argmin(np_arr)
        else:
            index_pos = np.argpartition(np_arr, several)[:several]
    else:
        if several == len(np_arr):
            index_pos = np.argmax(np_arr)
        else:
            index_pos = np.argpartition(np_arr, several)[several:]
    several_min_or_max = np_arr[index_pos]
    return index_pos, several_min_or_max

def array_md5(array, dtype=None):
    float_array = np.array(array, dtype=dtype)
    array_str = float_array.tobytes().hex()
    md5_str = hashlib.md5(array_str.encode()).hexdigest()
    return md5_str

def data_L_n_normalize(data, Ln=2, target = 1):
    assert target > 0
    assert Ln > 0
    data = np.array(data, 'float32')
    norm_value = (np.sum(data ** Ln)) ** (1 / Ln)
    return data / norm_value * target

def get_non_zero_parts(signal):
    signal = np.array(signal) != 0
    assert signal.ndim == 1
    if np.max(signal) == 0:
        return [], []
    temp_signal = np.hstack([0, signal, 0])
    diff_signal = temp_signal[1:] - temp_signal[:-1]
    starts = np.nonzero(diff_signal > 0)[0]
    ends = np.nonzero(diff_signal < 0)[0]
    assert len(starts) == len(ends)
    return starts, ends

def get_zero_parts(signal):
    signal = np.array(signal) == 0
    assert signal.ndim == 1
    return get_non_zero_parts(signal)

def find_longest_start_end(arr):
    substr = max(re.findall('1+', str(arr)))
    obj = re.search(substr, str(arr))
    return obj.start(), obj.end()


def get_longest_part(signal):
    signal = np.array(signal)
    if len(signal) == 0 or np.max(np.abs(signal)) == 0:
        return 0, 0

    signal = (signal != 0).astype('int8')
    str_signal = str(signal.tolist()).replace(', ', '')[1:-1]
    start, end = find_longest_start_end(str_signal)

    return start, end


def bilinear_interpolate(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1)
    x1 = np.clip(x1, 0, im.shape[1]-1)
    y0 = np.clip(y0, 0, im.shape[0]-1)
    y1 = np.clip(y1, 0, im.shape[0]-1)

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id


def bilinear_by_meshgrid_gray(image, x_grid, y_grid, fill_constant=None, zero_mask=False):

    #               Ia, Wd                          Ic, Wb
    #           (floor_x, floor_y)              (ceil_x, floor_y)   
    #
    #                               (x, y)
    #
    #               Ib , Wc                         Id, Wa
    #           (floor_x, ceil_y)               (ceil_x, ceil_y)   
    #

    assert x_grid.shape == y_grid.shape
    assert image.ndim == 2
    H, W = image.shape[:2]

    if fill_constant is not None:
        constant_mask_x = np.logical_or(x_grid < 0, x_grid >= W-1)
        constant_mask_y = np.logical_or(y_grid < 0, y_grid >= H-1)
        constant_mask = np.logical_or(constant_mask_x, constant_mask_y)

    x_grid = np.clip(x_grid, 0, W-1)
    y_grid = np.clip(y_grid, 0, H-1)

    floor_x_grid = np.floor(x_grid).astype('int32')
    floor_y_grid = np.floor(y_grid).astype('int32')

    ceil_x_grid = floor_x_grid + 1
    ceil_y_grid = floor_y_grid + 1

    if np.max(ceil_x_grid) > W -1 or  np.max(ceil_y_grid) > H -1 or np.min(floor_x_grid) < 0 or np.min(floor_y_grid) < 0:
        # print("Warning: index value out of original matrix, a crop operation will be applied.")

        floor_x_grid = np.clip(floor_x_grid, 0, W-1).astype('int32')
        ceil_x_grid = np.clip(ceil_x_grid, 0, W-1).astype('int32')
        floor_y_grid = np.clip(floor_y_grid, 0, H-1).astype('int32')
        ceil_y_grid = np.clip(ceil_y_grid, 0, H-1).astype('int32')

    x_grid = np.clip(x_grid, floor_x_grid, ceil_x_grid)
    y_grid = np.clip(y_grid, floor_y_grid, ceil_y_grid)

    Ia = image[ floor_y_grid, floor_x_grid ]
    Ib = image[ ceil_y_grid, floor_x_grid ]
    Ic = image[ floor_y_grid, ceil_x_grid ]
    Id = image[ ceil_y_grid, ceil_x_grid ]

    wa = (ceil_x_grid - x_grid) * (ceil_y_grid - y_grid)
    wb = (ceil_x_grid - x_grid) * (y_grid - floor_y_grid)
    wc = (x_grid - floor_x_grid) * (ceil_y_grid - y_grid)
    wd = (x_grid - floor_x_grid) * (y_grid - floor_y_grid)

    assert np.min(wa) >=0 and np.min(wb) >=0 and np.min(wc) >=0 and np.min(wd) >=0
    
    W = wa + wb + wc + wd

    empty_mask = W == 0
    if np.max(empty_mask) > 0:
        wa[empty_mask] = 0.25
        wb[empty_mask] = 0.25
        wc[empty_mask] = 0.25
        wd[empty_mask] = 0.25
    
    W = wa + wb + wc + wd

    assert np.abs(np.max(W) - 1) < 1e-8

    res_image = wa*Ia + wb*Ib + wc*Ic + wd*Id

    if fill_constant is not None:
        res_image = res_image * np.logical_not(constant_mask) + fill_constant * constant_mask

    if zero_mask:
        data_zero_mask = ~((Ia==0) | (Ib==0) | (Ic==0) | (Id==0))
        res_image = res_image * data_zero_mask

    return res_image

def bilinear_by_meshgrid(image, x_grid, y_grid, fill_constant=None):

    if np.ndim(image) == 2:
        return bilinear_by_meshgrid_gray(image, x_grid, y_grid, fill_constant)

    elif np.ndim(image) == 3:
        channel_res = list()
        for i in range(image.shape[2]):
            channel_res.append(bilinear_by_meshgrid_gray(image[:,:,i], x_grid, y_grid, fill_constant))
        return np.stack(channel_res, axis=2)
    else:
        raise NotImplementedError("Not implemented for data who has more than 3 dimensions.")


def get_start_and_end(data):
    data = np.array(data)
    assert data.ndim == 1
    signal = data != 0
    length = len(signal)
    start = np.argmax(signal)
    end = length - np.argmax(signal[::-1])
    return start, end
