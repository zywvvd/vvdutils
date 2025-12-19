# -*- coding: utf-8 -*-
# @Author: Zhang Yiwei
# @Date:   2020-08-12 17:39:40
# @Last Modified by:   Zhang Yiwei
# @Last Modified time: 2020-08-12 17:39:41
from functools import wraps
import cv2
from .utils import vvd_ceil
from ..loader import try_to_import


# Line Profiler Decorator
def line_profiling_deco(func):
    line_profiler = try_to_import('line_profiler', "pip install line_profiler")
    @wraps(func)
    def wrapped(*args, **kwargs):
        pr = line_profiler.LineProfiler(func)
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        pr.print_stats()
        return result

    return wrapped

try_to_import('numba', "pip install numba")
from numba import cuda

@cuda.jit
def image_process_cuda(img_cuda, result_img_cuda, y_size, x_size, radius=8, thre=18):
    y = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y
    x = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x

    radius = radius
    thre = thre
    if radius < y < y_size-radius and radius < x < x_size-radius:
        a = 0.0
        b = 0.0
        x_1 = float(img_cuda[y, x])
        for x_index in range(x-radius, x+radius+1):
            for y_index in range(y-radius, y+radius+1):
                x_i = float(img_cuda[y_index, x_index])
                
                temp_b = (1 - abs(x_i - x_1) / 2.5 / thre)
                if temp_b <= 0:
                    continue
                b += temp_b
                a += temp_b * x_i

        result_img_cuda[y, x] = int(round(a / b))


def surface_blur_gray(image, radius=8, thre=18):
    from numba import cuda
    BLOCK_SIZE = 16
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    y_size, x_size = image.shape[:2]
    
    threads_per_block = (BLOCK_SIZE, BLOCK_SIZE)
    blocks_per_grid_x = int(vvd_ceil(image.shape[1] / BLOCK_SIZE))
    blocks_per_grid_y = int(vvd_ceil(image.shape[0] / BLOCK_SIZE))
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    
    copy_image = image.copy()
    
    ori_image_cuda = cuda.to_device(image)
    
    copy_image_cuda = cuda.to_device(copy_image)
    image_process_cuda[blocks_per_grid, threads_per_block](ori_image_cuda, copy_image_cuda, y_size, x_size, radius, thre)
    result_img = copy_image_cuda.copy_to_host()

    return result_img
