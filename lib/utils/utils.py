# -*- coding: utf-8 -*-
# @Author: Zhang Yiwei
# @Date:   2020-07-18 02:40:35
# @Last Modified by:   Zhang Yiwei
# @Last Modified time: 2020-08-18 15:51:06
#
# vvd Tool functions
#
import inspect
import json
import time
import sys
import cv2 as cv
import logging
import os
import re
import random
import platform
import hashlib
import pickle
import uuid
import decimal
import shutil
import string
import datetime
import importlib

import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt

# 整合常用os操作
import os.path as osp
from os.path import basename as OS_basename
from os.path import join as OS_join
from os.path import exists as OS_exists
from os.path import isfile as OS_isfile
from os.path import isdir as OS_isdir
from os.path import dirname as OS_dirname

from numpy.lib.function_base import iterable
from pathlib2 import Path as Path2
from pathlib import Path
from glob import glob
from tqdm import tqdm
from collections import OrderedDict
from functools import wraps
from functools import reduce
from func_timeout import func_set_timeout, FunctionTimedOut

isfile = OS_isfile 

def rm_r(path):
    try:
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        elif os.path.exists(path):
            os.remove(path)
    except:
        print("Cannot remove %s" % path)

def get_line_info():
    # 获取当前帧对象
    current_frame = inspect.currentframe()
    # 获取上一级帧对象（即调用者）
    caller_frame = current_frame.f_back
    # 获取上上级帧对象
    caller_parent_frame = caller_frame.f_back
    
    # 获取上上级调用者的函数名、行号和文件名
    function_name = caller_parent_frame.f_code.co_name
    line_number = caller_parent_frame.f_lineno
    file_path = caller_parent_frame.f_code.co_filename
    
    # 提取文件名，去掉路径
    file_name = os.path.basename(file_path)
    
    # 使用 f-string 格式化字符串
    error_line = f"{file_name}:{function_name}:{line_number}, "
    
    return error_line

def lazy_import(module_name):
    print(f'Lazy import: {module_name}')
    return importlib.import_module(module_name)

def exists(input):
    if isinstance(input, list):
        res = True
        for path in input:
            if not OS_exists(str(path)):
                res = False
                break
        return res
    else:
        return OS_exists(input)

def random_color(norm=False):
    if norm:
        return (random.random(), random.random(), random.random())
    else:
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def get_list_from_list(data_list, call_back, absolutely=False):
    """[make a list through a input list,
    while the output list will collect the output of a function dealing with every item of the input list]

    Args:
        data_list ([list/np.ndarray]): [original input list]
        call_back ([function]): [a call back function to do sth with every item of input list]
        absolutely([bool]): add result of call_back function to output_list whatever it is
    Returns:
        output_list[list]: [collection of output of call_back function]
    """
    output_list = list()
    if isinstance(data_list, np.ndarray):
        data_list = data_list.tolist()
    if isinstance(data_list, list):
        for data in data_list:
            res = call_back(data)
            if res is not None or absolutely:
                output_list.append(res)
    elif isinstance(data_list, dict):
        for key, data in data_list.items():
            res = call_back(data)
            if res is not None or absolutely:
                output_list.append(res)
    else:
        raise RuntimeError('input should be list or dict')
    return output_list

def segment_intersection(seg_1, seg_2):
    """ get intersection area from two segment

    Args:
        seg_1: two values which indicate start1 and end1
        seg_2 : two values which indicate start2 and end2

    Returns:
        value: length of intersection area
        tuple: (min, max)
    """
    assert len(seg_1) == len(seg_2) == 2
    min_value = max(min(seg_1), min(seg_2))
    max_value = min(max(seg_1), max(seg_2))
    inter_dis = max(0, max_value - min_value)
    return inter_dis, (min_value, max_value)

def concat_generator(*iterables):
    """ merge iterables to one generator
    Args: 
        iterables: Several unnamed parameters. All parameters are iterable variables
    
    Returns:
        Returns a generator that integrates all iterators
    """
    for it in iterables:
        for element in it:
            yield element

def get_mac_address():
    """get mac address

    Returns:
        str: mac address
    """
    mac = uuid.UUID(int = uuid.getnode()).hex[-12:].upper()
    return ":".join([mac[e: e+2] for e in range(0, 11, 2)])

def join_substring(*strs, conn_str=''):
    """join string with conn_str

    Returns:
        str: joind string
    """
    return str(conn_str).join(map(str, strs))

def cal_distance(vector_1, vector_2, axis=None):
    """Calculate the Euclidean distance between the features 

    Args:
        vector_1: feature 1
        vector_2: feature 2
        axis: axis in numpy

    Returns:
        distance
    """
    assert len(vector_1) == len(vector_2) or len(vector_2) == 1
    distance = (np.sum((np.array(vector_1) - np.array(vector_2)) ** 2, axis=axis)) ** 0.5
    return distance

def get_file_size_M(file_path):
    """[get file size]

    Args:
        file_path ([str/Path]): [path to file]

    Returns:
        [float]: [size of file by M]
    """
    return os.path.getsize(str(file_path)) / 1024 / 1024


def unify_data_to_python_type(data):
    """[transfer numpy data to python type]

    Args:
        data ([dict/list]): [data to transfer]

    Returns:
        [type of input data]: [numpy data will be transfered to python type]
    """
    return json.loads(json.dumps(data, cls=MyEncoder))


def timer_vvd(func):
    """
    a timer for func
    you could add a @timer_vvd ahead of the fun need to be timed
    Args:
        func (function): a function to be timed

    Outputs:
        time message: a message which tells you how much time the func spent will be printed
    """
    is_static_method = False
    try :
        func_name = func.__name__
    except Exception as e:
        func_name = func.__func__.__name__
        func = func.__func__
        is_static_method = True

    @wraps(func)
    def wrapper(*args, **kwargs):
        if is_static_method:
            args = args[1:]

        start_time = time.time()
        res = func(*args, **kwargs)
        end_time = time.time()
        print('func: {_funcname_} runing: {_time_}s'.format(_funcname_=func_name, _time_=format(end_time - start_time, '.6f')))
        return res
    return wrapper


def two_way_dictionary(dict_obj):
    length = len(dict_obj)
    key_set = set(dict_obj.keys())
    value_set = set(dict_obj.values())
    
    new_dict_obj = dict()
    if len(set(value_set)) == length:
        if len(value_set | key_set) == length * 2:
            for key, value in dict_obj.items():
                new_dict_obj[value] = key
    new_dict_obj.update(dict_obj)
    return new_dict_obj
    

def file_rename(srcFile, dstFile):
    """rename file

    Args:
        srcFile (path): path to source file
        dstFile (path): path to target file
    """
    srcFile = str(srcFile)
    dstFile = str(dstFile)
    assert OS_exists(srcFile), f"srcFile {srcFile} not found"
    try:
        os.rename(srcFile, dstFile)
    except Exception as e:
        print(e)
        print('rename file fail\r\n')

def file_read_lines(file_path, encoding='utf8'):
    if not OS_exists(file_path):
        print("file {} not found, None will be return".format(file_path))
        return None

    with open(file_path, "r", encoding=encoding) as f:
        lines = f.readlines()
        for index, line in enumerate(lines):
            lines[index] = line.strip('\n')
        return lines


def file_write_lines(line_list, file_path, overwrite=False, append=False, verbose=False):
    dir_check(OS_dirname(file_path))
    exist = exists(file_path)
    if not append or not exist:
        file_path = save_file_path_check(file_path, overwrite, verbose)
        with open(file_path, 'w', encoding='utf8') as f:
            f.writelines('\n'.join(line_list))
    else:
        with open(file_path, 'a', encoding='utf8') as f:
            f.writelines('\n'.join(line_list))


def pickle_save(object, save_path, overwrite=False, verbose=False):
    """
    将object保存为pickle文件到save_path中
    """
    save_path = save_file_path_check(save_path, overwrite, verbose)
    with open(save_path, 'wb') as fp:
        pickle.dump(object, fp)


def pickle_load(load_path):
    """
    从load_path中读取object
    """
    assert isinstance(load_path, str) or is_path_obj(load_path)
    if isinstance(load_path, str):
        load_path = load_path.replace('\\', '/')
    with open(load_path, 'rb') as fp:
        return pickle.load(fp)


def find_sub_string(string, substring, times):
    """
    find the char position of the substring in string for times-th comes up
    """
    current = 0
    for _ in range(1, times+1):
        current = string.find(substring, current+1)
        if current == -1:
            return -1

    return current

def str_connection(*str_args, connect_char='_'):
    """
    connect strings in the list with underline
    """
    string = str.join(connect_char, map(str, str_args))
    return string


def get_main_file_name(string):
    """
    return file name without extension
    """
    assert isinstance(string, str) or is_path_obj(string)
    if is_path_obj(string):
        string = str(string)
    return os.path.splitext(os.path.basename(string))[0]


def strong_printing(*str_args):
    """
    print string strongly
    """
    assert isinstance(str_args, tuple)
    string = str_connection(*str_args, connect_char=' ')
    print()
    boudary_size = int(max(40, len(string)*1.4))
    split_string = boudary_size*'#'
    print(split_string)
    space_size = (boudary_size - len(string))//2
    print(space_size*' '+string.upper())
    print(split_string)
    print()


def current_system():
    """
    返回当前操作系统名称字符串
    """
    return platform.system()


def current_split_char():
    """
    返回当前操作系统的路径分隔符
    """
    if current_system() == 'Windows':
        return '\\'
    elif current_system() == 'Linux':
        return '/'
    else:
        return '/'


def save_file_path_check(save_file_path, overwrite=False, verbose=False):
    """
    检查要保存的文件路径
    - 如果文件已经存在 : 在文件名与扩展名之间加入当前时间作为后缀 避免覆盖之前的文件并给出提示
    - 如文件不存在 : 检查文件所在的文件夹目录
    返回检查后的文件路径
    """
    if is_path_obj(save_file_path):
        save_file_path = str(save_file_path)

    assert isinstance(save_file_path, str)
    if OS_exists(save_file_path):
        if overwrite:
            checked_save_file_path = save_file_path
            if verbose:
                print("file path {} already exists, the file will be overwrite.".format(save_file_path))
        else:
            main_file_name = get_main_file_name(save_file_path)
            new_base_name = OS_basename(save_file_path).replace(main_file_name, str_connection(main_file_name, time_stamp()))
            checked_save_file_path = OS_join(OS_dirname(save_file_path), new_base_name)
            if verbose:
                print("file path {} already exists, the file will be saved as {} instead.".format(save_file_path, checked_save_file_path))
    else:
        dir_check(str(Path(save_file_path).parent), verbose)
        assert OS_basename(save_file_path) != ''
        checked_save_file_path = save_file_path
    return checked_save_file_path


def has_chinese_char(input_str):
    """check if unASCII char in a string

    Args:
        input_str (str): test str

    Returns:
        True: there is at least one unACSII char in the input_str
        False: no unACSII char in the input_str
    """
    for char in input_str:
        if ord(char) > 255:
            return True
    return False


def encode_chinese_to_unicode(input_string, remove=False):
    '''
    将中文转换为 unicode #Uxxxx 形式
    Args:
        input_string: input string
        remove: if remove Chinese chars
    Returns:
        result string
    '''
    unicode_string = ''
    for char in input_string:
        if ord(char) > 255:
            if remove:
                continue
            char = "%%U%04x" % ord(char)
        unicode_string += char
    unicode_string = unicode_string.replace('%', '#')
    return unicode_string

def create_uuid():
    """ create a uuid (universally unique ID) with length 32"""
    md5_hash = hashlib.md5(uuid.uuid1().bytes)
    return md5_hash.hexdigest()

def string_hash(s):
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def md5(data):
    return hashlib.md5(str(data).encode(encoding='utf-8')).hexdigest()


def xor(A, B):
    if (A and not B) or (B and not A):
        return True
    else:
        return False

def get_file_hash_code(file, extra_info=None):
    """
    获取文件hash值
    """
    assert os.path.exists(file), f"file {file} does not exist"
    md5_hash = hashlib.md5()
    with open(file, "rb") as fid:
        md5_hash.update(fid.read())
        digest = md5_hash.hexdigest()
    if extra_info is not None:
        digest += str(extra_info)
        digest = md5(digest)
    return digest

def get_dir_hash_code(dir_path, extra_info=None):
    """
    获取文件夹hash值
    """
    assert os.path.exists(dir_path), f"dir {dir_path} does not exist"
    _, file_list = get_dir_file_list(dir_path, recursive=True)
    dir_path_obj = Path(dir_path)
    hash_str = ""

    for file in file_list:
        file_path_obj = Path(file)
        hash_str += get_file_hash_code(file)
        relative_path = str(file_path_obj.relative_to(dir_path_obj))
        hash_str += '-' + relative_path + '@'

    if extra_info is not None:
        hash_str += '-extra_info-' + str(extra_info)

    return md5(hash_str)

def zip_dir(dir_path, zip_file_path): # 压缩文件夹
    import zipfile as zipfile

    assert os.path.isdir(dir_path), f'dir_path {dir_path} is not a dir'
    assert dir_check(OS_dirname(zip_file_path)), f'zip_file_path {zip_file_path} dir create_failed exist'

    dir_name = get_path_stem(dir_path)

    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as zipf:
        _, file_path_list = get_dir_file_list(dir_path, recursive=True)
        get_dir_hash_code
        root_path = Path(dir_path)
        for file_path in file_path_list:
            cur_path = Path(file_path)
            zipf.write(file_path, Path(dir_name) / cur_path.relative_to(root_path))

def unzip_dir(zip_file_path, dir_path): # 解压文件夹
    import zipfile as zipfile

    assert os.path.isfile(zip_file_path), f'zip_file_path {zip_file_path} is not a file'
    assert dir_check(dir_path), f'dir_path {dir_path} create_failed exist'

    with zipfile.ZipFile(zip_file_path, 'r') as zipf:
        zipf.extractall(dir_path)
        

class MyEncoder(json.JSONEncoder):
    """
    自定义序列化方法, 解决 TypeError - Object of type xxx is not JSON serializable 错误
    使用方法:
    在json.dump时加入到cls中即可, 例如:
    json.dumps(data, cls=MyEncoder) 
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, Path) or isinstance(obj, Path2):
            return str(obj)
        elif isinstance(obj, decimal.Decimal):
            return float(obj)
        elif isinstance(obj, datetime.datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        else:
            return super(MyEncoder, self).default(obj)


class Sys_Logger(object):
    '''
    修改系统输出流
    '''

    def __init__(self, fileN="Default.log"):

        self.terminal = sys.stdout
        if OS_exists(fileN):
            self.log = open(fileN, "a")
        else:
            self.log = open(fileN, "w")

    def write(self, message):

        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()


class Loger_printer():
    """
    日志打印类
    会在控制台与日志同时打印信息    
    """

    def __init__(self, logger):
        self.logger = logger
        self.logger_dict = {
            logging.DEBUG: self.logger.debug,
            logging.INFO: self.logger.info,
            logging.WARNING: self.logger.warning,
            logging.ERROR: self.logger.error,
            logging.CRITICAL: self.logger.critical
        }
        self.level = self.logger.level
        self.closed = False

    def get_logger_func(self, level):
        if level in self.logger_dict:
            return self.logger_dict[level]
        else:
            raise RuntimeError(f"unknown level {level}, {self.logger_dict}")

    def vvd_logging(self, *message, level=logging.INFO, close_logger=False):
        if not self.closed:
            log_func = self.get_logger_func(level)
            if message is not None:
                for message_str in message:
                    if level >= self.level:
                        print(message_str)
                    log_func(message_str)
            if close_logger:
                for handler in self.logger.handlers:
                    handler.close()
        else:
            print('logger has been closed')

    def vvd_logging_quiet(self, *message, level=logging.INFO, close_logger=False):
        if not self.closed:
            log_func = self.get_logger_func(level)
            if message is not None:
                for message_str in message:
                    log_func(message_str)
            if close_logger:
                for handler in self.logger.handlers:
                    handler.close()
        else:
            print('logger has been closed')

def get_date_str():
    str_time = time.localtime(time.time())
    year = str_time.tm_year
    mounth = str_time.tm_mon
    day = str_time.tm_mday

    date_str = f"{year}-{mounth}-{day}"
    return date_str


def log_init(log_path, quiet=False, level=logging.INFO):
    """
    initialize logging 
    save the logging object in `config.Parameters.Logging_Object`

    after this operation,
    we could save logs with simple orders such as `logging.debug('test debug')` `logging.info('test info')` 
    logging level : debug < info < warning <error < critical

    Loger_printer.vvd_logging('test')
    """
    log_path= str(log_path)
    dir_name = os.path.dirname(log_path)

    dir_check(dir_name)
    log_file_path = log_path

    if os.path.exists(log_file_path):
        # open log file as  mode of append
        open_type = 'a'
    else:
        # open log file as  mode of write
        open_type = 'w'

    # logging.basicConfig(
    #     日志级别,logging.DEBUG,logging.ERROR
    #     level=level,
    #     # 日志格式: 时间、   日志信息
    #     format='%(asctime)s: %(message)s',
    #     # 打印日志的时间
    #     datefmt='%Y-%m-%d %H:%M:%S',
    #     # 日志文件存放的目录（目录必须存在）及日志文件名
    #     filename=log_file_path,
    #     # 打开日志文件的方式
    #     filemode=open_type
    # )
    # logging.StreamHandler()

    # create logger obj
    logger = logging.getLogger()
    # set log level
    logger.setLevel(level)
    # file handler 
    handler = logging.FileHandler(log_file_path, mode=open_type, encoding='utf-8')
    handler.setFormatter(logging.Formatter("%(asctime)s-%(name)s-%(levelname)s: %(message)s"))

    for old_handler in logger.handlers[::-1]:
        old_handler.stream.close()
        logger.removeHandler(old_handler)

    logger.addHandler(handler)

    if quiet:
        return Loger_printer(logger).vvd_logging_quiet
    else:
        return Loger_printer(logger).vvd_logging


def isdir(dir_path):
    """
    check if dir exists
    """
    dir_path = str(dir_path)
    if not os.path.isdir(dir_path):
        return False
    else:
        return True


def uniform_split_char(string, split_char='/'):
    """
    uniform the split char of a string
    """
    string = str(string)
    assert isinstance(string, str)
    return string.replace('\\', split_char).replace('/', split_char)


def dir_check(dir_path, verbose=False):
    """
    check if `dir_path` is a real directory path
    if dir not found, make one
    """

    dir_path = str(dir_path)
    if dir_path == '':
        return True
    assert isinstance(dir_path, str)
    dir_path = uniform_split_char(dir_path)
    if not os.path.isdir(dir_path):
        try:
            os.makedirs(dir_path)
            if verbose:
                print('dirs made: {}'.format(dir_path))
            return True
        except Exception as err:
            print(f'failed to make dir {dir_path}, error {err}')
            return False
    else:
        return True


def time_reduce(*data):
    """
    [计算输入数据的乘积]
    """
    data = list(data)
    return reduce(lambda x, y: x*y, data)


def get_function_name():
    '''获取正在运行函数(或方法)名称'''
    # print(sys._getframe().f_code.co_name)
    return inspect.stack()[1][3]


def draw_RG_map(y_true, y_pred, map_save_path=None):
    """draw Red Green map for a prediction
    the predicted data are sorted by score
    Correctly classified samples are marked in green
    The misclassified sample is marked in red

    Args:
        y_true (list or narray): ground truth
        y_pred (list or narray): prediction
        map_save_path (path): path to the output image
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    assert isinstance(y_pred, np.ndarray) and isinstance(y_true, np.ndarray)
    assert np.ndim(y_pred) == 1
    assert y_pred.shape == y_true.shape

    sorted_ids = np.argsort(y_pred+np.random.rand(y_pred.size)*1e-12)
    sorted_y_true = y_true[sorted_ids]
    ng_rank = np.where(sorted_y_true == 1)[0]
    ok_rank = np.where(sorted_y_true == 0)[0]

    plt.figure(figsize=(25, 3))
    plt.bar(ok_rank, 1, width=1, color='g')
    plt.bar(ng_rank, 1, width=int(ok_rank.size/ng_rank.size/5+1), color='r')
    plt.ylim([0, 1])
    plt.xlim([0, len(ok_rank)+len(ng_rank)])
    if map_save_path is not None:
        plt.savefig(map_save_path)
    else:
        plt.show()

        plt.figure(figsize=(25, 3))
        plt.hist(y_pred, bins=255)
        plt.title('ng_score distribution')
        plt.show()


def histogram(*data, bin_num=100, title=None, show=True):
    data_list = list(data)
    value, bins = np.histogram(data_list, bins=bin_num)
    for data_item in data_list:
        plt.hist(data_item, bin_num)
    if title is not None:
        plt.title(str(title))
    if show:
        plt.show()
    return value, bins


def data_show(data):
    '''
    show data in a chart
    '''
    plt.plot(data)
    plt.show()


def is_path_obj(path):
    """
    Determines whether the input path is a Path obj
    """
    if isinstance(path, Path) or isinstance(path, Path2):
        return True
    else:
        return False


def time_stamp():
    """
    返回当前时间戳字符串
    格式: 年-月-日_时-分-秒
    """
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

def get_suffix(path):
    """get suffix of a path

    Args:
        path (_type_): _description_
    """
    cur_path = Path(path)
    suffix = cur_path.suffix.replace('.', '')
    return suffix

def change_file_name_for_path(file_path, new_name=None, new_suffix=None):
    """change file name for a file path
    This operation is valid only for path, NOT for REAL FILE
    

    Args:
        file_path (path): a file path
        new_name (str): new file name
        new_suffix(str): new suffix, if None, old suffix will be kept

    Returns:
        new file path(str)
    """
    path = Path(str(file_path))
    parent = path.parent
    
    if new_name is None:
        new_file_stem = path.stem
    else:
        new_file_stem = new_name

    if new_suffix is not None:
        if new_suffix == '':
            new_file_suffix = ''
        else:
            if new_suffix[0] != '.':
                new_file_suffix = '.' + new_suffix
            else:
                new_file_suffix = new_suffix
    else:
        new_file_suffix = path.suffix

    new_file_name = new_file_stem + new_file_suffix
    new_path = parent / new_file_name
    new_path = uniform_split_char(str(new_path))
    return new_path

def smart_copy(source_file_path, target_path, verbose=False, remove_source_file=False, overwrite=False, keep_all_files=False, build_dir=False):
    """[复制文件从源到目标, 如果目标已经存在则跳过(默认), 如果 keep_all_files 为 True 会保留所有文件]]

    Args:
        source_file_path ([str]): [源文件路径]
        target_path ([str]): [目标文件夹/目标文件路径]
        verbose (bool, optional): [是否显示信息]. Defaults to False.
    """
    source_file_path = str(source_file_path).replace('\\', '/')
    target_path = str(target_path).replace('\\', '/')
    assert OS_exists(source_file_path)

    if build_dir:
        dir_check(target_path)

    if OS_isdir(target_path):
        target_path = OS_join(target_path, OS_basename(source_file_path))

    exists = OS_exists(target_path)
    if exists and not overwrite:
        if keep_all_files:
            target_path = save_file_path_check(target_path, False, False)
            shutil.copy(source_file_path, target_path)
        elif verbose:
            print("{} already exists! skip copying.".format(target_path))
    else:
        dir_check(Path(target_path).parent)
        if exists and overwrite:
            if verbose:
                print("{} already exists! start overwriting.".format(target_path))
        if remove_source_file:
            shutil.move(source_file_path, target_path)
        else:
            shutil.copy(source_file_path, target_path)
    return target_path


def json_load(json_path, verbose=False):
    """
    读取json文件并返回内容字典
    """
    json_path = str(json_path)
    if isinstance(json_path, str):
        json_path = json_path.replace('\\', '/')
    try:
        assert OS_exists(json_path)
    except Exception as e:
        if verbose:
            print('file not found !', e)
    try:
        with open(json_path, 'r') as fp:
            return json.load(fp)
    except Exception as e:
        if verbose:
            print('simple json load failed, try utf-8', e)
    try:
        with open(json_path, 'r', encoding='utf-8') as fp:
            return json.load(fp)
    except Exception as e:
        if verbose:
            print('utf-8 json load failed, try gbk', e)
    try:
        with open(json_path, 'r', encoding='gbk') as fp:
            return json.load(fp)
    except Exception as e:
        if verbose:
            print('gbk json load failed!', e)        


def json_save(json_dict, json_path, overwrite=False, verbose=False):
    """
    将内容字典保存为json文件
    """
    json_path = str(json_path)
    json_path = save_file_path_check(json_path, overwrite, verbose)
    with open(json_path, 'w', encoding='utf-8') as fp:
        json.dump(json_dict, fp, ensure_ascii=False, sort_keys=False, indent=4, cls=MyEncoder)


def glob_recursively(path, extensions, recursively=True):
    """
    extensions 为 extension 的列表，例如 ['png', 'jpeg'], 函数会返回两种格式的数据
    如果输入的数据为字符串，会自动转换为列表
    在path 路径中递归查找所有扩展名为 extensions 的文件, 返回完整路径名列表
    """
    path = str(path).replace('\\', '/')

    if not isinstance(extensions, list):
        extensions = [extensions]

    file_list = list()
    for extension in extensions:
        if recursively:
            file_list += glob(OS_join(path, '**', '*.' + extension), recursive=True)
        else:
            file_list += glob(OS_join(path, '*.' + extension), recursive=True)
    return file_list


popular_image_extensions = ['png', 'jpeg', 'bmp', 'jpg', 'PNG', 'JPEG', 'JPG', 'BMP', 'tif', 'TIF']
def glob_images(path, recursively=True):
    """
    在 glob_recursively 基础上进行封装，获取 path 路径下常见图像格式的图像
    extensions = ['png', 'jpeg', 'bmp', 'jpg', 'PNG', 'JPEG', 'JPG', 'BMP']
    返回完整路径名列表
    """
    
    return  glob_recursively(path, popular_image_extensions, recursively=recursively)


def glob_videos(path, recursively=True):
    """
    在 glob_recursively 基础上进行封装，获取 path 路径下常见视频格式的视频
    videoSuffixSet = {"WMV","ASF","ASX","RM","RMVB","MP4","3GP","MOV","M4V","AVI","DAT","MKV","FIV","VOB"}
    返回完整路径名列表
    """
    popular_video_extensions = ['mp4', 'avi', 'mov', 'MP4', 'AVI', 'MOV']
    return  glob_recursively(path, popular_video_extensions, recursively=recursively)


def find_all_numbers(string):
    """
    找出字符串中所有的数字，返回一个列表(科学计数法的不行)
    """
    numbers = re.findall(r'-?\d+\.?\d*', string)
    return get_list_from_list(numbers, lambda x: float(x))


def is_integer(num):
    """
    是否是整数, 返回bool结果
    """
    from numbers import Integral
    if isinstance(num, np.ndarray):
        return np.issubdtype(num.dtype, np.integer)
    else:
        return isinstance(num, Integral)


def is_float(num):
    """
    是否是浮点数, 返回bool结果
    """
    if isinstance(num, np.ndarray):
        return np.issubdtype(num.dtype, np.floating)
    else:
        return isinstance(num, float)


def is_number(num):
    """
    是否是数字, 此处暗指实数, 返回bool结果
    """
    return is_float(num) or is_integer(num)


def is_bool(data):
    """
    是否是 bool 值, 返回bool结果
    """
    if isinstance(data, np.ndarray):
        return np.issubdtype(data.dtype, np.bool_)
    return isinstance(data, bool)


def whether_divisible_by(to_be_divided, dividing):
    """
    to_be_divided 是否可以被 dividing 整除, 返回bool结果
    """
    assert is_integer(to_be_divided) and is_integer(dividing)
    if to_be_divided % dividing == 0:
        return True
    else:
        return False


def vvd_round(num):
    if iterable(num):
        return np.round(np.array(num)).astype('int32').tolist()
    return int(np.round(num))

round = vvd_round

def vvd_ceil(num):
    if iterable(num):
        return np.ceil(np.array(num)).astype('int32').tolist()
    return int(np.ceil(num))

ceil = vvd_ceil

def vvd_floor(num):
    if iterable(num):
        return np.floor(np.array(num)).astype('int32').tolist()
    return int(np.floor(num))

floor = vvd_floor

def get_gpu_str_as_you_wish(gpu_num_wanted, step = 1, verbose=0):
    """[get empty gpu index str as needed]
    Args:
        gpu_num_wanted ([int]): [the num of gpu you want]
    Returns:
        [str]: [gpu str returned]
    """
    try:
        import pynvml
    except Exception as e:
        print('can not import pynvml.', e)
        print('please make sure pynvml is installed correctly.')
        print('a simple pip install nvidia-ml-py3 may help.')
        print('now a 0 will be return')
        return '0', [0]
    NUM_EXPAND = 1024 * 1024
    try:
        # 初始化工具
        pynvml.nvmlInit()
    except Exception as e:
        print('pynvml.nvmlInit failed:', e)
        print('now a 0 will be return')
        return '0', [0]
    # 驱动信息
    if verbose:
        print("GPU driver version: ", pynvml.nvmlSystemGetDriverVersion())
    # 获取Nvidia GPU块数
    gpuDeviceCount = pynvml.nvmlDeviceGetCount()
    returned_gpu_num = max(min(gpu_num_wanted, gpuDeviceCount), 0)
    gpu_index_and_free_memory_list = list()
    for index in range(gpuDeviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_memory_total = info.total / NUM_EXPAND
        gpu_memory_used = info.used / NUM_EXPAND
        gpu_index_and_free_memory_list.append([index, gpu_memory_total - gpu_memory_used])
        if verbose:
            if index == 0:
                device_name = pynvml.nvmlDeviceGetName(handle)
                print('GPU device name:', device_name)
            print(f'gpu {index}: total_memory: {gpu_memory_total} memory_left: {gpu_memory_total - gpu_memory_used}')
    import random
    random.shuffle(gpu_index_and_free_memory_list)
    gpu_index_and_free_memory_list.sort(key=lambda x: - x[1]//step)
    gpu_index_picked_list = list()
    for i in range(returned_gpu_num):
        gpu_index_picked_list.append(gpu_index_and_free_memory_list[i][0])
    gpu_index_str = ','.join([str(index) for index in gpu_index_picked_list])
    if verbose:
        print(returned_gpu_num, 'gpu index will be return.')
        print(f'return gpu str: {gpu_index_str}')
    # 关闭工具
    pynvml.nvmlShutdown()
    return gpu_index_str, gpu_index_picked_list

def get_single_gpu_id(step=1, verbose=False):
    id_str, id_list = get_gpu_str_as_you_wish(1, verbose=verbose)
    gpu_id = id_list[0]
    return gpu_id

def get_dir_file_list(root_path, recursive=True, leaf_dir=False):
    """[get dir and file list under root_path recursively]

    Args:
        root_path ([str]): [root dir to querry]
        recursive ([bool]): [whether walk recursively]

    Returns:
        dir_list [list]: [output dir list]
        file_list [list]: [output file list]
    """

    dir_list = list()
    file_list = list()

    root_path = str(root_path)

    for root, dirs, files in os.walk(root_path):
        file_list += get_list_from_list(files, lambda x: os.path.join(root, x))
        if not leaf_dir:
            for dir in dirs:
                cur_dir_path = os.path.join(root, dir)
                dir_list.append(cur_dir_path)
        else:
            if not dirs:
                dir_list.append(root)
        if not recursive:
            break
    dir_list = get_list_from_list(dir_list, uniform_split_char)
    file_list = get_list_from_list(file_list, uniform_split_char)
    return dir_list, file_list


def get_segments(data):
    """
    get segments for data (for moved safe data)
    """
    data = np.array(data)
    assert data.ndim == 1
    data = (data > 0).astype('int8')
    mark = data[:-1] - data[1:]
    start_pos = np.nonzero(mark == -1)[0].tolist()
    end_pos = np.nonzero(mark == 1)[0].tolist()
    if data[0] > 0:
        start_pos = [-1] + start_pos
    if data[-1] > 0:
        end_pos = end_pos + [len(mark)]
    assert len(start_pos) == len(end_pos)
    segments_list = [[x + 1, y] for x, y in zip(start_pos, end_pos)]
    return segments_list


def try_exc_handler(try_func, exc_func, developer_mode=False):
    except_result = try_result = None

    def exception_handler(e, exc_func):
        try:
            ori_exception_info = list(e.args)
            if len(ori_exception_info) == 0:
                ori_exception_info.append('')
            exception_str = ' ErrorMessage: ' + str(ori_exception_info[0])
            if hasattr(e, '__traceback__'):
                cur_trace_obj = e.__traceback__
                level = 1
                while True:
                    if not hasattr(cur_trace_obj, 'tb_next'):
                        break
                    if level > 300:
                        break
                    frame_str = str(cur_trace_obj.tb_frame)
                    frame_str = frame_str[max(0, frame_str.find('file')):]
                    res_str =  '\n' + str(level) + ': ' + frame_str
                    exception_str += res_str
                    cur_trace_obj = cur_trace_obj.tb_next
                    level += 1

            ori_exception_info[0] = exception_str
            e.args = tuple(ori_exception_info)
            except_result = exc_func(e)
        except Exception as ee:
            print("!! exc_func failed.")
            print(f"!! error message: {str(ee)}")
            print("!! we can only return the previous exception info.")
            except_result = str(e)
        return except_result

    if developer_mode:
        try_result = try_func()
    else:
        try:
            try_result = try_func()
        except Exception as e:
            return exception_handler(e, exc_func)
        except FunctionTimedOut as e:
            return exception_handler(e, exc_func)

    return try_result



def class_timer(input_class):
    """
    get a new class through a Decorator
    new class will Time all funcs of origin class

    Args:
        input_class (class): class to be timed

    Returns:
        class: timer class
    """
    class Timer(input_class):
        def __getattribute__(self, name: str):

            func = super().__getattribute__(name)

            if str(type(func)) == "<class 'method'>":
                is_static_method = False
                try :
                    func_name = func.__name__
                except Exception as e:
                    func_name = func.__func__.__name__
                    func = func.__func__
                    is_static_method = True

                @wraps(func)
                def wrapper(*args, **kwargs):
                    if is_static_method:
                        args = args[1:]

                    start_time = time.time()
                    res = func(*args, **kwargs)
                    end_time = time.time()
                    print('func: {_funcname_} runing: {_time_}s'.format(_funcname_=func_name, _time_=format(end_time - start_time, '.6f')))
                    return res
                return wrapper
            return func
    return Timer


def remove_small_components(mask, area_thres):
    """ remove connected components from mask if their areas < area_thres """
    nccomps = cv.connectedComponentsWithStats(mask.astype(np.uint8))

    labels = nccomps[1]
    status = nccomps[2]

    for index, st in enumerate(status.tolist()):
        if st[0] == st[1] == 0:
            continue
        area = st[4]
        if area >= area_thres:
            labels[labels == index] = - index

    return labels < 0


def encode_path(path, seg_num = 4, key_word=None, sep_char='@', root_path=None, with_suffix=None, stem=False):
    """Change path to a string without slash

    Args:
        path (path): path to be encoded
        seg_num (int, optional): The num of subpath segments that need to be encoded. Defaults to 4.
        key_word (string, optional): if set(not None), the path will be truncated after the keyword. Defaults to None.
        sep_char (str, optional): a string who will concatenate subpaths. Defaults to '@'.
        root_path (str, optional): if set, a encoded file path under root_path will be return . Defaults to None.
        with_suffix (str, optional): if set, result path will use the input suffix. We both support suffixes start with . or not. Defaults to None.
        stem (bool, optional): if True, only stem of encoded path will be returned. Defaults to False.

    Returns:
        str: encoded path
    """
    path = uniform_split_char(str(path))
    if key_word is not None:
        pos = str(path).find(key_word)
        if pos == '-1':
            print('keyword encode path failed, use default function')
            key_word = None
        else:
            path = str(path)[pos:]
            tmp_path = Path(path)
            encoded_path = sep_char.join(tmp_path.parts)

    if key_word is None:
        tmp_path = Path(path)
        encoded_path = sep_char.join(tmp_path.parts[-seg_num:])

    if root_path is not None:
        encoded_path = str(Path(root_path) / encoded_path)

    if with_suffix is not None:
        with_suffix = str(with_suffix)
        if with_suffix[0] != '.':
            with_suffix = '.' + with_suffix
        encoded_path = str(Path(str(encoded_path)).with_suffix(with_suffix))

    if stem:
        encoded_path = Path(encoded_path).stem

    return uniform_split_char(encoded_path)


def extract_file_to(root_dir, suffix, target_dir, encode_level=3):
    """copy files with specified suffix in root dir to target dir
    new file name is encoded oirgin file path

    Args:
        root_dir (path): root dir
        suffix (str): suffix without .
        target_dir (path): target dir
        encode_level (int, optional): path encode level. Defaults to 3.
    """
    file_list = glob_recursively(root_dir, suffix)
    desc = f"Extracting {suffix} file from {root_dir} to {target_dir}:"
    for file_path in tqdm(file_list, desc=desc):
        file_name = encode_path(file_path, seg_num=encode_level)
        save_path = Path(target_dir) / file_name
        smart_copy(file_path, save_path)


def link(src, tar, hard=False):
    """build link for src file to tar file

    Args:
        src (path): source file
        tar (path): target file
        hard (bool): hard link if True, else symlink

    Returns:
        bool: if success return True, else False
    """

    if osp.exists(tar):
        return False
    else:
        if osp.exists(src):
            if hard:
                if not isdir(src):
                    os.link(src, tar)
                else:
                    return False
            else:
                os.symlink(src, tar)
        else:
            return False
    return True

def islink(path):
    """Test whether a path is a symbolic link"""
    return os.path.islink(path)

def get_cwd():
    """get current working directory"""
    return uniform_split_char(os.getcwd())

def hardlink(src, tar):
    """
    make hardlink for src and tar
    """
    return link(src, tar, True)

def symlink(src, tar):
    """
    make sumlink for src and tar
    """
    return link(src, tar, False)


class ListOrderedDict(OrderedDict):
    def __getitem__(self, key):
        if is_integer(key):
            key = self.get_key(key)
        elif isinstance(key, slice):
            slicedkeys = list(self.keys())[key]
            lod_obj = type(self)()
            for slicedkey in slicedkeys:
                lod_obj[slicedkey] = self[slicedkey]
            return lod_obj

        return super().__getitem__(key)

    def get_key(self, index):
        return list(self.keys())[index]

    def get_index(self, key):
        return list(self.keys()).index(key)

    def append(self, key, value):
        if is_integer(key):
            raise RuntimeError(f" integer key is not allowed in ListOrderedDict {key}.")
        assert key not in self, f"cannot add existed key. {key}"
        self[key] = value
    
    def __setitem__(self, key, value):
        'od.__setitem__(i, y) <==> od[i]=y'
        # Setting a new item creates a new link at the end of the linked list,
        # and the inherited dictionary is updated with the new key/value pair.
        if is_integer(key):
            raise RuntimeError(f" integer key is not allowed in ListOrderedDict {key}.")
        else:
            return super().__setitem__(key, value)

    def __delitem__(self, key):
        'od.__delitem__(y) <==> del od[y]'
        # Deleting an existing item uses self.__map to find the link which gets
        # removed by updating the links in the predecessor and successor nodes.
        if is_integer(key):
            key = list(self.keys())[key]

        return super().__delitem__(key)

    def __iter__(self): 
        index = 0
        while index < len(self):
            yield self[index]      # 使用 yield
            index = index + 1

    def move_to_end(self, key, last=True):
        '''Move an existing element to the end (or beginning if last is false).

        Raise KeyError if the element does not exist.
        '''
        if is_integer(key):
            key = list(self.keys())[key]

        return super().move_to_end(key, last)

    def __repr__(self):
        'od.__repr__() <==> repr(od)'
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self.items()))

    def pop(self, key=-1):
        '''od.pop(k[,d]) -> v, remove specified key and return the corresponding
        value.  If key is not found, d is returned if given, otherwise KeyError
        is raised.
        '''
        if is_integer(key):
            if len(list(self.keys())) == 0:
                print(f"pop a None from an empty ListOrderedDict.")
                return None
            key = list(self.keys())[key]
        return super().pop(key)

    def setdefault(self, key, default=None):
        '''Insert key with a value of default if key is not in the dictionary.

        Return the value for key if key is in the dictionary, else default.
        '''
        if is_integer(key):
            key = list(self.keys())[key]
        return super().setdefault(key, default)

def is_valid_identifier(input_str):
    """ Test whether a str is a identifier"""
    ascii = string.ascii_letters+'_'
    digits = string.digits
    if not isinstance(input_str, str):
        return False
    if len(input_str) == 0:
        return False
    if input_str[0] not in ascii:
        return False
    for char in input_str:
        if char not in ascii and char not in digits:
            return False
    return True


def chinese_str_to_pinyin(input_str: str, join_char='_') -> str:
    """change str with chinese char into pinyin chars

    Args:
        input_str (str): input string
        join_char (str, optional): string to split pinyin. Defaults to '_'.

    Returns:
        str: processed string
    """
    import pypinyin
    pinyin_str =  pypinyin.pinyin(input_str, style=pypinyin.NORMAL)
    pinyin_str_list = get_list_from_list(pinyin_str, lambda x: x[0])
    pinyin_res_str = join_char.join(pinyin_str_list)
    output_str = encode_chinese_to_unicode(pinyin_res_str, remove=True)
    return output_str


def remove_chinese_for_file_names(root_dir):
    """change all filenames of files in root_dir which have chinese char in file_name to pinyin names"""
    dir_list, file_list = get_dir_file_list(root_dir, recursive=True)
    file_name_changed = 0
    for file_path in tqdm(file_list, desc=f"changing file names: "):
        file_name = Path(file_path).name
        changed_name = chinese_str_to_pinyin(file_name)
        if file_name != changed_name:
            dst_path = Path(file_path).parent / changed_name
            file_rename(file_path, dst_path)
            file_name_changed += 1
    print(f"change {file_name_changed} file names.")
    
    dir_name_changed = 0
    for dir_path in tqdm(dir_list[::-1], desc=f"changing directory names: "):
        dir_name = Path(dir_path).name
        changed_name = chinese_str_to_pinyin(dir_name)
        if dir_name != changed_name:
            dst_path = Path(dir_path).parent / changed_name
            file_rename(dir_path, dst_path)
            dir_name_changed += 1
    print(f"change {dir_name_changed} dir names.")
    pass


def is_generator(obj):
    from inspect import isgenerator
    return isgenerator(obj)


def is_iterable(obj):
    from typing import Iterable
    return isinstance(obj, Iterable)


def get_file_time(file_path):
    """get file time record"""
    atime = os.path.getatime(file_path)  # 文件访问时间
    mtime = os.path.getmtime(file_path)  # 文件最近修改时间
    ctime = os.path.getctime(file_path)  # 文件的创建时间

    time_info = dict(
        access_time=time.localtime(atime),
        modify_time=time.localtime(mtime),
        create_time=time.localtime(ctime)
    )

    return time_info


def find_lcsubstr(s1: str, s2: str): 
    """
    Longest Common Substring
    最长公共子串 (连续串, 非序列)
    O(mn)

    Args:
        s1 (str): string 1
        s2 (str): string 2

    Returns:
        str: longest_sub_string, max_length
    """
    s1 = str(s1)
    s2 = str(s2)
    # 生成0矩阵，为方便后续计算，比字符串长度多了一列
    m = np.zeros([len(s1) + 1, len(s2) + 1], dtype='uint16')
    mmax = 0   # 最长匹配的长度
    p = 0  # 最长匹配对应在s1中的最后一位
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i+1][j+1] = m[i][j] + 1
                if m[i+1][j+1] > mmax:
                    mmax = m[i+1][j+1]
                    p = i+1
    return s1[p-mmax:p], mmax   # 返回最长子串及其长度


def find_lcseque(s1: str, s2: str):
    """
    Longest Common Subsequence
    最长公共子序列(不一定连续)
    O(mn)

    Args:
        s1 (str): string 1
        s2 (str): string 2

    Returns:
        str: Subsequence_str
    """
    #  生成字符串长度加1的0矩阵，m用来保存对应位置匹配的结果
    m = np.zeros([len(s1)+1, len(s2)+1], dtype='uint16')

    #  d用来记录转移方向
    d = np.zeros([len(s1)+1, len(s2)+1], dtype='uint8')

    for p1 in range(len(s1)): 
        for p2 in range(len(s2)): 
            if s1[p1] == s2[p2]:            # 字符匹配成功，则该位置的值为左上方的值加1
                m[p1+1][p2+1] = m[p1][p2]+1
                d[p1+1][p2+1] = 1          
            elif m[p1+1][p2] > m[p1][p2+1]:  # 左值大于上值，则该位置的值为左值，并标记回溯时的方向
                m[p1+1][p2+1] = m[p1+1][p2] 
                d[p1+1][p2+1] = 2         
            else:                           # 上值大于左值，则该位置的值为上值，并标记方向up
                m[p1+1][p2+1] = m[p1][p2+1]   
                d[p1+1][p2+1] = 3       

    (p1, p2) = (len(s1), len(s2)) 

    s = [] 
    while m[p1][p2]:    # 不为None时
        c = d[p1][p2]
        if c == 1:   # 匹配成功，插入该字符，并向左上角找下一个
            s.append(s1[p1-1])
            p1 -= 1
            p2 -= 1 
        if c == 2:  # 根据标记，向左找下一个
            p2 -= 1
        if c == 3:   # 根据标记，向上找下一个
            p1 -= 1

    s.reverse() 
    return ''.join(s) 

def random_string(length, digits=True, letters=True, punctuation=True):
    """make random string via specific char sets

    Args:
        length (int): length of target random string
        digits (bool, optional): if True, random string will contain digits. Defaults to True.
        letters (bool, optional): if True, random string will contain letters. Defaults to True.
        punctuation (bool, optional): if True, random string will contain punctuations. Defaults to True.

    Returns:
        _type_: _description_
    """
    str_temp = ''
    if digits:
        str_temp = str_temp + string.digits
    if letters:
        str_temp = str_temp + string.ascii_letters
    if punctuation:
        str_temp = str_temp + string.punctuation
    return ''.join(random.choice(str_temp) for _ in range(length))


def remove_file(path):
    """remove a file or a symlink"""
    path = str(path)
    if OS_exists(path):
        if OS_isfile(path) or islink(path):
            os.remove(path)

def get_path_stem(path):
    return Path(path).stem

def path_with_suffix(path, suffix):
    if suffix[0] != '.':
        suffix = '.' + suffix
    return uniform_split_char(Path(path).with_suffix(suffix))

def path_with_name(path, name):
    return uniform_split_char(Path(path).with_name(name))

def path_insert_content(path, content):
    path_obj = Path(path)
    suffix = path_obj.suffix
    parent = path_obj.parent
    stem = path_obj.stem
    return uniform_split_char(parent / (stem + content + suffix))

def get_path_parent(path):
    return uniform_split_char(Path(path).parent)

def get_path_name(path):
    return Path(path).name

def strip_path_slash(path):
    return uniform_split_char(path).strip('/')

def remove_dir(path):
    """remove a dir or a symlink"""
    path = str(path)
    if OS_exists(path):
        if islink(path):
            os.remove(path)
        else:
            shutil.rmtree(path)


def display_class_property(class_def, verbose=False):
    """Display Configuration values."""
    if verbose:
        print("\n")
        print("Configurations:")
    property_dict = dict()
    for property in dir(class_def):
        if not property.startswith("__") and not callable(getattr(class_def, property)):
            property_dict[property] = getattr(class_def, property)
            if verbose:
                print("{:50} {}".format(property, getattr(class_def, property)))
    if verbose:
        print("\n")
    return property_dict


class TickTock(object):
    def __init__(self):
        self._timestamps = list()
        self.begin()

    def begin(self):
        info = dict(
            timestamp = time.time(),
            begin_flag = True,
            message = None,
        )
        self._timestamps.append(info)

    def step(self, message='Time Cost'):
        assert len(self._timestamps) > 0
        assert self._timestamps[0]['begin_flag'], \
            "you need to run begin() before step()"
        info = dict(
            timestamp = time.time(),
            begin_flag = False,
            message = message,
        )
        self._timestamps.append(info)

    end = step

    def show(self):
        if len(self._timestamps) > 1:
            print("\nTimer Record:")
            print("----------------")
            total = 0
            for istep, (info_prev, info_curr) in enumerate(zip(self._timestamps[:-1], self._timestamps[1:])):
                if not info_curr['begin_flag']:
                    interval = info_curr['timestamp'] - info_prev['timestamp']
                    message  = info_curr['message']
                    print("[{:02d}] {}: {:.4f}s".format(istep, message, interval))
                    total += interval
            print("Total: {:.4f}s".format(total))
            print("----------------")


def line_cross(min1, max1, min2, max2):
    min_v = max(min1, min2)
    max_v = min(max1, max2)
    cross_length = max(0, max_v - min_v)
    cross = max_v - min_v >= 0
    return (min_v, max_v), cross_length, cross


def epsilon_equal(a, b, epsilon=1e-8):
    return np.abs(a - b) < epsilon


def line_merge(line_list):
    line_1D_list = list()
    for line in line_list:
        min_v, max_v = line
        if  min_v < max_v:
            Add = True
            for index, roi_1D in enumerate(line_1D_list):
                (merge_min, merge_max), cross_length, cross = line_cross(roi_1D[0], roi_1D[1], min_v, max_v)
                if cross:
                    assert Add
                    Add = False
                    line_1D_list[index] = [min(roi_1D[0], min_v), max(roi_1D[1], max_v)]
            if Add:
                line_1D_list.append([min_v, max_v])
    return line_1D_list

def parse_time_str(time_str):
    try:
        time_obj = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    except:
        time_obj = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")
    return time_obj

def make_time(year=None, month=None, day=None, hour=None, minute=None, second=None):

    now = datetime.datetime.now()

    if year is None:
        year = now.year

    if month is None:
        month = now.month

    if day is None:
        day = now.day

    if hour is None:
        hour = now.hour

    if minute is None:
        minute = now.minute

    if second is None:
        second = now.second

    # make time
    time_str = datetime.datetime(year, month, day, hour, minute, second)
    return time_str


def time_string(millisecond=False, microsecond=False, simple_str=False, year_month_day=False, datetime_obj=None):

    if datetime_obj is None:
        datetime_obj = datetime.datetime.now()

    if microsecond:
        time_str = datetime_obj.strftime("%Y-%m-%d %H:%M:%S.%f")
    elif millisecond:
        time_str = datetime_obj.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    else:
        time_str = datetime_obj.strftime("%Y-%m-%d %H:%M:%S")
    
    if year_month_day:
        time_str = datetime_obj.strftime("%Y-%m-%d")

    if simple_str:
        time_str = time_str.replace(":", "-").replace(" ", "_")

    return time_str


def random_sample(data, sample_num, remove_data=False):
    sample_num = int(sample_num)
    assert sample_num > 0 and sample_num <= len(data)

    if not remove_data:
        return random.sample(data, sample_num)
    else:
        # return data not sampled
        collection_list = list()
        picked_index_list = random.sample(range(len(data)), sample_num)
        picked_index_list.sort(reverse=True)
        for index in picked_index_list:
            collection_list.append(data.pop(index))
        return collection_list


def load_xml(xml_path, disable_entities=True):
    import xmltodict
    with open(xml_path, 'r', encoding='utf-8') as xml_file:
        xml_data = xml_file.read()
    dict_data = xmltodict.parse(xml_data, disable_entities=disable_entities)
    return dict_data


def save_xml(dict_data, xml_path, overwrite=True, verbose=False):
    import xmltodict
    xml_data = xmltodict.unparse(dict_data, pretty=True)
    file_path = save_file_path_check(xml_path, overwrite, verbose)
    with open(file_path, 'w', encoding='utf-8') as xml_file:
        xml_file.write(xml_data)


def get_file_line_number(file_path):
    """Count the number of lines in a file"""
    return sum(1 for line in open(file_path, 'r'))


def get_file_create_time(file_path):
    # 获取文件的创建时间
    creation_time = os.path.getctime(file_path)
    creation_time_formatted = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(creation_time))
    return creation_time_formatted


def sleep_count(seconds, minute=0, hour=0, verbose=True):
    "time sleep"
    total_seconds = seconds + minute * 60 + hour * 3600
    now_time_value = time.time()
    target_time_value = now_time_value + total_seconds

    tar_object = datetime.datetime.fromtimestamp(target_time_value)
    tar_datetime = tar_object.strftime('%H:%M:%S')

    verbose_index = 1

    while True:
        time.sleep(0.01)
        cur_time_value = time.time()
        if cur_time_value - now_time_value >= verbose_index:
            if verbose:
                dt_object = datetime.datetime.fromtimestamp(cur_time_value)
                formatted_datetime = dt_object.strftime('%H:%M:%S')

                time_left = round(target_time_value - cur_time_value)
                time_left = max(0, time_left)

                hour_left = int(time_left // 3600)
                minute_left = int((time_left % 3600) // 60)
                seconds_left = int(time_left % 60)

                verbose_str = f"{formatted_datetime} / {tar_datetime}    ----    "

                if hour_left > 0:
                    verbose_str = verbose_str + f"{hour_left}h "
                
                if minute_left > 0:
                    verbose_str += f"{minute_left}m "

                verbose_str += f"{seconds_left}s "

                print(verbose_str)

            verbose_index += 1
        
        if cur_time_value >= target_time_value:
            break
    
    print(" Time Up !")

class KDPoints:
    KDTree = None

    def __init__(self, points):
        if self.KDTree is None:
            # from scipy.spatial import KDTree
            type(self).KDTree = lazy_import("scipy.spatial").KDTree
        self.points = points
        self.kdtree = self.KDTree(points)

    def query(self, query_point):
        distance, point_index = self.kdtree.query(query_point)
        return distance, self.points[point_index]

def dump_yaml(yaml_data, yaml_path, overwrite=True, verbose=False):
    import yaml
    file_path = save_file_path_check(yaml_path, overwrite, verbose)
    with open(file_path, 'w', encoding='utf-8') as yaml_file:
        yaml.dump(yaml_data, yaml_file, allow_unicode=True, sort_keys=False)

def load_yaml(yaml_path):
    import yaml
    with open(yaml_path, 'r', encoding='utf-8') as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
    return yaml_data

def load_csv(csv_path):
    import csv
    info_list = list()
    with open(csv_path, 'r', encoding='utf8') as file:
        csv_reader = csv.DictReader(file)
        for temp_info in tqdm(csv_reader, desc=" @@ Loading csv file {csv_path}: "):
            info_list.append(temp_info)
    return info_list


if __name__ == '__main__':
    join_substring('asdfasdf', 123, 'asdf')
    pass