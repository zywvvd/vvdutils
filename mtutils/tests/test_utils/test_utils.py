import sys
sys.path.append('.')

import pytest
import numpy as np
import re
import time
from lib.utils.utils import *

test_file = 'assets/everytime3p.jpg'

def test_get_list_from_list():
    test_data = list(range(20))
    res = get_list_from_list(test_data, lambda x: x if not x % 3 else None)
    assert res == [x  for x in range(20) if not x % 3 ]

    data = np.array(range(20))
    res = get_list_from_list(data, lambda x: x if not x % 3 else None)
    assert res == [x  for x in range(20) if not x % 3 ]

    data = {x : x for x in range(20)}
    res = get_list_from_list(data, lambda x: x if not x % 3 else None)
    assert res == [x  for x in range(20) if not x % 3 ]

def test_segment_intersection():
    seg_1 = (5, 19)
    seg_2 = (16, 24)
    res = segment_intersection(seg_1, seg_2)
    assert res == (3, (16, 19))

def test_concat_generator():
    from inspect import isgenerator
    def gen1():
        for index in range(10):
            yield index
    def gen2():
        for index in range(30):
            yield index % 10
    gen3 = concat_generator(gen1(), gen1(), gen1())
    assert isgenerator(gen3)
    for var1, var2 in zip(gen3, gen2()):
        assert var1 == var2

def test_get_mac_address():
    mac = get_mac_address()
    assert re.match("([A-Z0-9]{2}:){5}[A-Z0-9]{2}", mac) is not None

def test_cal_distance():
    vector_1, vector_2 = [1,2,3,4], [2,3,4,5]
    assert cal_distance(vector_1, vector_2) == 2

def test_get_file_size_M():
    res = get_file_size_M(test_file)
    assert 0.008 < res < 0.01

def test_unify_data_to_python_type():
    data = np.ones([2, 2]), 234, 152.5, True, Path('test')
    res = unify_data_to_python_type(data)
    assert [[[1.0, 1.0], [1.0, 1.0]], 234, 152.5, True, 'test'] == res

def test_timer_vvd():
    def test():
        sum_value = 0
        for index in range(100001):
            sum_value += index
        return sum_value
    res = timer_vvd(test)()
    assert res == (1+100000)*50000

def test_file_lines():
    data = ['abc', 'def']
    temp_data_path = 'assets/temp.txt'
    file_write_lines(data, temp_data_path, overwrite=True)
    lines = file_read_lines(temp_data_path)
    assert lines == data
    remove_file(temp_data_path)
    assert not OS_exists(temp_data_path)

def test_pickle():
    data = ['abc', 'def']
    temp_data_path = 'assets/temp.pickle'
    pickle_save(data, temp_data_path)
    load_data = pickle_load(temp_data_path)
    assert load_data == data
    remove_file(temp_data_path)
    assert not OS_exists(temp_data_path)

def test_find_sub_string():
    test_str = 'abc_0abc_1abc_2abc_3abc_4abc_5abc_6'
    res = find_sub_string(test_str, 'abc_', 5)
    assert test_str[res+4] == '5'

def test_str_connection():
    sub_1,sub_2, sub3 = 'asdf', 'asodkfj', 'askljdg'
    con = '&&&'
    res = str_connection(sub_1,sub_2, sub3, connect_char=con)
    assert res == 'asdf&&&asodkfj&&&askljdg'

def test_get_main_file_name():
    test_file_name = __file__
    res = get_main_file_name(test_file_name)
    assert res == 'test_utils'

def test_strong_printing():
    try:
        strong_printing('test')
        assert True
    except:
        assert False

def test_current_system():
    try:
        res = current_system()
        assert isinstance(res, str)
    except:
        assert False

def test_current_split_char():
    try:
        res = current_split_char()
        assert res in ['/', '\\']
    except:
        assert False

def test_save_file_path_check():
    test_file = __file__
    res = save_file_path_check(test_file)
    assert len(res) - len(test_file) == 20

def test_has_chinese_char():
    assert not has_chinese_char('asdlfkj')
    assert has_chinese_char('测试')

def test_encode_chinese_to_unicode():
    assert 'abc #U6d4b#U8bd5' == encode_chinese_to_unicode("abc 测试")
    assert 'abc ' == encode_chinese_to_unicode("abc 测试", remove=True)

def test_create_uuid():
    uuid = create_uuid()
    assert len(uuid) == 32

def test_xor():
    assert xor(True, False)
    assert not xor(True, True)
    assert not xor(False, False)

def test_get_file_hash_code():
    hash_code = get_file_hash_code(test_file)
    assert 'b5fc9a14763c0c83939bacb7ae3a664c' == hash_code

def test_get_date_str():
    string = get_date_str()
    assert re.match('[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}', string) is not None

def test_isdir():
    parent_dir = Path(__file__).parent
    assert isdir(parent_dir)

def test_uniform_split_char():
    test_str = '\\asdfasdf\\asdfasdf/asdfasdf/asdfasdf/asdfasdf\\ewreg/asdfgsdfg'
    res1 = uniform_split_char(test_str)
    res2 = uniform_split_char(test_str, '\\')
    assert res1 == '/asdfasdf/asdfasdf/asdfasdf/asdfasdf/asdfasdf/ewreg/asdfgsdfg'
    assert res2 == '\\asdfasdf\\asdfasdf\\asdfasdf\\asdfasdf\\asdfasdf\\ewreg\\asdfgsdfg'
    pass

def test_dir_check():
    test_dir1 = Path(__file__).parent
    assert dir_check(test_dir1)
    test_dir2 = 'assets/test_dir/test_dir/test_dir'
    remove_dir('assets/test_dir')
    assert not dir_check(test_dir2)
    assert isdir(test_dir2)
    remove_dir('assets/test_dir')
    assert not isdir('assets/test_dir')

def test_time_reduce():
    assert 120 == time_reduce(1,2,3,4,5)

def test_get_function_name():
    res = get_function_name()
    assert res == 'test_get_function_name'

def test_draw_RB_map():
    y_true = [1] * 10 + [0] * 10
    y_pred = [res / 10 for res in range(10)] * 2
    file_path = 'assets/test.jpg'
    draw_RG_map(y_true, y_pred, map_save_path=file_path)
    assert OS_exists(file_path)
    remove_file(file_path)

def test_histgrom():
    value, bins = histgrom(range(100), show=False)
    assert all(value == 1)
    assert min(bins) == 0
    assert max(bins) == 99

def test_is_path_obj():
    path1 = Path('test')
    path2 = Path2('test')
    path3 = dict()
    path4 = 'str'
    assert is_path_obj(path1)
    assert is_path_obj(path2)
    assert not is_path_obj(path3)
    assert not is_path_obj(path4)

def test_time_stamp():
    stamp = time_stamp()
    assert re.match('[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}', stamp) is not None

def test_smart_copy():
    source_path = test_file
    new_path = change_file_name_for_path(source_path, 'test.asldkfj', '')
    smart_copy(source_path, new_path)
    assert get_file_hash_code(source_path) == get_file_hash_code(new_path)
    remove_file(new_path)

def test_json():
    data = [{'setasdf':'asdfasdf', '9234': {'afdasdf':'asdfasdf'}}, [1234, 'asdfasdf'], True]
    json_path = 'assets/test.json'
    json_save(data, json_path, overwrite=True)
    load_data = json_load(json_path)
    assert load_data == data
    remove_file(json_path)

def test_glob_recursively():
    json_path_list = glob_recursively('assets', 'json')
    assert len(json_path_list) > 0
    for json_path in json_path_list:
        assert OS_exists(json_path)

def test_is_integer():
    assert is_integer(3)
    assert is_integer(-3)
    assert is_integer(-0)
    assert is_integer(np.array(4))
    assert is_integer(np.array(4, dtype='int32'))
    assert is_integer(np.array([3, 4]))
    assert not is_integer(3.5)
    assert not is_integer(np.array(3.5))
    assert not is_integer(np.array([3, 3.5]))

def test_if_float():
    assert is_float(3.5)
    assert is_float(np.array(3.5))
    assert not is_float(3)
    assert not is_float(np.array(3))

def test_is_number():
    assert is_number(3)
    assert is_number(.3)
    assert not is_number('a')

def test_is_bool():
    assert is_bool(3 > 0)
    assert is_bool(np.array([3,4]) > 0)
    assert not is_bool(3)
    assert is_bool(True)
    assert is_bool(np.array(True))

def test_whether_divisible_by():
    assert whether_divisible_by(99, 3)
    assert not whether_divisible_by(97, 3)

def test_vvd_round():
    assert vvd_round(3.5) == 4
    assert vvd_round(3.6) == 4
    assert vvd_round(-3.6) == -4
    assert vvd_round([-3.6, 2.1]) == [-4, 2]

def test_vvd_ceil():
    assert vvd_ceil(3.6) == 4
    assert vvd_ceil(-3.6) == -3
    assert vvd_ceil([-3.6, 2.1]) == [-3, 3]

def vvd_floor():
    assert vvd_floor(3.6) == 3
    assert vvd_floor(-3.6) == -4
    assert vvd_floor([-3.6, 2.1]) == [-4, 2]

def test_get_gpu_str_as_you_wish():
    gpu_str, index_list = get_gpu_str_as_you_wish(1)
    assert isinstance(gpu_str, str)
    assert isinstance(index_list, list)
    gpu_str, index_list = get_gpu_str_as_you_wish(100)
    assert isinstance(index_list, list)
    res = get_single_gpu_id()
    assert isinstance(res, int)

def test_get_dir_file_list():
    dir_list, file_list = get_dir_file_list('.')
    for dir in dir_list:
        assert isdir(dir)
    for file in file_list:
        assert not isdir(file)
        assert OS_exists(file)
    all_dir_list, all_file_list = get_dir_file_list('.', recursive=True)
    assert len(all_dir_list) > len(dir_list)
    assert len(all_file_list) > len(file_list)

def test_get_segments():
    data = [0] * 3 + [1] * 30 + [0] * 13 + [1] * 3 + [0] * 3 + [1] * 3
    res = get_segments(data)
    assert res == [[3, 32], [46, 48], [52, 54]]

def test_try_exc_handler():
    def error():
        raise RuntimeError
    def exc_func():
        return 'abc'
    def ok():
        return True
    try:
        assert try_exc_handler(ok, error, True) == True
        assert True
    except:
        assert False

    try:
        assert try_exc_handler(error, ok) != True
        assert True
    except:
        assert False

    try:
        try_exc_handler(error, ok, True)
        assert False
    except:
        assert True
    
    try:
        try_exc_handler(error, error)
        assert True
    except:
        assert False
    assert try_exc_handler(exc_func, error) == 'abc'

def test_class_timer():
    class test:
        @staticmethod
        def test():
            return 'aaa'
    
    timmer = class_timer(test)
    assert timmer.test() == 'aaa'

def test_remove_small_components():
    mask = np.zeros([20, 20])
    mask[:3, :3] = 1
    mask[8:16, 8:16] = 1
    res = remove_small_components(mask, 10)
    assert np.sum(res) == 64

def test_encode_path():
    test_path = __file__
    assert 'test_utils@test_utils.py' == encode_path(test_path, 2)
    assert 'tests@test_utils@test_utils.py' == encode_path(test_path, 99, 'tests')
    assert 'tests&&&test_utils&&&test_utils.py' == encode_path(test_path, 99, 'tests', sep_char='&&&')
    assert 'abc/asdf/asdfasdf/asdf/tests-&test_utils-&test_utils.py' == encode_path(test_path, 99, 'tests', sep_char='-&', root_path='abc\\asdf\\asdfasdf/asdf')
    assert 'abc/asdf/asdfasdf/asdf/tests-&test_utils-&test_utils.ijiijijij' == encode_path(test_path, 99, 'tests', sep_char='-&', root_path='abc\\asdf\\asdfasdf/asdf', with_suffix='ijiijijij')
    assert 'tests-&test_utils-&test_utils' == encode_path(test_path, 99, 'tests', sep_char='-&', root_path='abc\\asdf\\asdfasdf/asdf', with_suffix='ijiijijij', stem=True)

def test_extract_file_to():
    root_dir = 'assets'
    target_dir = 'test_dir'
    extract_file_to(root_dir, 'jpg', target_dir)
    jpg_paths = glob_recursively(target_dir, 'jpg')
    assert jpg_paths
    remove_dir(target_dir)
    assert not OS_exists(target_dir)

def test_link():
    symlink('assets', 'test_dir')
    
    tar_temp_path = 'assets/test.json'
    assert hardlink('test_dir/mc.json', tar_temp_path)
    assert OS_exists(tar_temp_path)
    assert not islink(tar_temp_path)
    remove_file(tar_temp_path)
    assert not OS_exists(tar_temp_path)

    cwd = get_cwd()
    assert symlink(OS_join(cwd, 'assets/mc.json'), tar_temp_path)
    assert islink(tar_temp_path)
    data = json_load(tar_temp_path)
    assert OS_exists(tar_temp_path)
    remove_file(tar_temp_path)
    assert not OS_exists(tar_temp_path)
    remove_dir('test_dir')
    assert not OS_exists('test_dir')

def test_ListOrderedDict():
    obj = ListOrderedDict()
    obj.append('aa', 'aa')
    obj.append('ac', 'ac')
    obj.append('ad', 'ad')
    try:
        obj.append(3, 'ad')
        assert False
    except:
        assert True
    
    obj['bb'] = 1234
    obj['342'] = 'opq'
    
    assert obj[0] == 'aa'
    assert obj[1] == 'ac'
    assert obj[2] == 'ad'
    assert obj[3] == 1234
    assert obj[4] == 'opq'

def test_is_valid_identifier():
    assert not is_valid_identifier(123)
    assert not is_valid_identifier('123')
    assert not is_valid_identifier('')
    assert not is_valid_identifier('asdf?')
    assert is_valid_identifier('O')
    assert is_valid_identifier('_asdfa234')
    assert is_valid_identifier('iaslkdjflkajs')

def test_chinese_str_to_pinyin():
    assert 'asdf' == chinese_str_to_pinyin('asdf')
    assert 'as_zhe_shi_ge_ _sha_a_df' == chinese_str_to_pinyin('as这是个 啥啊df')
    assert 'astttzhetttshitttgettt tttshatttatttdf' == chinese_str_to_pinyin('as这是个 啥啊df', 'ttt')

def test_remove_chinese_for_file_names():
    try:
        remove_chinese_for_file_names('assets')
        assert True
    except:
        assert False

def test_get_file_time():
    res = get_file_time(test_file)
    for key, value in res.items():
        assert isinstance(value, time.struct_time)

def test_gen_iter():
    def gen():
        for index in range(20):
            yield index
    iter = list(range(10))
    assert is_generator(gen())
    assert is_generator(concat_generator(gen(), gen(), iter))
    assert is_iterable(iter)
    assert is_iterable(gen())
    assert not is_generator(iter)


def test_random_string():
    res_str = random_string(20)
    assert isinstance(res_str, str)
    assert len(res_str) == 20

def test_display_class_property():
    class Test:
        a = 3
        b = 'asdf'
    property_dict = display_class_property(Test)
    assert property_dict == {'a': 3, 'b': 'asdf'}


if __name__ == '__main__':
    pass
    # pytest.main()