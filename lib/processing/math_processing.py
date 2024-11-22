import math
import numpy as np
from ..utils import cal_distance
from ..utils import is_iterable
from ..utils import is_number


def calculate_mode(data):
    count_dict = {}
    for num in data:
        count_dict[num] = count_dict.get(num, 0) + 1
    max_count = max(count_dict.values())
    modes = [num for num, count in count_dict.items() if count == max_count]
    return modes

def cartesian_to_polar(point_xy, center_xy, reverse_y=True):
    x, y = point_xy
    center_x, center_y = center_xy
    r = cal_distance(point_xy, center_xy)
    if r == 0:
        theta = 0
    elif r > 0:
        y_ = y - center_y
        if reverse_y:
            y_ = - y_
        x_ = x - center_x
        theta = math.asin(abs(y_) / max(1, abs(r)))
        if y_ >= 0 and x_ >= 0:
            pass
        elif y_ >= 0 and x_ < 0:
            theta = np.pi - theta
        elif y_ < 0 and x_ < 0:
            theta = np.pi + theta
        elif y_ < 0 and x_ >= 0:
            theta = 2 * np.pi - theta
        else:
            raise RuntimeError(f'bad y_ {y_} x_ {x_}')
    else:
        raise RuntimeError(f'bad distance {r}')
    return r, theta

def polar_to_cartesian(r, theta, center_xy, reverse_y=False):
    center_x, center_y = center_xy
    r = np.array(r)
    theta = np.array(theta)
    if reverse_y:
        y = center_y - r * np.sin(theta)
    else:
        y = center_y + r * np.sin(theta)
    x = center_x + r * np.cos(theta)

    point_xy = np.vstack([x, y]).T.tolist()
    return point_xy

def cal_length(vector):
    return (np.array(vector) ** 2).sum() ** 0.5

cal_vector_length = cal_length

def get_angle_from_vector(vector, degree=False):
    assert len(vector) == 2
    assert (np.array(vector) ** 2).sum() > 0
    length = cal_length(vector)
    x, y = vector[:2]
    sin_res = y / length
    cos_res = x / length
    radians = np.arccos(cos_res)
    if sin_res < 0:
        radians = np.pi * 2 - radians
    if degree:
        return radians / np.pi * 180
    else:
        return radians

def cal_vector_degree_by_X_axis(vector_xy):
    # cal counterclockwise degree from vector_start [1, 0] to vector_end
    if cal_length(vector_xy) == 0:
        print('Warning: the length of #degree_cal# input vector is 0 !!! a zero will be returned.')
        return 0
    vector_xy = np.array(vector_xy)
    vector_xy = vector_xy / cal_length(vector_xy)
    degree = math.acos(vector_xy[0]) * 180 / math.pi
    if vector_xy[1] < 0:
        degree = 360 - degree
    return degree


class LogicOp:
    @staticmethod
    def EqualTo(value):
        def func(input):
            if input == value:
                return True
            else:
                return False
        return func

    @staticmethod
    def NotEqualTo(value):
        def func(input):
            if input != value:
                return True
            else:
                return False
        return func

    @staticmethod
    def LargerThan(value):
        def func(input):
            if input > value:
                return True
            else:
                return False
        return func

    @staticmethod
    def LessThan(value):
        def func(input):
            if input < value:
                return True
            else:
                return False
        return func

    @staticmethod
    def NotLessThan(value):
        def func(input):
            if input >= value:
                return True
            else:
                return False
        return func
    
    @staticmethod
    def NotLargerThan(value):
        def func(input):
            if input <= value:
                return True
            else:
                return False
        return func
    
    @staticmethod
    def In(value):
        assert is_iterable(value)
        def func(input):
            if input in value:
                return True
            else:
                return False
        return func

    @staticmethod
    def NotIn(value):
        assert is_iterable(value)
        def func(input):
            if input in value:
                return False
            else:
                return True
        return func
    
    @staticmethod
    def Is(value):
        def func(input):
            if input is value:
                return True
            else:
                return False
        return func
    
    @staticmethod
    def IsNot(value):
        def func(input):
            if input is not value:
                return True
            else:
                return False
        return func
    
    @staticmethod
    def IsTrue():
        def func(input):
            if input:
                return True
            else:
                return False
        return func
    
    @staticmethod
    def IsFalse():
        def func(input):
            if not input:
                return True
            else:
                return False
        return func


def make_rotate_matrix_2d(alpha, radian_mode=True):
    """
    x 轴正方向 - 3 点
    y 轴正方向 - 12 点
    顺时针旋转 alpha 角
    """
    if not radian_mode:
        alpha = alpha / 180 * np.pi
    matrix = np.array([
        [np.cos(alpha), - np.sin(alpha)],
        [np.sin(alpha), np.cos(alpha)]
        ])
    return matrix


def point_rotate(point, center, alpha=0, rotate_matrix=None, radian_mode=True):
    """_summary_

    Args:
        point ([x,y]): point to be rotated
        center ([x,y]): center of rotation
        alpha (float): rotate angle, if rotate_matrix set, this value will be ignored. Defaults to 0.
        rotate_matrix (2d matrix): rotate matrix 2d, if not set, will build by angle. Defaults to None.
        radian_mode (bool, optional): if True, alpha is considered to be a radian. Defaults to True.

    Returns:
        _type_: _description_
    """
    point = np.array(point)
    center = np.array(center)
    if (point == center).all():
        return point
    if rotate_matrix is None:
        rotate_matrix = make_rotate_matrix_2d(alpha, radian_mode)
    delt_x, delt_y = point - center
    new_point = np.matmul(np.array([delt_x, delt_y]), rotate_matrix)
    return new_point


def polygon_rotate(polygon, center, alpha=0, rotate_matrix=None, radian_mode=True):
    if len(polygon):
        polygon_matrix = np.array(polygon) - center
    else:
        return polygon

    assert polygon_matrix.ndim == 2
    assert polygon_matrix.shape[1] == 2

    if rotate_matrix is None:
        rotate_matrix = make_rotate_matrix_2d(alpha, radian_mode)

    rotated_polygon = np.matmul(polygon_matrix, rotate_matrix) + center

    return rotated_polygon.tolist()


def gaussian_2D(x, y, x0, y0, sigma_x, sigma_y, rho, e=1e-7):

    sigma_x = max(sigma_x, e)
    sigma_y = max(sigma_y, e)
    rho = np.clip(rho, -1 + e, 1 - e)
    
    det_Sigma = sigma_x ** 2 * sigma_y ** 2 * (1 - rho ** 2)

    dx = x - x0
    dy = y - y0

    A = 1 / (2 * np.pi * (det_Sigma ** 0.5))
    B = - (1 / 2) * (1 / det_Sigma) * ((dx**2) * (sigma_y ** 2) - 2*rho*dx*dy*sigma_x*sigma_y + (dy**2) * (sigma_x ** 2))

    res = A * np.exp(B)
    return res


def fit_gaussian_2D(point_xy_list):
    point_array = np.array(point_xy_list)
    assert point_array.ndim == 2
    assert point_array.shape[1] == 2
    assert point_array.shape[0] >= 5

    mean_x = np.mean(point_array[:, 0])
    mean_y = np.mean(point_array[:, 1])
    sigma_x = np.std(point_array[:, 0])
    sigma_y = np.std(point_array[:, 1])

    rho = ((point_array[:, 0] * point_array[:, 1]).mean() - mean_x * mean_y) / (sigma_x * sigma_y)
    return mean_x, mean_y, sigma_x, sigma_y, rho

def get_distance_by_lat_lon_1(lat1, lon1, lat2, lon2):
    # 将经纬度转换为弧度
    lat1 = float(lat1)
    lon1 = float(lon1)
    lat2 = float(lat2)
    lon2 = float(lon2)

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    # 使用 haversine 公式
    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # 地球平均半径大约为 6371 km
    R = 6371000
    # 计算两点的距离
    distance = R * c
    return distance

# 计算距离
def get_distance_by_lat_lon_2(latA, lonA, latB, lonB):
    # 将经纬度转换为弧度
    latA = float(latA)
    lonA = float(lonA)
    latB = float(latB)
    lonB = float(lonB)

    ra = 6378140  # 赤道半径
    rb = 6356755  # 极半径
    flatten = (ra - rb) / ra  # Partial rate of the earth
    # change angle to radians
    radLatA = math.radians(latA)
    radLonA = math.radians(lonA)
    radLatB = math.radians(latB)
    radLonB = math.radians(lonB)
 
    pA = math.atan(rb / ra * math.tan(radLatA))
    pB = math.atan(rb / ra * math.tan(radLatB))
    x = math.acos(math.sin(pA) * math.sin(pB) + math.cos(pA) * math.cos(pB) * math.cos(radLonA - radLonB))
    c1 = (math.sin(x) - x) * (math.sin(pA) + math.sin(pB)) ** 2 / math.cos(x / 2) ** 2
    c2 = (math.sin(x) + x) * (math.sin(pA) - math.sin(pB)) ** 2 / math.sin(x / 2) ** 2
    dr = flatten / 8 * (c1 - c2)
    distance = ra * (x + dr)
    distance = round(distance , 8)
    return distance

import math
 
# 计算角度
def get_degree_by_lat_lon(latA, lonA, latB, lonB):
    radLatA = math.radians(latA)
    radLonA = math.radians(lonA)
    radLatB = math.radians(latB)
    radLonB = math.radians(lonB)
    dLon = radLonB - radLonA
    y = math.sin(dLon) * math.cos(radLatB)
    x = math.cos(radLatA) * math.sin(radLatB) - math.sin(radLatA) * math.cos(radLatB) * math.cos(dLon)
    brng = math.degrees(math.atan2(y, x))

    brng = round((360 - brng + 90) % 360, 8)
    # brng = int(brng)
    dir_str = ''
    if (brng == 0.0) or ((brng == 360.0)):
        dir_str = '正东方向'
    elif brng == 90.0:
        dir_str = '正北方向'
    elif brng == 180.0:
        dir_str = '正西方向'
    elif brng == 270.0:
        dir_str = '正南方向'
    elif 0 < brng < 90:
        dir_str = f'北偏东{90 - brng}'
    elif 90 < brng < 180:
        dir_str = f'北偏西{brng - 90}'
    elif 180 < brng < 270:
        dir_str = f'南偏西{270 - brng}'
    elif 270 < brng < 360:
        dir_str = f'南偏东{brng - 270}'
    else:
        dir_str = '未知方向'
    return brng, dir_str

def get_destination_by_lat_lon(start_lat, start_lon, distance, bearing):
    # 接受起始经纬度坐标、距离（以米为单位）和方位角作为输入
    # 函数计算了在地表上按照给定方向和距离前进后的目标经纬度坐标。
    # bearing 为正北 90 度，正西 180度， 正东 0 度，正南 270 度

    bearing = (360-(bearing - 90))%360

    # 方位角转换为 —— 以北为0度，向东为90度，向南为180度，向西为270度。

    # 将经纬度从度转换为弧度
    start_lat_rad = math.radians(start_lat)
    start_lon_rad = math.radians(start_lon)
    
    # 定义地球半径
    earth_radius = 6371000
    
    # 将距离转换为米
    distance_meters = distance
    
    # 将方位角转换为弧度
    bearing_rad = math.radians(bearing)
    
    # 计算终点纬度
    lat2_rad = math.asin(math.sin(start_lat_rad) * math.cos(distance_meters / earth_radius) +
                         math.cos(start_lat_rad) * math.sin(distance_meters / earth_radius) * math.cos(bearing_rad))
    
    # 计算终点经度
    lon2_rad = start_lon_rad + math.atan2(math.sin(bearing_rad) * math.sin(distance_meters / earth_radius) * math.cos(start_lat_rad),
                                           math.cos(distance_meters / earth_radius) - math.sin(start_lat_rad) * math.sin(lat2_rad))
    
    # 将弧度转换为度
    lat2 = math.degrees(lat2_rad)
    lon2 = math.degrees(lon2_rad)
    
    return lat2, lon2
    
    
def point_in_triangle_2d(P, A, B, C):
    # P is the point, A, B, C are the vertices of the triangle
    # P could be a 2d vector which means a point in 2d space
    # or P also could be a N x 2 matrix which means N points in 2d space in which case the function will return a N x 1 matrix of booleans

    P = np.array(P)
    A = np.array(A[:2])
    B = np.array(B[:2])
    C = np.array(C[:2])

    if np.cross(A-B, B-C) == 0:
        return False
    
    res1 = np.cross(P - A, B - A)
    res2 = np.cross(P - B, C - B)
    res3 = np.cross(P - C, A - C)

    return ((res1 >= 0) * (res2 >= 0) * (res3 >= 0)) | ((res1 <= 0) * (res2 <= 0) * (res3 <= 0))

def get_z_from_xy_on_plane_ABC_3d(x, y, A, B, C):
    # A, B, C are 3d points who define a plane
    # function will calculate z value of x y on this plane
    if is_number(x) and is_number(y):
        x = [x]
        y = [y]

    x = np.array(x)
    y = np.array(y)

    assert len(x) == len(y), f' !! Error: x and y must have the same length, but x has {len(x)} and y has {len(y)}'

    input_length = len(x)

    A = np.array(A[:3])
    B = np.array(B[:3])
    C = np.array(C[:3])

    assert len(A) == 3 and len(B) == 3 and len(C) == 3, f' !! Error: A, B and C must have 3 elements, but A has {len(A)}, B has {len(B)} and C has {len(C)}'
    
    # cal normal vector
    n = np.cross(B - A, C - A)

    if np.linalg.norm(n, 1) == 0:
        print(" !! Warning: normal vector is zero vector")
        return np.zeros([input_length, 2])
    elif n[2] == 0:
        print(" !! Warning: normal vector is parallel to z axis")
        return np.zeros([input_length, 2])
    else:
        # pA is vertical to normal vector
        z = (n[2] * A[2] - n[0] * (x - A[0]) - n[1] * (y - A[1])) / n[2]
        return z


def cal_triangle_area_by_side_length(a, b, c):
    # a, b, c are the side length of a triangle
    # return the area of the triangle
    s = (a + b + c) / 2
    return math.sqrt(s * (s - a) * (s - b) * (s - c))


def cal_triangle_area_by_point(x1, y1, x2, y2, x3, y3):
    # x1, y1, x2, y2, x3, y3 are the coordinates of the three points of a triangle
    # return the area of the triangle
    a = cal_distance([x1, y1], [x2, y2])
    b = cal_distance([x2, y2], [x3, y3])
    c = cal_distance([x3, y3], [x1, y1])
    return cal_triangle_area_by_side_length(a, b, c)
