from ..utils import load_xml
from ..utils import get_file_line_number
from ..utils import tqdm
from ..utils import get_list_from_list
from ..utils import pickle_load, pickle_save
from ..utils import glob_recursively
from ..utils import cal_distance
from ..utils import vvd_floor

from ..processing import PIS
from ..processing import polygon2bbox

import re
import cv2
import numpy as np
from typing import List
from numpy.linalg import lstsq
from scipy.spatial import KDTree
from shapely.geometry import Polygon
from shapely.geometry import Point as ShapelyPoint
from pyproj import Transformer


transformer_4538_to_4326 = Transformer.from_crs("EPSG:4538", "EPSG:4326", always_xy=True)
transformer_4326_to_4538 = Transformer.from_crs("EPSG:4326", "EPSG:4538", always_xy=True)

def trans_4326_to_4538(lat, lon):
    return transformer_4326_to_4538.transform(lon, lat)

def trans_4538_to_4326(x, y):
    # output : lon, lat
    return transformer_4538_to_4326.transform(x, y)

class Point:
    def __init__(self, lat, lon, z=0):
        assert lat >= -90 and lat <= 90, "Latitude must be between -90 and 90"
        assert lon >= -180 and lon <= 180, "Longitude must be between -180 and 180"

        self.lat = float(lat)
        self.lon = float(lon)

        self.x, self.y = trans_4326_to_4538(self.lat, self.lon)
        self.z = float(z)

        self.x = float(self.x)
        self.y = float(self.y)

        self.key_point = False
        self.edge_point = False
        self.connection_point = False
        self.key_value = 0

    @classmethod
    def from_xy(cls, x, y, z=0, decimal_places=None):
        if decimal_places is not None:
            x = round(x, decimal_places)
            y = round(y, decimal_places)

        lon, lat = trans_4538_to_4326(x, y)
        return cls(lat, lon, z)

    def get_xy(self):
        return self.x, self.y

    def __str__(self):
        return f"(lat: {self.lat}, lon: {self.lon})"

    def __eq__(self, point) -> bool:
        return self.lat == point.lat and self.lon == point.lon

    def __hash__(self):
        return hash((self.lat, self.lon))

    def distance_to(self, other):
        return cal_distance([self.x, self.y], [other.x, other.y])

    def vector_xy(self, other):
        return np.array([other.x - self.x, other.y - self.y])

    def vector_lat_lon(self, other):
        return np.array([other.lat - self.lat, other.lon - self.lon])
    
    def clone(self):
        return type(self)(self.lat, self.lon)

    def __lt__(self, other):
        if self.lat == other.lat:
            return self.lon < other.lon
        else:
            return self.lat < other.lat


class ObjManager:
    ReferencePointNum = 8
    Margin = 10

    # 误差在 0.034m 左右
    def __init__(self, xml_file_path=None):
        self.dx = 0
        self.dy = 0
        self.dz = 0
        self.SRS = 'Unknown'

        self.point_list = list()

        self.polygon = None
        self.kdtree = None

        self.bad_point_num = 0
        self.duplicate_point_num = 0
        self.prepared = False

        if xml_file_path is not None:
            xml_dict = load_xml(xml_file_path)
            Metadata = xml_dict['ModelMetadata']
            self.SRS = Metadata['SRS']
            self.dx, self.dy, self.dz = get_list_from_list(Metadata['SRSOrigin'].split(','), lambda x: float(x))

    def __len__(self):
        return len(self.point_list)

    def add_obj_file(self, obj_file_path):
        line_number = get_file_line_number(obj_file_path)
        bar = tqdm(total=line_number, desc=' @@ Reading obj file: ')

        point_set = set((point[0], point[1]) for point in self.point_list)
        self.prepared = False

        with open(obj_file_path, 'r') as file:
            while True:
                line = file.readline()

                bar.update(1)
                if not line:
                    break

                if len(line) < 5:
                    continue

                if line[0:2] == 'v ':
                    match_res = re.match(r'v +(.*) +(.*) +(.*).', line)
                    x = round(float(match_res[1]) + self.dx, 6)
                    y = round(float(match_res[2]) + self.dy, 6)
                    z = float(match_res[3]) + self.dz

                    cur_point = (x, y)

                    if cur_point not in point_set:
                        self.point_list.append([x, y, z])
                        point_set.add(cur_point)
                    else:
                        self.duplicate_point_num += 1
        pass

    def get_distance(self, x, y):
        sp_point = ShapelyPoint(x, y)
        # 返回 0 在三角形内，正数为在三角形外
        return self.polygon.distance(sp_point)

    def save(self, file_path, check_data=True):
        if not self.prepared:
            self.prepare()

        if check_data:
            self.data_check()

        pickle_save(self, file_path, overwrite=True)

    def data_check(self):
        self.bad_point_num = 0
        for point_index, point in tqdm(enumerate(self.point_list), desc=' @@ Data checking: '):
            index_list = self.kdtree.query_ball_point([point[0], point[1]], 5)
            alt_list = [self.point_list[index][2] for index in index_list]
            median_alt = np.median(alt_list)
            if abs(point[2] - median_alt) > 10:
                self.point_list[point_index][2] = median_alt
                self.bad_point_num += 1

    @classmethod
    def load(cls, file_path):
        obj = pickle_load(file_path)
        if not obj.prepared:
            obj.prepare()
        return obj

    @classmethod
    def from_dir_list(cls, dir_list):
        if not isinstance(dir_list, list):
            dir_list = [dir_list]
        cur_obj = cls()
        for dir in dir_list:
            temp_obj = cls.from_dir(dir)
            cur_obj_point_set = set((point[0], point[1]) for point in cur_obj.point_list)
            for point in temp_obj.point_list:
                if (point[0], point[1]) not in cur_obj_point_set:
                    cur_obj.point_list.append(point)

        cur_obj.prepare()
        return cur_obj

    @classmethod
    def from_dir(cls, dir):
        xml_file_path = glob_recursively(dir, 'xml')
        assert len(xml_file_path) == 1, 'xml file number is not 1'
        obj = cls(xml_file_path[0])
        
        obj_file_path_list = glob_recursively(dir, 'obj')
        for obj_file_path in obj_file_path_list:
            obj.add_obj_file(obj_file_path)
            pass
        obj.prepare()
        return obj

    def prepare(self):
        points_array = np.array([[point[0], point[1]] for point in self.point_list]).astype('float32')

        # make edge polygon
        polygon = cv2.convexHull(points_array)
        self.polygon = Polygon(polygon[:,0,:])

        # make kd tree obj
        self.kdtree = KDTree(points_array)
        self.prepared = True
        pass

    def get_altitude(self, x, y):
        if not self.prepared:
            self.prepare()

        query_point = [x, y]

        def cal_z(coefficients, x ,y):
            a, b, c, d, e, f = coefficients
            return a * x**2 + b * y**2 + c * x * y + d * x + e * y + f

        distance_list, index_list = self.kdtree.query(query_point, k=self.ReferencePointNum)

        if self.ReferencePointNum == 1:
            index_list = [index_list]
            distance_list = [distance_list]

        points = [self.point_list[index] for index in index_list]
        points = np.array(points)

        # weight_list = 1 / (np.square(points[:, :2] - query_point).sum(axis=1) + 1e-6)
        # weight_list = weight_list / weight_list.sum()

        # tar_z = (points[:, 2] * weight_list).sum()

        A = np.c_[points[:, 0]**2, points[:, 1]**2, points[:, 0]*points[:, 1], points[:, 0], points[:, 1], np.ones(self.ReferencePointNum)]
        B = points[:, 2]

        # 使用最小二乘法求解系数
        coefficients, _, _, _ = lstsq(A, B, rcond=None)

        # print(f'二次曲面方程: z = {a:.2f}x^2 + {b:.2f}y^2 + {c:.2f}xy + {d:.2f}x + {e:.2f}y + {f:.2f}')
        max_error = -1
        for point in points:
            cur_z = cal_z(coefficients, point[0], point[1])
            max_error = max(max_error, abs(cur_z - point[2]))

        tar_z = cal_z(coefficients, x, y)
        return tar_z, distance_list, max_error

    def make_alt_graph(self, point_polygon: List[Point], resolution=1, median_radius=0):
        if not self.prepared:
            self.prepare()

        assert resolution > 0, ' !! resolution must be positive'

        points_array = np.array([[point.x, point.y] for point in point_polygon]).astype('float32')

        polygon_obj = Polygon(points_array)

        # make edge polygon
        bbox = polygon2bbox(points_array)

        width = vvd_floor((bbox[2] - bbox[0]) / resolution) + 2 * self.Margin
        height = vvd_floor((bbox[3] - bbox[1]) / resolution) + 2 * self.Margin

        canvas = np.zeros((height, width), dtype=np.float32)

        for index_x in tqdm(range(width)):
            if index_x < self.Margin or index_x >= width - self.Margin:
                continue

            for index_y in range(height):
                if index_y < self.Margin or index_y >= height - self.Margin:
                    continue

                x = bbox[0] + (index_x - self.Margin) * resolution
                y = bbox[1] + (index_y - self.Margin) * resolution

                sp_point = ShapelyPoint(x, y)

                if polygon_obj.distance(sp_point) > 0:
                    continue

                if self.get_distance(x, y) > 0:
                    continue

                if median_radius <= 0:
                    cur_alt = self.get_altitude(x, y)[0]
                else:
                    index_list = self.kdtree.query_ball_point([x, y], median_radius)
                    alt_list = [self.point_list[index][2] for index in index_list]
                    cur_alt = np.median(alt_list)

                canvas[index_y, index_x] = cur_alt

        result_img = canvas[::-1, :]
        return result_img