from ..utils import load_xml
from ..utils import get_file_line_number
from ..utils import tqdm
from ..utils import get_list_from_list
from ..utils import pickle_load, pickle_save

import cv2
import numpy as np
from numpy.linalg import lstsq
from scipy.spatial import KDTree


class ObjManager:
    ReferencePointNum = 8

    def __init__(self, xml_file_path=None):
        self.dx = 0
        self.dy = 0
        self.dz = 0
        self.SRS = 'Unknown'

        self.point_list = list()

        self.edge_polygon = None
        self.kdtree = None

        if xml_file_path is not None:
            xml_dict = load_xml(xml_file_path)
            Metadata = xml_dict['ModelMetadata']
            self.SRS = Metadata['SRS']
            self.dx, self.dy, self.dz = get_list_from_list(Metadata['SRSOrigin'].split(','), lambda x: float(x))

    def add_obj_file(self, obj_file_path):
        line_number = get_file_line_number(obj_file_path)
        bar = tqdm(total=line_number, desc='Reading obj file')

        with open(obj_file_path, 'r') as file:
            while True:
                line = file.readline()

                bar.update(1)
                if not line:
                    break
                if len(line) < 3:
                    continue

                if line[0:2] == 'v ':
                    x, y, z = get_list_from_list(line[:-1].split(' ')[1:], lambda x: float(x))
                    cur_point = [x + self.dx, y + self.dy, z + self.dz]
                    self.point_list.append(cur_point)

        self.prepare()
        pass

    def save(self, file_path):
        pickle_save(self, file_path, overwrite=True)

    @classmethod
    def load(cls, file_path):
        return pickle_load(file_path)

    def prepare(self):
        points_array = np.array([point[:2] for point in self.point_list]).astype('float32')

        # make edge polygon
        self.edge_polygon = cv2.convexHull(points_array)

        # make kd tree obj
        self.kdtree = KDTree(points_array)
        pass

    def get_altitude(self, x, y):
        query_point = [x, y]

        def cal_z(coefficients, x ,y):
            a, b, c, d, e, f = coefficients
            return a * x**2 + b * y**2 + c * x * y + d * x + e * y + f

        distance_list, index_list = self.kdtree.query(query_point, k=self.ReferencePointNum)
        points = [self.point_list[index] for index in index_list]
        points = np.array(points)

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