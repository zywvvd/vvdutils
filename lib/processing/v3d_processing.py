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

from ..loader import try_to_import
try_to_import("rasterio", "please install rasterio by `pip install rasterio` if you want to use this function")

import rasterio
from rasterio.transform import from_origin
from rasterio.enums import Compression

from rasterio.io import MemoryFile
import warnings

try_to_import("pygltflib", "please install pygltflib by `pip install pygltflib` if you want to use this function")
import pygltflib


class ProjTransformerManager:
    def __init__(self):
        self.transformer_dict = dict()
        try_to_import('pyproj', "please install pyproj by `pip install pyproj` if you want to use this function")

    def get_transformer(self, src_crs, dst_crs):
        from pyproj import Transformer
        if (src_crs, dst_crs) not in self.transformer_dict:
            self.transformer_dict[(src_crs, dst_crs)] = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
        return self.transformer_dict[(src_crs, dst_crs)]
    
    def transform(self, param_1, param_2, src_crs, dst_crs):
        return self.get_transformer(src_crs, dst_crs).transform(param_1, param_2)

proj_transformer_manager = ProjTransformerManager()

def trans_4326_to_4538(lat, lon):
    return proj_transformer_manager.transform(lon, lat, "4326", "4538")

def trans_4538_to_4326(x, y):
    # output : lon, lat
    return proj_transformer_manager.transform(x, y, "4538", "4326")

def get_utm_zone_from_zone_num(zone_num, is_south=False):
    return f'EPSG:327{format(zone_num, "02d")}' if is_south else f'EPSG:326{format(zone_num, "02d")}'

def get_utm_zone_from_lon(lon, is_south=False):
    """根据经度确定 UTM 区域"""     
    zone_number = int((lon + 180) / 6) + 1     
    return get_utm_zone_from_zone_num(zone_number, is_south)

def get_utm_zone_from_lon_lat(lon, lat):
    return get_utm_zone_from_lon(lon, lat < 0)

def get_utm_zone_from_wgs84_str(wgs84_str):
    # WGS84 UTM 51N
    match_res = re.match(r'WGS84 UTM(.+)(N|S)', wgs84_str)
    zone_num = int(match_res.group(1))
    is_south = match_res.group(2) == 'S'
    return get_utm_zone_from_zone_num(zone_num, is_south)

def get_wgs84_str_from_utm_zone_str(utm_zone_str):
    match_res = re.match(r"EPSG:32(\d)(\d{2})", utm_zone_str)
    s_or_n = 'S' if match_res[1] == '7' else 'N'
    zone_num = match_res[2]

    return f'WGS84 UTM {zone_num}{s_or_n}'

class Point:
    def __init__(self, lat, lon, z=0, utm_zone=None):
        self.lat = round(float(lat), 12)
        self.lon = round(float(lon), 12)

        assert self.lat >= -90 and self.lat <= 90, "Latitude must be between -90 and 90"
        assert self.lon >= -180 and self.lon <= 180, "Longitude must be between -180 and 180"

        if utm_zone is None:
            self.utm_zone_str = get_utm_zone_from_lon_lat(self.lon, self.lat)
            self.wgs84_str = get_wgs84_str_from_utm_zone_str(self.utm_zone_str)
        else:
            self.utm_zone_str = get_utm_zone_from_wgs84_str(utm_zone)
            self.wgs84_str = utm_zone


        self.x, self.y = trans_4326_to_4538(self.lat, self.lon)
        self.x_utm, self.y_utm = proj_transformer_manager.transform(self.lon, self.lat, "4326", self.utm_zone_str)
        self.z = float(z)

        self.x = round(float(self.x), 8)
        self.y = round(float(self.y), 8)

        self.x_utm = round(float(self.x_utm), 8)
        self.y_utm = round(float(self.y_utm), 8)

        self.key_point = False
        self.edge_point = False
        self.connection_point = False
        self.key_value = 0

    @classmethod
    def from_xy(cls, x, y, z=0, utm_zone=None):
        lon, lat = trans_4538_to_4326(x, y)
        return cls(lat, lon, z, utm_zone)

    @classmethod
    def from_xy_utm(cls, utm_zone, x, y, z=0):
        utm_zone_str = get_utm_zone_from_wgs84_str(utm_zone)
        lon, lat = proj_transformer_manager.transform(x, y, utm_zone_str, "4326")
        return cls(lat, lon, z)

    def get_xy(self):
        return self.x, self.y

    def get_xy_utm(self):
        return self.x_utm, self.y_utm

    def __str__(self):
        return f"(lat: {self.lat}, lon: {self.lon})"

    def __eq__(self, point) -> bool:
        return self.lat == point.lat and self.lon == point.lon

    def __hash__(self):
        return hash((self.lat, self.lon))

    def distance_to(self, other):
        return cal_distance([self.x_utm, self.y_utm, self.z], [other.x_utm, other.y_utm, other.z])

    def vector_xy(self, other):
        return np.array([other.x_utm - self.x_utm, other.y_utm - self.y_utm])

    def vector_lat_lon(self, other):
        return np.array([other.lat - self.lat, other.lon - self.lon])

    def clone(self):
        return type(self)(self.lat, self.lon)

    def __lt__(self, other):
        if self.lat == other.lat:
            return self.lon < other.lon
        else:
            return self.lat < other.lat
    
    def get_interpolation_point(self, target, ratio):
        assert self.wgs84_str == target.wgs84_str, "Points must be in the same UTM zone"
        x_utm = self.x_utm + (target.x_utm - self.x_utm) * ratio
        y_utm = self.y_utm + (target.y_utm - self.y_utm) * ratio
        z = self.z + (target.z - self.z) * ratio
        return Point.from_xy_utm(self.wgs84_str, x_utm, y_utm, z)
    
    def move(self, dx, dy, dz=0):
        x_utm = self.x_utm + dx
        y_utm = self.y_utm + dy
        z = self.z + dz
        return Point.from_xy_utm(self.wgs84_str, x_utm, y_utm, z)



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
        print('@@ Loading data ...')
        obj = pickle_load(file_path)
        if not hasattr(obj, "prepared") or not obj.prepared:
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
        print('@@ Preparing data ...')
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


def create_tiff_file(data, tif_img_path, left, top, res_width, res_height, width, height, utm_zone, compression=Compression.jpeg):
    utm_zone_str = get_utm_zone_from_wgs84_str(utm_zone)
    match_res = re.match("EPSG:(\d+)", utm_zone_str)[1]
    crs = rasterio.crs.CRS.from_epsg(int(match_res))

    transform = from_origin(left, top, res_width, res_height)
    
    array_data = np.array(data, dtype=np.uint8)
    ndim = data.ndim

    # compression = Compression.zstd

    if ndim == 2:
        # 创建TIFF文件并写入数据
        with rasterio.open(
            tif_img_path, 'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,  # 波段数
            dtype=rasterio.uint8,
            crs=crs,
            transform=transform,
            compress=compression,
            JPEG_QUALITY=90,
        ) as dst:
            # 写入波段
            dst.write(array_data, 1)

    elif ndim == 3:
        channel_num = data.shape[2]
        with rasterio.open(
            tif_img_path, 'w',
            driver='GTiff',
            height=height,
            width=width,
            count=channel_num,  # 波段数
            dtype=rasterio.uint8,
            crs=crs,
            transform=transform,
            compress=compression,
            JPEG_QUALITY=90,
        ) as dst:
            # 分别写入每个波段
            for i in range(channel_num):
                dst.write(array_data[:, :, i], i+1)

    else:
        raise ValueError(" !! Tiff Image Create - Invalid data dimensions {ndim}")
    



# from opendm import system
# from opendm import io
# from opendm import log

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

def transform_coordinates_zup_to_yup(vertices):
    """将坐标系从Z-up转换为Y-up"""
    # 旋转矩阵: [X, Z, -Y]
    return np.stack([vertices[:, 0], vertices[:, 2], -vertices[:, 1]], axis=1)

def load_obj(obj_path, _info=print, switch_to_yup=True):
    if not os.path.isfile(obj_path):
        raise IOError("Cannot open %s" % obj_path)

    obj_base_path = os.path.dirname(os.path.abspath(obj_path))
    obj = {
        'materials': {},
    }
    vertices = []
    uvs = []
    normals = []

    faces = {}
    current_material = "_"

    with open(obj_path) as f:
        _info("Loading %s" % obj_path)

        for line in f:
            if line.startswith("mtllib "):
                # Materials
                mtl_file = "".join(line.split()[1:]).strip()
                obj['materials'].update(load_mtl(mtl_file, obj_base_path, _info=_info))
            elif line.startswith("v "):
                # Vertices
                vertices.append(list(map(float, line.split()[1:4])))
            elif line.startswith("vt "):
                # UVs
                uvs.append(list(map(float, line.split()[1:3])))
            elif line.startswith("vn "):
                normals.append(list(map(float, line.split()[1:4])))
            elif line.startswith("usemtl "):
                mtl_name = "".join(line.split()[1:]).strip()
                if not mtl_name in obj['materials']:
                    raise Exception("%s material is missing" % mtl_name)

                current_material = mtl_name
            elif line.startswith("f "):
                if current_material not in faces:
                    faces[current_material] = []
                
                a,b,c = line.split()[1:]

                if a.count("/") == 2:
                    av, at, an = map(int, a.split("/")[0:3])
                    bv, bt, bn = map(int, b.split("/")[0:3])
                    cv, ct, cn = map(int, c.split("/")[0:3])

                    faces[current_material].append((av - 1, bv - 1, cv - 1, at - 1, bt - 1, ct - 1, an - 1, bn - 1, cn - 1)) 
                else:
                    av, at = map(int, a.split("/")[0:2])
                    bv, bt = map(int, b.split("/")[0:2])
                    cv, ct = map(int, c.split("/")[0:2])
                    faces[current_material].append((av - 1, bv - 1, cv - 1, at - 1, bt - 1, ct - 1)) 

    obj['vertices'] = np.array(vertices, dtype=np.float32)
    obj['uvs'] = np.array(uvs, dtype=np.float32)
    obj['normals'] = np.array(normals, dtype=np.float32)
    obj['faces'] = faces

    obj['materials'] = convert_materials_to_jpeg(obj['materials'])

    if switch_to_yup:
        obj['vertices'] = transform_coordinates_zup_to_yup(obj['vertices'])
        if 'normals' in obj and len(obj['normals']) > 0:
            obj['normals'] = transform_coordinates_zup_to_yup(obj['normals'])

    return obj

def convert_materials_to_jpeg(materials):

    min_value = 0
    value_range = 0
    skip_conversion = False

    for mat in materials:
        image = materials[mat]

        # Stop here, assuming all other materials are also uint8
        if image.dtype == np.uint8:
            skip_conversion = True
            break

        # Find common min/range values
        try:
            data_range = np.iinfo(image.dtype)
            min_value = min(min_value, 0)
            value_range = max(value_range, float(data_range.max) - float(data_range.min))
        except ValueError:
            # For floats use the actual range of the image values
            min_value = min(min_value, float(image.min()))
            value_range = max(value_range, float(image.max()) - min_value)
    
    if value_range == 0:
        value_range = 255 # Should never happen

    for mat in materials:
        image = materials[mat]

        if not skip_conversion:
            image = image.astype(np.float32)
            image -= min_value
            image *= 255.0 / value_range
            np.around(image, out=image)
            image[image > 255] = 255
            image[image < 0] = 0
            image = image.astype(np.uint8)

        with MemoryFile() as memfile:
            bands, h, w = image.shape
            bands = min(3, bands)
            with memfile.open(driver='JPEG', jpeg_quality=90, count=bands, width=w, height=h, dtype=rasterio.dtypes.uint8) as dst:
                for b in range(1, min(3, bands) + 1):
                    dst.write(image[b - 1], b)
            memfile.seek(0)
            materials[mat] = memfile.read()

    return materials

def load_mtl(mtl_file, obj_base_path, _info=print):
    mtl_file = os.path.join(obj_base_path, mtl_file)

    if not os.path.isfile(mtl_file):
        raise IOError("Cannot open %s" % mtl_file)
    
    mats = {}
    current_mtl = ""

    with open(mtl_file) as f:
        for line in f:
            if line.startswith("newmtl "):
                current_mtl = "".join(line.split()[1:]).strip()
            elif line.startswith("map_Kd ") and current_mtl:
                map_kd_filename = "".join(line.split()[1:]).strip()
                map_kd = os.path.join(obj_base_path, map_kd_filename)
                if not os.path.isfile(map_kd):
                    raise IOError("Cannot open %s" % map_kd)
                
                _info("Loading %s" % map_kd_filename)
                with rasterio.open(map_kd, 'r') as src:
                    mats[current_mtl] = src.read()
    return mats

def paddedBuffer(buf, boundary):
    r = len(buf) % boundary
    if r == 0: 
        return buf 
    pad = boundary - r
    return buf + b'\x00' * pad

def obj2glb(input_obj, output_glb, rtc_center=(None, None), draco_compression=True, switch_to_yup=True, _info=print):
    _info("Converting %s --> %s" % (input_obj, output_glb))
    obj = load_obj(input_obj, _info=_info, switch_to_yup=switch_to_yup)

    vertices = obj['vertices']
    uvs = obj['uvs']
    # Flip Y
    uvs = (([0, 1] - (uvs * [0, 1])) + uvs * [1, 0]).astype(np.float32)
    normals = obj['normals']

    binary = b''
    accessors = []
    bufferViews = []
    primitives = []
    materials = []
    textures = []
    images = []

    bufOffset = 0
    def addBufferView(buf, target=None):
        nonlocal bufferViews, bufOffset
        bufferViews += [pygltflib.BufferView(
            buffer=0,
            byteOffset=bufOffset,
            byteLength=len(buf),
            target=target,
        )]
        bufOffset += len(buf)
        return len(bufferViews) - 1

    for material in obj['faces'].keys():
        faces = obj['faces'][material]
        faces = np.array(faces, dtype=np.uint32)

        prim_vertices = vertices[faces[:,0:3].flatten()]
        prim_uvs = uvs[faces[:,3:6].flatten()]

        if faces.shape[1] == 9:
            prim_normals = normals[faces[:,6:9].flatten()]
            normals_blob = prim_normals.tobytes()
        else:
            prim_normals = None
            normals_blob = None

        vertices_blob = prim_vertices.tobytes()
        uvs_blob = prim_uvs.tobytes()

        binary += vertices_blob + uvs_blob
        if normals_blob is not None:
            binary += normals_blob
        
        verticesBufferView = addBufferView(vertices_blob, pygltflib.ARRAY_BUFFER)
        uvsBufferView = addBufferView(uvs_blob, pygltflib.ARRAY_BUFFER)
        normalsBufferView = None
        if normals_blob is not None:
            normalsBufferView = addBufferView(normals_blob, pygltflib.ARRAY_BUFFER)
        
        accessors += [
            pygltflib.Accessor(
                bufferView=verticesBufferView,
                componentType=pygltflib.FLOAT,
                count=len(prim_vertices),
                type=pygltflib.VEC3,
                max=prim_vertices.max(axis=0).tolist(),
                min=prim_vertices.min(axis=0).tolist(),
            ),
            pygltflib.Accessor(
                bufferView=uvsBufferView,
                componentType=pygltflib.FLOAT,
                count=len(prim_uvs),
                type=pygltflib.VEC2,
                max=prim_uvs.max(axis=0).tolist(),
                min=prim_uvs.min(axis=0).tolist(),
            ),
        ]

        if prim_normals is not None:
            accessors += [
                pygltflib.Accessor(
                    bufferView=normalsBufferView,
                    componentType=pygltflib.FLOAT,
                    count=len(prim_normals),
                    type=pygltflib.VEC3,
                    max=prim_normals.max(axis=0).tolist(),
                    min=prim_normals.min(axis=0).tolist(),
                )
            ]

        primitives += [pygltflib.Primitive(
                attributes=pygltflib.Attributes(POSITION=verticesBufferView, TEXCOORD_0=uvsBufferView, NORMAL=normalsBufferView), material=len(primitives)
            )]

    for material in obj['faces'].keys():
        texture_blob = paddedBuffer(obj['materials'][material], 4)
        binary += texture_blob
        textureBufferView = addBufferView(texture_blob)

        images += [pygltflib.Image(bufferView=textureBufferView, mimeType="image/jpeg")]
        textures += [pygltflib.Texture(source=len(images) - 1, sampler=0)]

        mat = pygltflib.Material(pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(baseColorTexture=pygltflib.TextureInfo(index=len(textures) - 1), metallicFactor=0, roughnessFactor=1), 
                alphaMode=pygltflib.OPAQUE)
        mat.extensions = {
            'KHR_materials_unlit': {}
        }
        materials += [mat]

    gltf = pygltflib.GLTF2(
        scene=0,
        scenes=[pygltflib.Scene(nodes=[0])],
        nodes=[pygltflib.Node(mesh=0)],
        meshes=[pygltflib.Mesh(
                primitives=primitives
            )],
        materials=materials,
        textures=textures,
        samplers=[pygltflib.Sampler(magFilter=pygltflib.LINEAR, minFilter=pygltflib.LINEAR)],
        images=images,
        accessors=accessors,
        bufferViews=bufferViews,
        buffers=[pygltflib.Buffer(byteLength=len(binary))],
    )

    gltf.extensionsRequired = ['KHR_materials_unlit']
    gltf.extensionsUsed = ['KHR_materials_unlit']

    if rtc_center != (None, None) and len(rtc_center) >= 2:
        gltf.extensionsUsed.append('CESIUM_RTC')
        gltf.extensions = {
            'CESIUM_RTC': {
                'center': [float(rtc_center[0]), float(rtc_center[1]), 0.0]
            }
        }

    # 添加CESIUM_RTC扩展（支持完整3D坐标）
    if rtc_center != (None, None) and len(rtc_center) >= 3:
        # 确保使用浮点数
        center = [float(coord) for coord in rtc_center[:3]]
        
        # 添加必需的扩展声明
        if not gltf.extensionsUsed:
            gltf.extensionsUsed = []
        gltf.extensionsUsed.append('CESIUM_RTC')
        
        # 创建扩展对象
        if not gltf.extensions:
            gltf.extensions = {}
        gltf.extensions['CESIUM_RTC'] = {
            "center": center
        }

    gltf.set_binary_blob(binary)

    _info("Writing...")
    gltf.save(output_glb)
    _info("Wrote %s" % output_glb)

    # if draco_compression:
    #     _info("Compressing with draco")
    #     try:
    #         compressed_glb = io.related_file_path(output_glb, postfix="_compressed")
    #         system.run('draco_transcoder -i "{}" -o "{}" -qt 16 -qp 16'.format(output_glb, compressed_glb))
    #         if os.path.isfile(compressed_glb) and os.path.isfile(output_glb):
    #             os.remove(output_glb)
    #             os.rename(compressed_glb, output_glb)
    #     except Exception as e:
    #         log.ODM_WARNING("Cannot compress GLB with draco: %s" % str(e))
            

if __name__ == '__main__':
    obj_path = "odm_texturing_25d_fixed/model/odm_textured_model_geo.obj"
    center = Point.from_xy_utm("WGS84 UTM 45N", 547792.00000003, 4946469.00000003)
    obj2glb(obj_path, "test.glb", rtc_center=(center.lon, center.lat))
    pass