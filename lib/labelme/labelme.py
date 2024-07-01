#
# Class for labelme json management
#
import json
from json import encoder
import os, cv2, copy
import os.path as osp
import json
from ..utils import MyEncoder
from ..utils import json_load
from ..processing import get_polygons_from_mask
from .polygon import Polygon


class Labelme(object):
    def __init__(self, image_info, shapes, **kwargs):
        self.shape_list = shapes
        self.image_info = image_info
        self.image_path = image_info['image_path']
        self.image_height = image_info['height']
        self.image_width = image_info['width']
        self.kwargs = kwargs

    @classmethod
    def from_json(cls, labelme_json):
        labelme_json = str(labelme_json)
        data = cls.parse_json(labelme_json)
        return cls(image_info=data['image_info'], shapes=data['shapes'])

    def __len__(self):
        return len(self.shape_list)

    def __getitem__(self, idx):
        return self.shape_list[idx]

    def iterator(self):
        for shape in self.shape_list:
            yield shape

    def clone(self):
        return copy.deepcopy(self)

    def add_shape(self, points, label, shape_type='polygon'):
        self.shape_list.append(Polygon(points=points, class_name=label, shape_type=shape_type))

    def add_polygon(self, polygon):
        assert isinstance(polygon, Polygon)
        self.shape_list.append(polygon)

    def add_shape_from_mask(self, mask, label, approx=True, epsilon=50):
        mask = mask > 0
        if mask.sum() == 0:
            return
        polygon_list = get_polygons_from_mask(mask, approx=approx, epsilon=epsilon)
        for polygon in polygon_list:
            self.add_shape(polygon, label, shape_type='polygon')        

    @staticmethod
    def parse_json(labelme_json):
        obj = json_load(labelme_json)

        # get shapes
        shapes = list()
        for shape in obj['shapes']:
            polygon = Polygon(
                points=shape['points'],
                class_name=shape['label'],
                shape_type=shape['shape_type']
            )
            shapes.append(polygon)

        image_info = {
            'image_path': obj['imagePath'].replace('\\', '/'),  # path to linux style
            'height': obj['imageHeight'], 'width': obj['imageWidth']
        }
        data = {'shapes': shapes, 'image_info':  image_info}

        return data

    def load_image(self, image_root=''):
        image_path = os.path.join(image_root, self.image_path)
        assert os.path.exists(image_path), "Cannot find image: {}".format(image_path)
        image_bgr = cv2.imread(image_path)
        image_rgb = copy.deepcopy(image_bgr[...,::-1]) # deepcopy to get around the polyline drawing error
        return image_rgb

    def draw_shapes(self, image, linewidth=20, color=(255,0,0)):
        """ draw shapes on image """
        for shape in self.iterator():
            image = shape.draw_on(image, linewidth=linewidth, color=color)
        return image

    def save_json(self, json_path):
        json_dir = osp.dirname(json_path)
        os.makedirs(json_dir, exist_ok=True)

        self.kwargs.update({
            'version': '5.1.1',
            'flags': {},
            'shapes': [shp.json_format() for shp in self.shape_list],
            'imagePath': self.image_path,
            'imageData': None,
            'imageHeight': self.image_height,
            'imageWidth': self.image_width
        })

        with open(json_path, 'w') as fid:
            json.dump(self.kwargs, fid, indent=4, cls=MyEncoder)
