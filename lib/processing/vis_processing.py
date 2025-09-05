import cv2
import numpy as np
from loguru import logger
from typing import List
from . import box_area
from . import boxes_painter
from . import is_polygon
from . import is_line
from . import is_point
from . import is_circle
from ..utils import ceil



class DrawItem:
    def __init__(self, item_list: list, color):
        self.item_list = item_list
        
        assert len(color) == 3, "Color must be a tuple of 3 values"
        self.color = tuple(np.clip(color, 0, 255).astype(np.uint8).tolist())

        self.data_check()
        self.item_type

    def add_item(self, item):
        self.item_list.append(item)
        self.data_check()
    
    def add_items(self, items):
        self.item_list.extend(items)
        self.data_check()

    def data_check(self):
        raise NotImplementedError("Subclass must implement abstract method")
    
    @property
    def item_type(self):
        raise NotImplementedError("Subclass must implement abstract method")
    
    def get_coor_edges(self):
        raise NotImplementedError("Subclass must implement abstract method")
    
    def draw(self, img, coor_to_uv, thickness=1):
        raise NotImplementedError("Subclass must implement abstract method")


class BBoxItem(DrawItem):
    def data_check(self):
        # check if the item_list is a list of tuples
        assert isinstance(self.item_list, list), "item_list must be a list"
        for item in self.item_list:
            assert box_area(item) > 0, f" !! item must be a tuple of bbox, and area > 0, {item}"

    @property
    def item_type(self):
        return "bbox"
    
    def get_coor_edges(self):
        x_min, y_min = np.inf, np.inf
        x_max, y_max = -np.inf, -np.inf
        for item in self.item_list:
            x_min = min(x_min, item[0])
            y_min = min(y_min, item[1])
            x_max = max(x_max, item[2])
            y_max = max(y_max, item[3])
        return (x_min, y_min, x_max, y_max)
    
    def draw(self, img, coor_to_uv, thickness=1):
        logger.info(f"Drawing {len(self.item_list)} {self.item_type}s")
        uv_bbox_list = list()
        for item in self.item_list:
            left, top = coor_to_uv(item[0], item[1])
            right, bottom = coor_to_uv(item[2], item[3])
            uv_bbox_list.append((left ,bottom ,right, top))
            pass
        return boxes_painter(img, uv_bbox_list, color=self.color, line_thickness=thickness)


class PolygonItem(DrawItem):
    def data_check(self):
        # check if the item_list is a list of tuples
        assert isinstance(self.item_list, list), "item_list must be a list"
        for item in self.item_list:
            assert is_polygon(item), f" !! item must be a tuple of polygon, {item}"

    @property
    def item_type(self):
        return "polygon"
    
    def get_coor_edges(self):
        x_min, y_min = np.inf, np.inf
        x_max, y_max = -np.inf, -np.inf
        for item in self.item_list:
            x_min = min(x_min, np.min(item[:, 0]))
            y_min = min(y_min, np.min(item[:, 1]))
            x_max = max(x_max, np.max(item[:, 0]))
            y_max = max(y_max, np.max(item[:, 1]))
        return (x_min, y_min, x_max, y_max)
    

class LineItem(DrawItem):
    def data_check(self):
        # check if the item_list is a list of tuples
        assert isinstance(self.item_list, list), "item_list must be a list"
        for item in self.item_list:
            assert is_line(item), f" !! item must be a tuple of line, {item}"

    @property
    def item_type(self):
        return "line"
    
    def get_coor_edges(self):
        x_min, y_min = np.inf, np.inf
        x_max, y_max = -np.inf, -np.inf
        for item in self.item_list:
            x_min = min(x_min, np.min(item[:, 0]))
            y_min = min(y_min, np.min(item[:, 1]))
            x_max = max(x_max, np.max(item[:, 0]))
            y_max = max(y_max, np.max(item[:, 1]))
        return (x_min, y_min, x_max, y_max)


class PointItem(DrawItem):
    def __init__(self, item_list: list, color, radius=3):
        super().__init__(item_list, color)
        self.radius = radius
    
    def data_check(self):
        # check if the item_list is a list of tuples
        assert isinstance(self.item_list, list), "item_list must be a list"
        for item in self.item_list:
            assert is_point(item), f" !! item must be a tuple of point, {item}"

    @property
    def item_type(self):
        return "point"
    
    def get_coor_edges(self):
        x_min, y_min = np.inf, np.inf
        x_max, y_max = -np.inf, -np.inf
        for item in self.item_list:
            x_min = min(x_min, item[0])
            y_min = min(y_min, item[1])
            x_max = max(x_max, item[0])
            y_max = max(y_max, item[1])
        return (x_min, y_min, x_max, y_max)

    def draw(self, img, coor_to_uv, thickness=1):
        logger.info(f"Drawing {len(self.item_list)} {self.item_type}s")
        uv_point_list = list()
        for item in self.item_list:
            uv_point_list.append(coor_to_uv(item[0], item[1]))
        
        for uv in uv_point_list:
            cv2.circle(img, (int(uv[0]), int(uv[1])), radius=self.radius, color=self.color, thickness=-1)
        return img


class CircleItem(DrawItem):
    # x, y, r
    def data_check(self):
        # check if the item_list is a list of tuples
        assert isinstance(self.item_list, list), "item_list must be a list"
        for item in self.item_list:
            assert is_circle(item), f" !! item must be a tuple of circle, {item}"

    @property
    def item_type(self):
        return "circle"
    
    def get_coor_edges(self):
        x_min, y_min = np.inf, np.inf
        x_max, y_max = -np.inf, -np.inf
        for item in self.item_list:
            x_min = min(x_min, item[0] - item[2])
            y_min = min(y_min, item[1] - item[2])
            x_max = max(x_max, item[0] + item[2])
            y_max = max(y_max, item[1] + item[2])
        return (x_min, y_min, x_max, y_max)


class ItemDrawer:
    def __init__(self, draw_item_obj_list: List[DrawItem], image_size = 8000, margin = 20):
        if not isinstance(draw_item_obj_list, list):
            draw_item_obj_list = [draw_item_obj_list]
        self.draw_item_obj_list = draw_item_obj_list
        self.x_min, self.y_min, self.x_max, self.y_max = self.get_coor_edges()
        self.image_size = image_size
        self.margin = margin
        self.max_distance = max(self.x_max - self.x_min, self.y_max - self.y_min) + self.margin * 2
        self.gsd = self.max_distance / self.image_size
        self.img_width = ceil((self.x_max - self.x_min + self.margin * 2) / self.gsd)
        self.img_height = ceil((self.y_max - self.y_min + self.margin * 2) / self.gsd)
        self.img = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)

    def get_coor_edges(self):
        x_min, y_min = np.inf, np.inf
        x_max, y_max = -np.inf, -np.inf

        coor_edges_list = []
        for draw_item_obj in self.draw_item_obj_list:
            coor_edges_list.append(draw_item_obj.get_coor_edges())

        for coor_edges in coor_edges_list:
            x_min = min(x_min, coor_edges[0])
            y_min = min(y_min, coor_edges[1])
            x_max = max(x_max, coor_edges[2])
            y_max = max(y_max, coor_edges[3])

        return (x_min, y_min, x_max, y_max)
    
    def coor_to_uv(self, x, y):
        u = (x - self.x_min + self.margin) / self.gsd
        v = (self.y_max - y + self.margin) / self.gsd
        return u, v

    def draw(self):
        for draw_item_obj in self.draw_item_obj_list:
            if draw_item_obj.item_type == "bbox":
                self.img = draw_item_obj.draw(self.img, self.coor_to_uv)
            elif draw_item_obj.item_type == "line":
                self.img = draw_item_obj.draw(self.img, self.coor_to_uv)
            elif draw_item_obj.item_type == "point":
                self.img = draw_item_obj.draw(self.img, self.coor_to_uv)
            elif draw_item_obj.item_type == "circle":
                self.img = draw_item_obj.draw(self.img, self.coor_to_uv)
            else:
                raise ValueError(f"Unknown item type: {draw_item_obj.item_type}")
        
        return self.img

