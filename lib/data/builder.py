from ..utils import uniform_split_char
from ..utils import path_with_suffix
from ..utils import OS_exists
from ..utils import create_uuid
from ..utils import tqdm
from ..processing import cv_rgb_imread
from ..labelme import Labelme
from .class_mapper import DataManagerClassManipulator as DataManager
# special OK code: __HARD_OK__ and __OK__ instance is legitimate but would not be encoded to json
__OK__ = '__OK__'              # no json image or empty json (including those that all instances are ignored)
__HARD_OK__ = '__HARD_OK__'    # hard ok instance or image (in which case all instances are __HARD_OK__ instance)


class DatamanagerBuilder:
    def __init__(self, class_list, class_mapper={}, SKIP_INSTANCE_CLASS=[], SKIP_IMAGE_CLASS=[], VALID_CLASSNAMES=None, image_data_root='', rec_callback=None):
        self.class_list = class_list
        self.class_mapper = class_mapper
        self.SKIP_INSTANCE_CLASS = SKIP_INSTANCE_CLASS
        self.SKIP_IMAGE_CLASS = SKIP_IMAGE_CLASS
        if VALID_CLASSNAMES is None:
            self.VALID_CLASSNAMES = class_list
        else:
            self.VALID_CLASSNAMES = VALID_CLASSNAMES
        self.image_data_root = uniform_split_char(image_data_root)
        self.rec_callback = rec_callback
        self.dt = DataManager(class_list=self.class_list)
        pass

    def map_class(self, clsname):
        if clsname not in self.class_mapper:
            return clsname
        else:
            return self.class_mapper[clsname]

    def imagepath2record(self, image_path):
        image_path = uniform_split_char(image_path)
        unregistered_classes = list()
        json_path = path_with_suffix(image_path, 'json')

        if OS_exists(json_path):
            labelme_obj = Labelme.from_json(json_path)
            H, W = labelme_obj.image_height, labelme_obj.image_width

            # extract all instances
            instances = list()
            if len(labelme_obj.shape_list) == 0:
                instance_level_classes = {__OK__}
            else:
                instance_level_classes = set()
                for shape in labelme_obj.shape_list:
                    class_name = self.map_class(shape.class_name)

                    if class_name in self.SKIP_INSTANCE_CLASS:
                        class_name = __OK__
                    elif class_name in self.SKIP_IMAGE_CLASS:
                        instance_level_classes.add(class_name)
                        break
                    elif class_name not in self.VALID_CLASSNAMES:
                        unregistered_classes.append(class_name)
                        # raise TypeError("class '{}' not found, skip this instance: {}".format(class_name, image_path))

                    if class_name in self.class_list:
                        inst = self.dt.create_detection_instance_from_classname(
                            class_name, shape=shape.points, shape_type='polygon')
                        instances.append(inst)

                    instance_level_classes.add(class_name)


            # determine the image level class
            assert len(instance_level_classes) > 0
            if instance_level_classes == {__OK__, __HARD_OK__}:
                instance_level_classes = {__HARD_OK__}
            elif instance_level_classes == {__OK__}:
                pass
            else:
                if __OK__ in instance_level_classes:
                    instance_level_classes.remove(__OK__)
                if __HARD_OK__ in instance_level_classes:
                    instance_level_classes.remove(__HARD_OK__)

        else:
            instances = list()
            img = cv_rgb_imread(image_path)
            H, W = img.shape[:2]
            instance_level_classes = {__OK__}

        if len(instance_level_classes.intersection(self.SKIP_IMAGE_CLASS)) > 0:
            return None

        if len(unregistered_classes) == 0:
            info = {
                'uuid': create_uuid(),
                'image_path': image_path.replace(self.image_data_root.rstrip('/'), '').lstrip('/'),
                'height': H, 'width': W,
            }
            return dict(info = info, instances = instances)
        else:
            return unregistered_classes

    def record_call_back(self, record):
        if self.rec_callback is None:
            return record
        else:
            return self.rec_callback(record)

    def make_cnn_json(self, data_path_list):
        cnn_json_obj = DataManager(class_list=self.class_list)
        record_list = list()
        for data_path in tqdm(data_path_list):
            record = self.imagepath2record(data_path)
            if record is not None:
                record = self.record_call_back(record)
                record_list.append(record)
        cnn_json_obj.record_list = record_list
        return cnn_json_obj