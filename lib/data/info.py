import copy
import numpy as np

from .format import DataManagerFormater
from ..utils import decode_distribution
from ..utils import decode_labelme_shape


class DataManagerInfoExtractor(DataManagerFormater):
    @property
    def task_type(self):
        record_types = set(self.extract_info(self._get_record_task_type))
        assert len(record_types) in [0, 1], "Multiple task types exist in one dataset!"
        return record_types.pop() if len(record_types) else 'detection'

    def class2label(self, classname):
        for classid, name in enumerate(self.class_list):
            if name == classname:
                return classid
        raise RuntimeError("Unrecognized class: {}".format(classname))

    def label2class(self, classid):
        return self.class_list[classid]

    ## Implementation of different classname getting stuffs
    # 1) get classname
    def get_detection_classname_list_from_record(self, record):
        assert self._get_record_task_type(record) == 'detection'
        return [self.get_detection_classname_from_instance(inst) for inst in record['instances']]

    def get_detection_classname_from_instance(self, instance):
        return instance['classname']

    def get_multiclass_classname_from_record(self, record):
        assert self._get_record_task_type(record) == 'multiclass'
        distribution = decode_distribution(record['distribution'])
        return self.get_multiclass_classname_from_distribution(distribution)

    def get_multiclass_classname_from_distribution(self, distribution):
        assert len(distribution) == len(self.class_list)
        classid = np.argmax(distribution)
        return self.label2class(classid)

    def get_classname_from_classid(self, classid):
        return self.label2class(classid)

    # 2) get classid
    def get_detection_classid_list_from_record(self, record):
        assert self._get_record_task_type(record) == 'detection'
        return [self.get_detection_classid_from_instance(inst) for inst in record['instances']]

    def get_detection_classid_from_instance(self, instance):
        return self.class2label(instance['classname'])

    def get_multiclass_classid_from_record(self, record):
        assert self._get_record_task_type(record) == 'multiclass'
        distribution = decode_distribution(record['distribution'])
        return self.get_multiclass_classid_from_distribution(distribution)

    def get_multiclass_classid_from_distribution(self, distribution):
        assert len(distribution) == len(self.class_list)
        classid = np.argmax(distribution)
        return classid

    def get_classid_from_classname(self, classname):
        return self.class2label(classname)

    # 3) get score (detection/multiclass)
    def get_detection_score_list_from_record(self, record):
        assert self._get_record_task_type(record) == 'detection'
        return [self.get_detection_score_from_instance(inst) for inst in record['instances']]
    
    def get_detection_score_from_instance(self, instance):
        return instance['score']

    def get_multiclass_score_from_record(self, record):
        distribution = self.get_multiclass_distribution_from_record(record)
        return self.get_multiclass_score_from_distribution(distribution)

    def get_multiclass_score_from_distribution(self, distribution):
        distribution = decode_distribution(distribution)
        return np.max(distribution)

    # 4) get scores (multilabel)
    def get_multilabel_scores_from_record(self, record):
        assert self._get_record_task_type(record) == 'multilabel'
        return decode_distribution(record['scores'])

    # 5) get distribution (multiclass)
    def get_multiclass_distribution_from_record(self, record):
        assert self._get_record_task_type(record) == 'multiclass'
        return decode_distribution(record['distribution'])

    # 6) get shape/xyxy/xywh (detection)
    def get_detection_shape_list_from_record(self, record):
        assert self._get_record_task_type(record) == 'detection'
        return [self.get_detection_shape_from_instance(inst) for inst in record['instances']]

    def get_detection_shape_from_instance(self, instance):
        assert 'shape_type' in instance
        if None: pass
        elif instance['shape_type'] == 'polygon':
            points = decode_labelme_shape(instance['points'])
            points = np.reshape(points, [-1,2])
            return points.tolist()
        elif instance['shape_type'] == 'rectangle':
            (x1,y1),(x2,y2) = decode_labelme_shape(instance['points'])
            return [[x1,y1],[x1,y2],[x2,y2],[x2,y1],[x1,y1]]
        else:
            raise RuntimeError("Unknown shape type: {}".format(instance['shape_type']))

    def get_detection_xyxy_list_from_record(self, record):
        assert self._get_record_task_type(record) == 'detection'
        return [self.get_detection_xyxy_from_instance(inst) for inst in record['instances']]

    def get_detection_xyxy_from_instance(self, instance):
        points = self.get_detection_shape_from_instance(instance)
        points = np.reshape(points, [-1,2])
        x1,y1,x2,y2 = points[:,0].min(), points[:,1].min(), points[:,0].max(), points[:,1].max()
        return [x1,y1,x2,y2]

    def get_detection_xywh_list_from_record(self, record):
        assert self._get_record_task_type(record) == 'detection'
        return [self.get_detection_xywh_from_instance(inst) for inst in record['instances']]

    def get_detection_xywh_from_instance(self, instance):
        x1,y1,x2,y2 = self.get_detection_xyxy_from_instance(instance)
        return [x1, y1, x2-x1, y2-y1]

    def data_statistics(self):
        data_task_type = self.task_type
        if data_task_type == 'detection':
            get_classname = self.get_detection_classname_list_from_record
            class_occurence_by_instance = self.occurrence(get_classname)
            class_occurence_by_image = self.occurrence(lambda rec: list(set(get_classname(rec))))
            ok_ng_counter = self.occurrence(lambda rec: len(rec['instances'])>0)
            return {
                '#images': len(self),
                '#ok_images': ok_ng_counter.get(False, 0),
                '#ng_images': ok_ng_counter.get(True,  0),
                '#classes_by_image': class_occurence_by_image,
                '#classes_by_instance': class_occurence_by_instance
            }
        elif data_task_type == 'multiclass':
            class_occurrence = self.occurrence(self.get_multiclass_classname_from_record)
            return {
                '#images': len(self),
                '#classes': class_occurrence
            }
        else:
            assert data_task_type == 'multilabel'
            return {'#images': len(self)}