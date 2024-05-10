import copy
import numpy as np

from .info import DataManagerInfoExtractor
from ..utils import create_uuid
from ..utils import encode_distribution
from ..utils import encode_labelme_shape


class DataManagerSaver(DataManagerInfoExtractor):
    def _format_shape_to_polygon(self, shape, shape_type):
        assert isinstance(shape, (list, tuple, np.ndarray))
        assert len(shape) >= 2
        if None: pass
        elif shape_type == 'rectangle':
            assert len(shape) == 2, "shape as rectangle example: [(x_topleft,y_topleft), (x_bottomrignt, y_bottomright)]"
            assert isinstance(shape[0], (list, tuple, np.ndarray))
            (x1,y1),(x2,y2) = shape[0], shape[1]
            points = [[x1,y1],[x1,y2],[x2,y2],[x2,y1],[x1,y1]]
        elif shape_type == 'xyxy':
            assert len(shape) == 4, "shape as xyxy example: [x1, y1, x2, y2]"
            x1,y1,x2,y2 = shape
            points = [[x1,y1],[x1,y2],[x2,y2],[x2,y1],[x1,y1]]
        elif shape_type == 'xywh':
            assert len(shape) == 4, "shape as xyxy example: [x, y, w, h]"
            x,y,w,h = shape
            x1,y1,x2,y2 = x,y,x+w,y+h
            points = [[x1,y1],[x1,y2],[x2,y2],[x2,y1],[x1,y1]]
        elif len(shape) >= 3:
            points = [[x, y] for x, y in shape]
        else:
            raise RuntimeError("Invalid shape: {}".format(shape))
        return points

    def create_multilabel_scores_from_classid(self, classid, score=1.):
        classname = self.label2class(classid)
        return self.create_multilabel_scores_from_classname(classname)

    def create_multilabel_scores_from_classname(self, classname, score=1.):
        scores = [0] * len(self.class_list)
        classid = self.get_classid_from_classname(classname)
        scores[classid] = score
        return encode_distribution(scores)

    def create_multilabel_scores_from_prediction(self, prediction):
        """
        Create encoded scores from class prediction
        """
        scores = [0] * len(self.class_list)
        assert len(prediction) == len(scores)
        for classid, pred_score in enumerate(prediction):
            scores[classid] = pred_score
        return encode_distribution(scores)

    def create_multiclass_distribution_from_classid(self, classid, score=1.):
        return self.create_multilabel_scores_from_classid(classid, score)

    def create_multiclass_distribution_from_classname(self, classname, score=1.):
        return self.create_multilabel_scores_from_classname(classname, score)

    def create_multiclass_distribution_from_prediction(self, prediction):
        assert all(0 <= sc <= 1 for sc in prediction)
        assert abs(1 - np.sum(prediction)) < 1e-5, "prob distribution should be sum to 1: {}".format(np.sum(prediction))
        return self.create_multilabel_scores_from_prediction(prediction)

    def create_detection_instance_from_classname(self, classname, score=1., shape=[-1,-1,-1,-1], shape_type='xyxy'):
        # create class id
        assert classname in self.class_list
        points = self._format_shape_to_polygon(shape, shape_type)
        instance = {
            'uuid': create_uuid(),
            'classname': classname,
            'score': score,
            'shape_type': 'polygon',
            'points': encode_labelme_shape(points)
        } 
        return instance

    def create_detection_instance_from_classid(self, classid, score=1., shape=[-1,-1,-1,-1], shape_type='xyxy'):
        classname = self.label2class(classid)
        return self.create_detection_instance_from_classname(classname, score, shape, shape_type)
