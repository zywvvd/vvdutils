import copy
import numpy as np

from .base import DataManagerBase
from ..utils import encode_distribution
from ..utils import decode_distribution


class DataManagerFormater(DataManagerBase):
    def __init__(self, record_list=[], class_list=[]):
        assert isinstance(record_list, list)
        assert isinstance(class_list,  list)
        self.class_list  = self._extract_leaf_classes(class_list)
        self.record_list = self._format_record_list(record_list)


    def _format_distribution(self, old_distribution):
        """
        Format distribution/scores from legacy detection/multiclass/multilabel data
        The legacy data contains parent classes, which are incompatible with current
        version of dataset, we need to fix that 

        Args:
            distribution: 1D numpy array or list

        Return:
            a new distribution aligned with self.class_list
        """
        _old_class2label = {x['class_name']:x['class_id'] for x in self._old_class_dict}
        new_distribution = list()
        for classname in self.class_list:
            classid = _old_class2label[classname]
            new_distribution.append(old_distribution[classid])
        return new_distribution


    def _format_record(self, record, target_task=None):
        """
        Format record

        Args:
            record: you want to format
            target_task: task type you want format the record to (None: keep the same)

        Return:
            A reformated record
        """
        source_type = self._get_record_task_type(record)
        target_task = source_type.split('-')[0] if target_task is None else target_task
        if source_type == 'detection-legacy':
            # 1) covert standard detection task
            new_record = copy.deepcopy(record)
            for inst in new_record['instances']:
                old_distribution = decode_distribution(inst.pop('distribution'))
                distribution = self._format_distribution(old_distribution)
                classid = np.argmax(distribution)
                score, classname = distribution[classid], self.class_list[classid]
                inst.update({
                    'score': score,
                    'classname': classname,
                })

            # 2) convert to target task type
            if target_task == 'detection':
                return new_record
            elif target_task == 'multilabel':
                instances  = new_record.pop('instances')
                new_scores = [0] * len(self.class_list)
                for inst in instances:
                    classname, score = inst['classname'], inst['score']
                    classid = self.class2label(classname)
                    new_scores[classid] = max(score, new_scores[classid])
                new_record['scores'] = encode_distribution(new_scores)
            else:
                raise RuntimeError("Can't format record from task '{}' to '{}'".format(source_type, target_task))

        elif source_type == 'detection':
            new_record = copy.deepcopy(record)
            if target_task == 'detection':
                for inst in new_record['instances']:
                    if 'class_name' in inst:
                        inst['classname'] = inst.pop('class_name')
            elif target_task == 'multilabel':
                instances = new_record.pop('instances')
                new_scores = [0] * len(self.class_list)
                for inst in instances:
                    classname, score = inst['classname'], inst['score']
                    classid = self.class2label(classname)
                    new_scores[classid] = max(score, new_scores[classid])
                new_record['scores'] = encode_distribution(new_scores)
            else:
                raise RuntimeError("Can't format record from task '{}' to '{}'".format(source_type, target_task))

        elif source_type == 'multiclass-legacy':
            if target_task == 'multiclass':
                new_record = copy.deepcopy(record)
                old_distribution = decode_distribution(new_record['info'].pop('distribution'))
                distribution = self._format_distribution(old_distribution)
                new_record['distribution'] = encode_distribution(distribution)
            else:
                raise RuntimeError("Can't format record from task '{}' to '{}'".format(source_type, target_task))

        elif source_type == 'multiclass':
            if target_task == 'multiclass':
                new_record = copy.deepcopy(record)
            else:
                raise RuntimeError("Can't format record from task '{}' to '{}'".format(source_type, target_task))

        elif source_type == 'multilabel-ancient':
            if target_task == 'multilabel':
                new_record = copy.deepcopy(record)
                instances = new_record.pop('instances')
                assert len(instances) in [0, 1], instances
                new_record['scores'] = [0] * len(self.class_list)
                for inst in instances:
                    old_scores = decode_distribution(inst.pop('distribution'))
                    new_scores = self._format_distribution(old_scores)
                    new_record['scores'] = encode_distribution(new_scores)
            else:
                raise RuntimeError("Can't format record from task '{}' to '{}'".format(source_type, target_task))

        elif source_type == 'multilabel-legacy':
            if target_task == 'multilabel':
                new_record = copy.deepcopy(record)
                old_distribution = decode_distribution(new_record['info'].pop('scores'))
                distribution = self._format_distribution(old_distribution)
                new_record['scores'] = encode_distribution(distribution)
            else:
                raise RuntimeError("Can't format record from task '{}' to '{}'".format(source_type, target_task))

        elif source_type == 'multilabel':
            if target_task == 'multilabel':
                new_record = copy.deepcopy(record)
            else:
                raise RuntimeError("Can't format record from task '{}' to '{}'".format(source_type, target_task))

        else:
            raise RuntimeError("Unrecognized record task: {}".format(source_type))

        return new_record

    def _format_record_list(self, record_list, target_task=None):
        """
        Format record list

        Args:
            record_list: list of record
            target_task: task type you want format the record to

        Return:
            A reformated record list
        """
        new_record_types = [self._format_record(rec, target_task) for rec in record_list]
        return new_record_types

    def _extract_leaf_classes(self, class_list):
        self._old_class_dict = class_list

        leaf_class_list = list()
        for x in class_list:
            if isinstance(x, dict): # this indicates that we are dealing with the lagecy dataset
                nonleaf_classes = set([x['parent'] for x in class_list if x['parent'] is not None])
                if x['class_name'] not in nonleaf_classes:
                    leaf_class_list.append(x['class_name'])
            elif isinstance(x, str):
                return class_list
            else:
                raise TypeError("Unrecognized class list")
        return leaf_class_list

    def _get_record_task_type(self, record):
        """
        Return the type of the given record
        """
        assert isinstance(record, dict)
        assert 'info' in record

        # detection record
        if 'instances' in record:
            modes = set()
            instances = record['instances']
            if len(instances) > 0:
                for inst in instances:
                    if 'distribution' in inst:
                        assert 'shape_type' in inst, inst
                        if inst['shape_type'] is None:
                            modes.add('multilabel-ancient')
                        else:
                            modes.add('detection-legacy')
                    else:
                        modes.add('detection')
                        assert 'score' in inst, "Unknown instances: {}".format(inst)

            if len(modes) == 0:
                return 'detection'
            else:
                assert len(modes) == 1, "Unknown detection type: {}".format(record)
                return modes.pop()

        # classification record
        if 'scores' in record['info']:
            return 'multilabel-legacy'
        elif 'scores' in record:
            return 'multilabel'
        elif 'distribution' in record['info']:
            return 'multiclass-legacy'
        elif 'distribution' in record:
            return 'multiclass'
        else:
            raise TypeError('Unknown record type: {}'.format(record))

    def _det2ml(self):
        data_task_type = self.task_type
        assert data_task_type == 'detection', \
            "Only detection dataset can be converted to multilabel dataset: {}".format(data_task_type)
        new_record_list = []
        for rec in self.record_list:
            scores = [0]*len(self.class_list)
            for inst in rec['instances']:
                classid = self.class2label(inst['classname'])
                scores[classid] = max(scores[classid], inst['score'])
            new_record = {
                'info': copy.deepcopy(rec['info']),
                'scores': encode_distribution(scores),
            }
            new_record_list.append(new_record)
        return type(self)(record_list=new_record_list, class_list=self.class_list)

    def _ml2mc(self):
        data_task_type = self.task_type
        assert data_task_type == 'multilabel'
        print(
            "Warning: Convert from Multilabel to Multiclass is not legitimate, \n"
            "We adopt a simple strategy for doing that: \n"
            "multiclass distribution = multilabel scores / sum(multilabel scores)"
        )
        new_record_list = []
        for rec in self.record_list:
            scores = self.get_multilabel_scores_from_record(rec)
            distribution = scores / (np.sum(scores) + 1e-12)
            new_record = {
                'info': copy.deepcopy(rec['info']),
                'distribution': encode_distribution(distribution),
            }
            new_record_list.append(new_record)
        return type(self)(record_list=new_record_list, class_list=self.class_list)

    def _mc2ml(self):
        data_task_type = self.task_type
        assert data_task_type == 'multiclass'
        print(
            "Warning: Convert from Multiclass to Multilabel is not legitimate, \n"
            "We adopt a simple strategy for doing that: \n"
            "multilabel scores = multiclass distribution"
        )
        new_record_list = []
        for rec in self.record_list:
            distribution = self.get_multiclass_distribution_from_record(rec)
            new_record = {
                'info': copy.deepcopy(rec['info']),
                'scores': encode_distribution(distribution),
            }
            new_record_list.append(new_record)
        return type(self)(record_list=new_record_list, class_list=self.class_list)

    def to_ml(self):
        data_task_type = self.task_type
        if data_task_type == 'detection':
            return self._det2ml()
        elif data_task_type == 'multilabel':
            return self.clone()
        else:
            return self._mc2ml()

    def to_mc(self):
        data_task_type = self.task_type
        if data_task_type == 'detection':
            data_ml = self._det2ml()
            return data_ml.to_mc()
        elif data_task_type == 'multilabel':
            return self._ml2mc()
        else:
            return self.clone()
