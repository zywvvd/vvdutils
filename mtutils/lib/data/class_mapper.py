import copy

from .saver import DataManagerSaver
from ..utils import encode_distribution


class DataManagerClassManipulator(DataManagerSaver):
    def map_class(self, class_mapping_dict):
        """
        Manipulate the dataset classes by remapping current class_list to a new one

        Args:
            class_mapping_dict: a dictionary, which takes form of {old_classname: new_classname},
            if old_classname not found in class_mapping_dict, it remains the same

        Return:
            a new dataset with new class_list, and the class order is sorted
        """
        assert set(class_mapping_dict.keys()).issubset(self.class_list), \
            "Unknown classname (key) found: ".format(class_mapping_dict.keys())
        # 0) preparation
        oldclass2newclass = dict()
        for classname in self.class_list:
            if classname in class_mapping_dict:
                oldclass2newclass[classname] = class_mapping_dict[classname]
            else:
                oldclass2newclass[classname] = classname

        # 1) get new class_list
        new_class_list = sorted(list(set(oldclass2newclass.values())))
        new_class2label = {classname:classid for classid, classname in enumerate(new_class_list)}

        # 2) get new record_list
        data_task_type = self.task_type
        new_record_list = list()
        for record in self.record_list:
            new_record = copy.deepcopy(record)
            if data_task_type == 'detection':
                for inst in new_record['instances']:
                    old_classname = self.get_detection_classname_from_instance(inst)
                    new_classname = oldclass2newclass[old_classname]
                    inst['classname'] = new_classname
            elif data_task_type == 'multiclass':
                old_distribution = self.get_multiclass_distribution_from_record(record)
                new_distribution = [0] * len(new_class_list)
                for old_classid, old_classname in enumerate(self.class_list):
                    old_score = old_distribution[old_classid]
                    new_classname = oldclass2newclass[old_classname]
                    new_classid = new_class2label[new_classname]
                    new_distribution[new_classid] += old_score
                new_record['distribution'] = encode_distribution(new_distribution)
            else:
                assert data_task_type == 'multilabel'
                old_scores = self.get_multilabel_scores_from_record(record)
                new_scores = [0] * len(new_class_list)
                for old_classid, old_classname in enumerate(self.class_list):
                    old_score = old_scores[old_classid]
                    new_classname = oldclass2newclass[old_classname]
                    new_classid = new_class2label[new_classname]
                    new_scores[new_classid] = max(old_score, new_scores[new_classid])
                new_record['scores'] = encode_distribution(new_scores)

            new_record_list.append(new_record)

        assert len(new_record_list) == len(self.record_list)

        return type(self)(record_list=new_record_list, class_list=new_class_list)
