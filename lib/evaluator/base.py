from tqdm import tqdm
from ..data import DataManager
from ..utils import ListOrderedDict


class EvaluatorBase(object):
    TYPE = None
    def __init__(self, data_gt, data_pred, verbose=True):
        """
        :classname_list: list of classes that we are interested in
        """
        self.verbose = verbose

        # load datamanager
        self._dm_gt = self._load_dm_obj(data_gt)
        self._dm_pred = self._load_dm_obj(data_pred)

        # get am obj from dm
        self._get_am(self._dm_gt, self._dm_pred)
        self.class_list = data_gt.class_list
        self.classnames = data_gt.class_list

        # check datamanager obj before evaluation
        self._eva_check(self._dm_gt, self._dm_pred)
        self.gt_data_list, self.pred_data_list, self.data_path_list = self.extract_data_pair(self._dm_gt, self._dm_pred)

    def _data_check(self, gt_data, pd_data):
        assert gt_data.class_list == pd_data.class_list, f"differect class_list between {gt_data.class_list} and {pd_data.class_list}"
        assert gt_data.task_type.lower() == pd_data.task_type.lower() == self.TYPE.lower(), f"task type error: gt_data.task_type {gt_data.task_type}, pd_data.task_type {pd_data.task_type}, self.TYPE {self.TYPE}"

    def extract_data_pair(self, dm_gt, dm_pred):
        self._logging(f"@@ extracting data pair.")
        _gt_data_dict = self._get_gt_result_dict(dm_gt)
        _pred_data_dict = self._get_pred_result_dict(dm_pred)
        _data_path_dict = self._get_data_path_dict(dm_gt)

        assert isinstance(_gt_data_dict, dict), f"{_gt_data_dict} should be an obj of dict"
        assert isinstance(_pred_data_dict, dict), f"{_pred_data_dict} should be an obj of dict"
        assert isinstance(_data_path_dict, dict), f"{_data_path_dict} should be an obj of dict"
        assert len(dm_gt) == len(dm_pred) == len(_gt_data_dict) == len(_pred_data_dict) == len(_data_path_dict), f"data num error : {len(dm_gt)} {len(dm_pred)} {len(_gt_data_dict)} {len(_pred_data_dict)} {len(_data_path_dict)}"
        assert set(_gt_data_dict) == set(_pred_data_dict) == set(_data_path_dict), f"uuid keys of dm objs do not match."

        self._logging(f"@@ total data {len(dm_gt)}, start packing data pair.")
        gt_data_list = list()
        pred_data_list = list()
        data_path_list = list()
        for key in tqdm(list(_gt_data_dict), desc="Packing data: "):
            gt_data_list.append(_gt_data_dict[key])
            pred_data_list.append(_pred_data_dict[key])
            data_path_list.append(_data_path_dict[key])

        return gt_data_list, pred_data_list, data_path_list

    @staticmethod
    def _get_am(dm_gt, dm_pred):
        assert dm_gt.class_list == dm_pred.class_list, f"class_dicts are not same. {dm_gt.class_list}, {dm_pred.class_list}"
        assert type(dm_gt.task_type) == type(dm_pred.task_type), f"AM data type {type(dm_gt.task_type)}, {dm_pred.task_type}"

    def _load_dm_obj(self, data):
        if isinstance(data, str):
            return DataManager.load(data)
        elif isinstance(data, DataManager):
            return data
        else:
            raise RuntimeError("Unknown data format: {}".format(type(data)))

    def _logging(self, data=None):
        if data is None:
            return getattr(self, 'verbose', False)
        else:
            if getattr(self, 'verbose', False):
                print(data)

    def _eva_check(self, data_gt, data_pred):
        # type check
        assert type(data_gt) == type(data_pred), f"type of data_gt {type(data_gt)} and data_pred {type(data_pred)} are not same."
        if 'DMML' in str(type(data_gt)):
            assert self.TYPE == 'MultiLabel', f"data's type {str(type(data_gt))} is not same as evaluator's {self.TYPE}"
        if 'DMMC' in str(type(data_gt)):
            assert self.TYPE == 'MultiClass', f"data's type {str(type(data_gt))} is not same as evaluator's {self.TYPE}"
        if 'DMDet' in str(type(data_gt)):
            assert self.TYPE == 'Detection', f"data's type {str(type(data_gt))} is not same as evaluator's {self.TYPE}"
        self._logging("@@ Checking input data.")
        assert isinstance(data_gt, DataManager), f"data_gt {data_gt} should be an obj of DataManager"
        assert isinstance(data_pred, DataManager), f"data_gt {data_pred} should be an obj of DataManager"
        assert data_gt and data_pred, "Please load both gt and pred before evaluation."
        assert data_gt == data_pred, "Make sure the gt data and pred data match"
        self._data_check(data_gt, data_pred)

    def set_threshold(self, policy='inflection', value=None, *args, **kwargs):
        """
        Args:
            policy (str, optional): could be one of 'inflection', 'recall', 'precision' or 'manual'. Defaults to 'inflection'.
            value (optional): 
                under 'inflection' policy: value is insignificant
                under 'recall' or 'precision': value should be the minimum float we can accept
                under 'manual': value can be a list or dict who has the same length or keys with class_list. 
                . Defaults to None.
        """
        thre_dict = self._set_threshold(policy=policy, value=value, *args, **kwargs)
        assert len(thre_dict) == len(self.class_list), f"data length error {len(thre_dict)} != {len(self.class_list)}"
        assert set(thre_dict) == set(self.class_list), f"duplicate key error {set(thre_dict)} != {set(self.class_list)}"
        for _, value in thre_dict.items():
            assert 0 <= value <= 1, f"bad threshold value {value} in {thre_dict}"
        threshold_dict = ListOrderedDict()
        for key in self.class_list:
            threshold_dict[key] = thre_dict[key]
        self.threshold_dict = threshold_dict

    def _get_data_path_dict(self, data):
        data_path_dict = dict()
        for rec in data:
            uuid = rec['info']['uuid']
            data_path = rec['info']['image_path']
            assert uuid not in data_path_dict, f"duplicate uuid in dataset {uuid}"
            data_path_dict[uuid] = data_path
        return data_path_dict

    def _get_gt_result_dict(self, dm_gt):
        raise NotImplementedError(f"Base Evaluator func, _get_gt_result_dict, not Implemented")

    def _get_pred_result_dict(self, dm_pred):
        raise NotImplementedError(f"Base Evaluator func, _get_pred_result_dict, not Implemented")

    def eval_ap(self, *args, **kwargs):
        raise NotImplementedError(f"Base Evaluator func, _eval_ap, not Implemented")

    def eval_judgment(self, *args, **kwargs):
        raise NotImplementedError(f"Base Evaluator func, _eval_ap, not Implemented")
    
    def _set_threshold(self, *args, **kwargs):
        raise NotImplementedError(f"Base Evaluator func, _set_threshold, not Implemented")

    def get_confusion_matrix(self):
        raise NotImplementedError(f"Base Evaluator func, get_confusion_matrix, not Implemented")
    
    def dump_failure_case(self, data_root, target_dir):
        raise NotImplementedError(f"Base Evaluator func, dump_failure_case, not Implemented")
