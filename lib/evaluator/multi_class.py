from tqdm import tqdm

from sklearn.metrics import confusion_matrix
from pathlib import Path
from ..utils import encode_path
from ..processing import cv_rgb_imwrite
from ..processing import cv_rgb_imread
from .base import EvaluatorBase
import pandas as pd


class ClassificationMultiClassEvaluator(EvaluatorBase):
    TYPE = 'MultiClass'

    def _get_result_dict(self, dm):
        result_dict = dict()
        for rec in dm:
            class_name = dm.get_multiclass_classname_from_record(rec)
            result_dict[rec['info']['uuid']] = class_name
        return result_dict

    def _get_gt_result_dict(self, dm_gt):
        return self._get_result_dict(dm_gt)

    def _get_pred_result_dict(self, dm_pred):
        return self._get_result_dict(dm_pred)

    def get_confusion_matrix(self):
        y_true = self.gt_data_list
        y_pred = self.pred_data_list
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=self.class_list)
        cm_pd = pd.DataFrame(cm, index=self.class_list, columns=self.class_list)
        return cm_pd

    def dump_failure_case(self, data_root, target_dir):
        target_dir = Path(target_dir)
        for y_true, y_pred, data_path in tqdm(zip(self.gt_data_list, self.pred_data_list, self.data_path_list), total=len(self.gt_data_list)):
            if y_true != y_pred:
                ori_img_path = Path(data_root) / data_path
                if not ori_img_path.exists():
                    print(f"Ori image path {ori_img_path} not found.")
                    continue
                else:
                    try:
                        image = cv_rgb_imread(ori_img_path)
                    except Exception as e:
                        print(f"image load failed! {ori_img_path}")
                        print(f"error message: {e}")
                        continue
                gt_label = y_true
                pred_label = y_pred
                save_path = target_dir / ('GT_' + gt_label) / ('PRED_' + pred_label) / encode_path(ori_img_path)
                cv_rgb_imwrite(image, save_path)


MultiClassEvaluator = ClassificationMultiClassEvaluator
