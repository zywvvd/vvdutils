import numpy as np
import pandas as pd
from tqdm import tqdm

from pathlib import Path

from ..utils import ClassifierEvalMultilabel

from ..utils import get_TFPN
from .base import EvaluatorBase

from ..processing import cv_rgb_imread
from ..processing import cv_rgb_imwrite
from ..utils import encode_path
from ..utils import dir_check
from ..processing import puzzle


class ClassificationMultiLabelEvaluator(EvaluatorBase):
    TYPE = 'MultiLabel'

    def _get_result_dict(self, dm):
        result_dict = dict()
        for rec in dm:
            scores = dm.get_multilabel_scores_from_record(rec)
            result_dict[rec['info']['uuid']] = scores
        return result_dict

    def _get_gt_result_dict(self, dm_gt):
        return self._get_result_dict(dm_gt)

    def _get_pred_result_dict(self, dm_pred):
        return self._get_result_dict(dm_pred)

    def compute_wmAP(self, df):
        """ Compute mAP weighted by the actual number ground truth class """
        class_occurrence_list = np.sum(self.gt_data_list, axis=0).tolist()
        total_gt = max(1, np.sum(class_occurrence_list))
        class_weights = {key:val / total_gt for key, val in zip(self.class_list, class_occurrence_list)}
        wmAP = sum([class_weights[row.name] * row.AP for _, row in df.iterrows()])
        return wmAP

    def eval_miss_fa(self, threshold, class_list=None):
        
        annotation_distribution_list, prediction_distribution_list = self.gt_data_list, self.pred_data_list
        classnames = self.class_list
        if class_list is None:
            class_list = self.class_list

        for class_name in class_list:
            assert class_name in classnames, f"classname {class_name} not in classnames {classnames}."
        index_list = list(map(self._dm_gt.get_classid_from_classname, class_list))

        interested_ann_array = np.array(annotation_distribution_list)[:, index_list]
        interested_pred_array = (np.array(prediction_distribution_list)[:, index_list] > threshold).astype('float64')
        
        class_results = dict()
        
        for index in range(len(class_list)):
            ann_array = interested_ann_array[:, index]
            pred_array = interested_pred_array[:, index]
            info_dict = get_TFPN(ann_array, pred_array)

            class_results[class_list[index]] = info_dict
        return class_results

    def eval_ap(self, verbose=True):
        annotation_distribution_list = self.gt_data_list
        prediction_distribution_list = self.pred_data_list

        # compute ap
        inflection_info = ClassifierEvalMultilabel.get_inflection_info(
            y_true  = np.array(annotation_distribution_list),
            y_score = np.array(prediction_distribution_list),
            class_name_list = self.class_list
        )
        p_at_r_100 = ClassifierEvalMultilabel.compute_p_at_r(            
            y_true  = np.array(annotation_distribution_list),
            y_score = np.array(prediction_distribution_list),
            class_name_list = self.class_list,
            recall_thresh=1
        )
        thr_at_p_99 = ClassifierEvalMultilabel.get_threshold_at_p(            
            y_true  = np.array(annotation_distribution_list),
            y_score = np.array(prediction_distribution_list),
            class_name_list = self.class_list,
            precision_thresh=0.99
        )
        p_at_r_99 = ClassifierEvalMultilabel.compute_p_at_r(            
            y_true  = np.array(annotation_distribution_list),
            y_score = np.array(prediction_distribution_list),
            class_name_list = self.class_list,
            recall_thresh=0.99
        )
        p_at_r_95 = ClassifierEvalMultilabel.compute_p_at_r(            
            y_true  = np.array(annotation_distribution_list),
            y_score = np.array(prediction_distribution_list),
            class_name_list = self.class_list,
            recall_thresh=0.95
        )
        ap_dict = ClassifierEvalMultilabel.compute_ap(
            y_true  = np.array(annotation_distribution_list),
            y_score = np.array(prediction_distribution_list),
            class_name_list = self.class_list
        )

        ap_info = dict()
        for key in self.class_list:
            res_dict = {
                'ap': ap_dict[key],
                'inflection': inflection_info[key],
                'p_at_r_99': p_at_r_99[key],
                'p_at_r_100': p_at_r_100[key],
                'p_at_r_95': p_at_r_95[key],
                'thr_at_p_99': thr_at_p_99[key]
            }
            ap_info[key] = res_dict

        df = pd.DataFrame.from_dict(ap_dict, orient='index', columns=['AP'])
        df_notnull = df[df['AP'].notnull()]  # remove NaN data (No samples of this class in gt)
        wmAP = self.compute_wmAP(df_notnull)

        # print evaluation
        if verbose:
            if len(df_notnull) > 1:
                # inflection info
                print()
                print('---------------')
                print('inflection info')
                df_inflection = pd.DataFrame.from_dict(inflection_info).T
                print(df_inflection)

                # p_at_r_99
                print()
                print('---------------')
                print('precision at recall 0.99')
                p_at_r_99_pd = pd.DataFrame.from_dict(p_at_r_99, orient='index', columns=['precision_at_recall_0.99'])
                print(p_at_r_99_pd)

                # threshold at precision 0.99
                print()
                print('---------------')
                print('threshold at precision 0.99')
                thr_at_p_99_pd = pd.DataFrame.from_dict(thr_at_p_99, orient='index', columns=['threshold_at_precision_0.99'])
                print(thr_at_p_99_pd)

                # ap info
                print()
                print('---------------')
                print(df_notnull)
                mAP = df_notnull.mean()['AP']
                print('------------')
                print(' mAP: {:.3f}'.format(mAP))
                print('wmAP: {:.3f}\n'.format(wmAP))

        ap_result = {
            'AP': ap_info,
            'wmAP': wmAP
        }

        self.ap_result = ap_result
        return ap_result

    def _set_threshold(self, policy='inflection', value=None):
        """
        Args:
            policy (str, optional): could be one of 'inflection' or 'manual'. Defaults to 'inflection'.
            value (optional): 
                under 'inflection' policy: value is insignificant
                under 'recall' or 'precision': value should be the minimum float we can accept
                under 'manual': value can be a list or dict who has the same length or keys with classnames. 
                . Defaults to None.
        """
        assert policy in ['inflection', 'manual'], f"policy {policy} should in ['inflection', 'recall', 'precision', 'manual']"
        threshold_dict = dict()

        if policy != 'manual':
            if not hasattr(self, 'ap_result'):
                print("@@ Warning: You did not run eval_ap manually, and we have to run it with default args.")
                self.eval_ap(verbose=False)
            assert hasattr(self, 'ap_result'), f"inflection threshold need self.ap_result, and we can not find it."

        if policy == 'inflection':
            ap_res = self.ap_result['AP']
            for key, ap_data in ap_res.items():
                inflection = ap_data['inflection']['inflection_score']

                threshold_dict[key] = inflection

        elif policy == 'manual':
            if isinstance(value, dict):
                threshold_dict = value
            else:
                assert isinstance(value, list), f"under manual threshold policy, value should be dict or list. {value}"
                assert len(value) == len(self.class_list), f"value list is not as long as class_list {len(value)} != {len(self.class_list)}"
                for index in range(len(self.class_list)):
                    threshold_dict[self.class_list[index]] = value[index]
        else:
            raise RuntimeError(f"unknown policy {policy}")

        self.threshold_dict = threshold_dict
        return threshold_dict

    def eval_judgment(self):
        if not hasattr(self, 'threshold_dict'):
            print("@@ Warning: You did not run set_threshold manually, and we have to run it with default args.")
            self.set_threshold()
        assert hasattr(self, 'threshold_dict'), f"eval_judgment failed for threshold_dict not found."
        pred_score_array = np.array(self.pred_data_list)
        thre_array = np.array([self.threshold_dict[key] for key in self.class_list])
        
        judge_res_list = (pred_score_array > thre_array).astype('uint8').tolist()
        self.judge_res_list = judge_res_list

    def get_confusion_matrix(self):
        if not hasattr(self, 'judge_res_list'):
            print("@@ Warning: You did not run eval_judgment manually, and we have to run it with default args.")
            self.eval_judgment()
        assert hasattr(self, 'judge_res_list'), f"get_confusion_matrix failed for judge_res_list not found."
        cm_dict = {key:dict(TP=0, TN=0, FN=0, FP=0) for key in self.class_list}

        y_true = np.array(self.gt_data_list).astype('bool').astype('uint8')
        y_pred = np.array(self.judge_res_list).astype('bool').astype('uint8')

        TP = (y_true * y_pred).sum(axis=0)
        TN = ((1 - y_true) * (1 - y_pred)).sum(axis=0)
        FP = ((1 - y_true) * y_pred).sum(axis=0)
        FN = (y_true * (1 - y_pred)).sum(axis=0)

        for index, key in enumerate(self.class_list):
            cm_dict[key]['TP'] = TP[index]
            cm_dict[key]['TN'] = TN[index]
            cm_dict[key]['FP'] = FP[index]
            cm_dict[key]['FN'] = FN[index]

        for key, cm_info in cm_dict.items():
            assert cm_info['TP'] + cm_info['TN'] + cm_info['FP'] + cm_info['FN'] == len(self.judge_res_list), f"cm_info {cm_info} sum error."
        
        # cm_list = multilabel_confusion_matrix(y_true=y_true, y_pred=y_pred)
        cm_pd = pd.DataFrame(cm_dict)
        return cm_pd

    def dump_failure_case(self, data_root, target_dir):
        if not hasattr(self, 'judge_res_list'):
            print("@@ Warning: You did not run eval_judgment manually, and we have to run it with default args.")
            self.eval_judgment()
        assert hasattr(self, 'judge_res_list'), f"dump_failure_case failed for judge_res_list not found."

        dir_check(target_dir)
        target_dir = Path(target_dir)

        for judge_res, gt_data, data_path in tqdm(zip(self.judge_res_list, self.gt_data_list, self.data_path_list), total=len(self.gt_data_list)):
            if not isinstance(data_path, str) and np.iterable(data_path):
                need_puzzle = True
                assert len(data_path) > 0, f"empty path list."
            else:
                need_puzzle = False

            for index, label in enumerate(self.class_list):
                pred_label = judge_res[index]
                gt_label = gt_data[index]
                
                if pred_label == gt_label:
                    continue
                else:
                    if gt_label == 0:
                        mode = 'false_alarm'
                    elif gt_label == 1:
                        mode = 'miss'
                    else:
                        raise ValueError(f"bad gt label value {gt_label}")

                    # save image name
                    if need_puzzle:
                        ori_img_path = Path(data_root) / data_path[0]
                    else:
                        ori_img_path = Path(data_root) / data_path
                    
                    # image load
                    if need_puzzle:
                        # image_list = [ cv_rgb_imread(Path(data_root) / path) for path in data_path]
                        ######################
                        image_list = list()
                        for image_path in data_path:
                            data_path = str(Path(data_root) / image_path).replace('\\', '/')
                            try:
                                image = cv_rgb_imread(data_path)
                            except:
                                try:
                                    image = cv_rgb_imread(data_path.replace('attention/normal', 'attention/dark_field'))
                                except:
                                    image = cv_rgb_imread(data_path.replace('attention/normal', 'attention/large_image'))
                            image_list.append(image)
                        #####################
                        image = puzzle(image_list)
                    else:
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

                    save_path = target_dir / mode / label / encode_path(ori_img_path)
                    cv_rgb_imwrite(image, save_path)

    eval = eval_ap
    evaluate = eval_ap
    evaluation = eval_ap


ClassificationEvaluator = ClassificationMultiLabelEvaluator
MultiLabelEvaluator = ClassificationMultiLabelEvaluator