#
# Class for evaluation (classification/detection/segmentation)
#

import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm

from .base import EvaluatorBase

from ..processing import compute_box_box_iou
from ..utils import dir_check
from ..processing import cv_rgb_imread
from ..utils import encode_path
from ..processing import boxes_painter
from ..processing import cv_rgb_imwrite

from ..utils import eval_map


class DetectionEvaluator(EvaluatorBase):
    TYPE = 'Detection'

    def _get_pred_result_dict(self, dm_pred):
        # collect pred data for ap computation
        pred_result_dict = dict()
        for rec in dm_pred:
            # collect detection results
            bboxes = np.reshape(dm_pred.get_detection_xyxy_list_from_record(rec),  [-1,4])
            scores = np.reshape(dm_pred.get_detection_score_list_from_record(rec), [-1,1])
            labels = np.reshape(dm_pred.get_detection_classid_list_from_record(rec), [-1])
            det = np.hstack([bboxes, scores])
            det_res = [det[labels==classid,:] for classid in range(len(self.classnames))]
            pred_result_dict[rec['info']['uuid']] = det_res

        return pred_result_dict

    def _get_gt_result_dict(self, dm_gt):
        # collect gt data for ap computation
        gt_result_dict = dict()
        for rec in dm_gt:
            # collect annotation results
            ann = dict(
                bboxes = np.reshape(dm_gt.get_detection_xyxy_list_from_record(rec), [-1,4]),
                labels = np.reshape(dm_gt.get_detection_classid_list_from_record(rec), [-1])
            )
            gt_result_dict[rec['info']['uuid']] = ann

        return gt_result_dict

    def eval_ap(self, iou_threshold=.1, verbose=True):
        """
        :iou_threshold: the minimal iou that a TP prediction is required when overlapped with a gt bbox
        """
        # compute ap
        _, ap_results = eval_map(
            det_results=self.pred_data_list,
            annotations=self.gt_data_list,
            classnames=self.classnames,
            iou_thr=iou_threshold,
        )

        # print evaluation
        if verbose:
            ap_dict = {classname:x['ap'] for classname, x in zip(self.classnames, ap_results)}
            df = pd.DataFrame.from_dict(ap_dict, orient='index', columns=['AP'])
            df_notnull = df[df['AP'].notnull()]  # remove NaN data (No samples of this class in gt)
            print(df_notnull)

        wmAP = 0
        data_num = 0
        for ap_result in ap_results:
            wmAP += ap_result['num_gts'] * ap_result['ap']
            data_num += ap_result['num_gts']
        if data_num == 0:
            wmAP = 0
        else:
            wmAP /= data_num

        if verbose:
            print(f" wmAP: {wmAP}")

        eval_result = {
            'AP': ap_results,
            'wmAP': wmAP
        }

        self.ap_result = eval_result
        return eval_result

    def _set_threshold(self, policy='inflection', value=None):
        """
        Args:
            policy (str, optional): could be one of 'inflection', 'recall', 'precision' or 'manual'. Defaults to 'inflection'.
            value (optional): 
                under 'inflection' policy: value is insignificant
                under 'recall' or 'precision': value should be the minimum float we can accept
                under 'manual': value can be a list or dict who has the same length or keys with classnames. 
                . Defaults to None.
        """
        assert policy in ['inflection', 'recall', 'precision', 'manual'], f"policy {policy} should in ['inflection', 'recall', 'precision', 'manual']"
        threshold_dict = dict()

        if policy != 'manual':
            if not hasattr(self, 'ap_result'):
                print("@@ Warning: You did not run eval_ap manually, and we have to run it with default args.")
                self.eval_ap(verbose=False)
            assert hasattr(self, 'ap_result'), f"inflection threshold need self.ap_result, and we can not find it."

        if policy == 'inflection':
            ap_res = self.ap_result['AP']
            for key, ap_data in ap_res.items():
                recall = ap_data['recall']
                precision = ap_data['precision']
                score = ap_data['score']
                area = np.array(recall) * np.array(precision)
                if len(area) > 0 and area.max() > 0:
                    threshold_dict[key] = score[np.argmax(area)]
                else:
                    threshold_dict[key] = 1

        elif policy == 'recall':
            assert 0 <= value <= 1, f"bad value {value} for minimum recall"
            ap_res = self.ap_result['AP']
            for key, ap_data in ap_res.items():
                recall = ap_data['recall']
                mark = recall >= value
                score = ap_data['score']

                if mark.max() > 0:
                    threshold_dict[key] = score[np.argmax(mark)]
                else:
                    threshold_dict[key] = score[-1]

        elif policy == 'precision':
            assert 0 <= value <= 1, f"bad value {value} for minimum precision"
            ap_res = self.ap_result['AP']
            for key, ap_data in ap_res.items():
                precision = ap_data['precision']
                mark = precision >= value
                score = ap_data['score']

                if mark.max() > 0:
                    threshold_dict[key] = score[len(mark) - np.argmax(mark[::-1]) - 1]
                else:
                    threshold_dict[key] = score[0]
        elif policy == 'manual':
            if isinstance(value, dict):
                threshold_dict = value
            else:
                assert isinstance(value, list), f"under manual threshold policy, value should be dict or list. {value}"
                assert len(value) == len(self.classnames), f"value list is not as long as classnames {len(value)} != {len(self.classnames)}"
                for index in range(len(self.classnames)):
                    threshold_dict[self.classnames[index]] = value[index]

        else:
            raise RuntimeError(f"unknown policy {policy}")

        return threshold_dict

    def eval_judgment(self, iou_threshold=.1, verbose=True):
        """
        :iou_threshold: the minimal iou that a TP prediction is required when overlapped with a gt bbox
        """
        if not hasattr(self, 'threshold_dict'):
            print("@@ Warning: You did not run set_threshold manually, and we have to run it with default args.")
            self.set_threshold()
        assert hasattr(self, 'threshold_dict'), f"eval_judgment failed for threshold_dict not found."

        judge_res_list = list()
        for gt_data, pred_data, data_path in zip(self.gt_data_list, self.pred_data_list, self.data_path_list):
            judge_result = self.data_judge(gt_data, pred_data, self.threshold_dict, iou_threshold)
            judge_res_list.append([judge_result, data_path])
        self.judge_res_list = judge_res_list

    @staticmethod
    def data_judge(gt_data, pred_data, threshold_dict, iou_threshold):
        judge_result = [[] for _ in range(len(threshold_dict))]
        gt_bbox_dict = dict()
        for index, label in enumerate(gt_data['labels'].tolist()):
            if label not in gt_bbox_dict:
                gt_bbox_dict[label] = list()
            bbox_info = {
                'type': 'gt',
                'bbox': gt_data['bboxes'][index].tolist(),
                'hit': False
            }
            gt_bbox_dict[label].append(bbox_info)
            judge_result[label].append(bbox_info)
        for index, pred_boxes in enumerate(pred_data):
            for bbox_score in pred_boxes:
                bbox = bbox_score[:4].tolist()
                score = bbox_score[-1]

                if score < threshold_dict[index]:
                    continue
                else:
                    hit = False
                    if index in gt_bbox_dict:
                        for gt_bbox_info in gt_bbox_dict[index]:
                            gt_bbox = gt_bbox_info['bbox']
                            iou = compute_box_box_iou(gt_bbox, bbox)
                            if iou >= iou_threshold:
                                hit = True
                                gt_bbox_info['hit'] = True

                    bbox_info = {
                            'score': score,
                            'type': 'pred',
                            'bbox': bbox,
                            'hit': hit
                        }
                    judge_result[index].append(bbox_info)
        return judge_result

    def get_confusion_matrix(self):
        if not hasattr(self, 'judge_res_list'):
            print("@@ Warning: You did not run eval_judgment manually, and we have to run it with default args.")
            self.eval_judgment()
        assert hasattr(self, 'judge_res_list'), f"get_confusion_matrix failed for judge_res_list not found."
        cm_dict = {key:dict(TP=0, TN=0, FN=0, FP=0) for key in self.classnames}

        # y_true = np.zeros([len(self.judge_res_list), len(self.classnames)])
        # y_pred = np.zeros([len(self.judge_res_list), len(self.classnames)])

        for data_index, judge_res in enumerate(self.judge_res_list):
            for index, class_bboxes in enumerate(judge_res[0]):
                pred_label = False
                gt_label = False
                for bbox_info in class_bboxes:
                    if bbox_info['type'] == 'gt':
                        gt_label = True
                    elif bbox_info['type'] == 'pred':
                        pred_label = True
                    else:
                        raise ValueError(f"unknown type value {bbox_info['type']}")
                # if gt_label:
                #     y_true[data_index][index] = 1
                # if pred_label:
                #     y_pred[data_index][index] = 1
                
                if pred_label == gt_label == False:
                    cm_dict[self.classnames[index]]['TN'] += 1
                elif pred_label == gt_label == True:
                    cm_dict[self.classnames[index]]['TP'] += 1
                elif pred_label == True and gt_label == False:
                    cm_dict[self.classnames[index]]['FP'] += 1
                elif pred_label == False and gt_label == True:
                    cm_dict[self.classnames[index]]['FN'] += 1
                else:
                    raise ValueError(f"unknown pred_label {pred_label} and gt_label {gt_label} value.")

        for key, cm_info in cm_dict.items():
            assert cm_info['TP'] + cm_info['TN'] + cm_info['FP'] + cm_info['FN'] == len(self.judge_res_list), f"cm_info {cm_info} sum error."
        
        # cm_list = multilabel_confusion_matrix(y_true=y_true, y_pred=y_pred)
        
        cm_pd = pd.DataFrame(cm_dict)
        
        return cm_pd

    def dump_failure_case(self, data_root, target_dir):
        if not hasattr(self, 'judge_res_list'):
            print("@@ Warning: You did not run eval_judgment manually, and we have to run it with default args.")
            self.eval_judgment()
        assert hasattr(self, 'judge_res_list'), f"get_confusion_matrix failed for judge_res_list not found."

        dir_check(target_dir)
        target_dir = Path(target_dir)

        for judge_res in tqdm(self.judge_res_list):
            mode_list = list()

            box_list = list()
            label_list = list()
            score_list = list()
            color_list = list()

            draw = False

            for index, class_bboxes in enumerate(judge_res[0]):
                data_mode = set()
                label = self.classnames[index]
                for bbox_info in class_bboxes:
                    box = bbox_info['bbox']
                    if bbox_info['type'] == 'gt':
                        score = 1
                        if bbox_info['hit'] == False:
                            draw = True
                            data_mode.add('miss')
                            color = [255, 0, 0]
                        else:
                            color = [0, 255, 0]
                    elif bbox_info['type'] == 'pred':
                        score = bbox_info['score']
                        if bbox_info['hit'] == False:
                            draw = True
                            data_mode.add('false_alarm')
                            color = [255, 255, 0]
                        else:
                            color = [0, 0, 255]
                    else:
                        raise ValueError(f"unknown type value {bbox_info['type']}")
                    box_list.append(box)
                    label_list.append(label)
                    score_list.append(score)
                    color_list.append(color)

                mode_list.append(data_mode)
            if draw:
                ori_img_path = Path(data_root) / judge_res[1]
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
                painted_image = boxes_painter(image, box_list=box_list, label_list=label_list, score_list=score_list, color_list=color_list)

            for index, data_mode in enumerate(mode_list):
                label = self.classnames[index]
                for mode in data_mode:
                    save_path = target_dir / mode / label / encode_path(ori_img_path)
                    cv_rgb_imwrite(painted_image, save_path)

    eval = eval_ap
    evaluate = eval_ap
    evaluation = eval_ap
