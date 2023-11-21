import sys
sys.path.append('.')

import pytest
from lib.data import DataManager
from lib.utils import decode_distribution

data_det = DataManager.load('assets/det.json')
data_mc  = DataManager.load('assets/mc.json')
data_ml  = DataManager.load('assets/ml.json')

record_det1 = {
    "info": {
            "uuid": "59fe019ce0722fad5ae2b6fcba9b68a9",
            "image_path": "MP2112712A_HJB169-04-F3_UBM_9_1_1.jpg"
        },
        "instances": [
            {
                "uuid": "73db27eb3785d6523b6dfa32b1dcf693",
                "shape_type": "polygon",
                "points": "298+344,298+406,368+406,368+344,298+406.",
                "score": 0.06946061,
                "classname": "BD"
            },
            {
                "uuid": "73db27eb3785d6523b6dfa32b1dcf693",
                "shape_type": "polygon",
                "points": "298+644,298+406,368+406,368+644,298+406.",
                "score": 0.7825,
                "classname": "FM"
            }
        ]
}
record_det2 = {
    "info": {
            "uuid": "59fe019ce0722fad5ae2b6fcba9b68a9",
            "image_path": "MP2112712A_HJB169-04-F3_UBM_9_1_1.jpg"
        },
        "instances": []
}
record_mc = {
    "info": {
        "uuid": "bff6e80a1b0ad51f60ab6159a64d411f",
        "image_path": "X26477-01B4_X_74_Y_75_215_1.jpg"
    },
    "distribution": "1.733311e-01,0.0,0.0,0.0,0.0,0.0,0.0,0.0,8.266689e-01,0.0,0.0,0.0,0.0,0.0"
}
record_ml = {
    "info": {
        "uuid": "bff6e80a1b0ad51f60ab6159a64d411f",
        "image_path": "X26477-01B4_X_74_Y_75_215_1.jpg"
    },
    "scores": "0.0,0.0,0.0,4.889707e-02,4.368331e-02,9.424161e-02,0.0,8.266689e-01,6.326669e-02,0.0,0.0,0.0,0.0"
}



@pytest.mark.parametrize('data', [data_det, data_mc, data_ml])
def test_dataset_format(data):
    class_list = data.class_list
    # check class list
    assert isinstance(class_list, list)
    assert all([isinstance(classname, str) for classname in class_list])

    # check record
    record_list = data.record_list
    assert isinstance(record_list, list)

    ## check record
    task_type = data.task_type
    assert task_type in ['multiclass', 'multilabel', 'detection']

    for record in data:
        assert 'info' in record
        assert 'uuid' in record['info']
        assert 'image_path' in record['info']

        if task_type == 'multiclass':
            'distribution' in record
        elif task_type == 'multilabel':
            'scores' in record
        else:
            'instances' in record
            # check instances
            for inst in record['instances']:
                assert 'uuid' in inst
                assert 'shape_type' in inst
                assert 'points' in inst
                assert 'score' in inst
                assert 'classname' in inst


def test_detection_info_extraction():
    data = data_det
    instance = record_det1['instances'][0]

    ## classid
    classid = data.get_detection_classid_from_instance(instance)
    assert classid == 2
    classid_list = data.get_detection_classid_list_from_record(record_det1)
    assert len(classid_list) == 2 and classid_list[0] == 2 and classid_list[1] == 5
    classid_list = data.get_detection_classid_list_from_record(record_det2)
    assert len(classid_list) == 0

    ## classname
    classname = data.get_detection_classname_from_instance(instance)
    assert classname == 'BD'
    classname_list = data.get_detection_classname_list_from_record(record_det1)
    assert len(classname_list) == 2 and classname_list[0] == 'BD' and classname_list[1] == 'FM'
    classname_list = data.get_detection_classname_list_from_record(record_det2)
    assert len(classname_list) == 0

    ## score
    score = data.get_detection_score_from_instance(instance)
    assert score == .06946061
    scores = data.get_detection_score_list_from_record(record_det1)
    assert len(scores) == 2 and scores[0] == .06946061 and scores[1] == .7825
    scores = data.get_detection_score_list_from_record(record_det2)
    assert len(scores) == 0

    ## shape
    points = data.get_detection_shape_from_instance(instance)
    assert len(points) == 5 and points[0][1] == 344
    pt_list = data.get_detection_shape_list_from_record(record_det1)
    assert len(pt_list) == 2 and len(pt_list[0]) == 5 and pt_list[0][0][1] == 344
    xyxy = data.get_detection_xyxy_from_instance(instance)
    assert tuple(xyxy) == (298.0, 344.0, 368.0, 406.0)
    xyxys = data.get_detection_xyxy_list_from_record(record_det1)
    assert len(xyxys) == 2 and tuple(xyxys[1]) == (298.0, 406.0, 368.0, 644.0)
    xywh = data.get_detection_xywh_from_instance(instance)
    assert tuple(xywh) == (298.0, 344.0, 70.0, 62.0)
    xywhs = data.get_detection_xywh_list_from_record(record_det1)
    assert len(xywhs) == 2 and tuple(xywhs[1]) == (298.0, 406.0, 70.0, 238.0)


def test_multiclass_info_extraction():
    data = data_mc
    record = record_mc
    distribution = decode_distribution(record['distribution'])

    ## classid
    classid = data.get_multiclass_classid_from_distribution(distribution)
    assert classid == 8
    classid = data.get_multiclass_classid_from_record(record)
    assert classid == 8

    ## classname
    classname = data.get_multiclass_classname_from_distribution(distribution)
    assert classname == 'BS_White'
    classname = data.get_multiclass_classname_from_record(record)
    assert classname == 'BS_White'

    ## score
    distribution = data.get_multiclass_distribution_from_record(record)
    assert tuple(distribution) == (0.1733311, 0, 0, 0, 0, 0, 0, 0, 0.8266689, 0, 0, 0, 0, 0)
    score = data.get_multiclass_score_from_distribution(distribution)
    assert score == .8266689
    score = data.get_multiclass_score_from_record(record)
    assert score == .8266689



def test_multiclass_info_extraction():
    data = data_ml
    record = record_ml

    ## scores
    scores = data.get_multilabel_scores_from_record(record)
    assert tuple(scores) == (0, 0, 0, 0.04889707, 0.04368331, 0.09424161, 0, 0.8266689, 0.06326669, 0, 0, 0, 0)
