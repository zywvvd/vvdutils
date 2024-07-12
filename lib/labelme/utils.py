from pathlib import Path
from ..utils import create_uuid
from ..processing import cv_rgb_imread
from ..utils import glob_images
from tqdm import tqdm
from .labelme import Labelme


def make_ok_labelme_obj(image_path):
    if Path(image_path).exists():
        json_path = Path(image_path).with_suffix('.json')
        if not json_path.exists():
            uuid = create_uuid()
            realative_image_path = Path(image_path).name
            image = cv_rgb_imread(image_path)
            H, W = image.shape[:2]
            imaging = 2

            info = {
                'uuid': uuid,
                'image_path': realative_image_path,
                'height': H, 'width': W,
                'roi': [0, 0, H, W],  # roi in parent_image: xyxy
                'parent_uuid': uuid,
                "imaging": imaging,  # 1,2,3,4
            }

            labelme_info = Labelme(info, [])
        else:
            labelme_info = Labelme.from_json(str(json_path))
            
        return labelme_info
    else:
        print(f'Warning !! {image_path} not exists')
        return ''


def make_ok_labelme_json(image_path):
    if Path(image_path).exists():
        json_path = Path(image_path).with_suffix('.json')
        if not json_path.exists():
            labelme_info = make_ok_labelme_obj(image_path)
            labelme_info.save_json(str(json_path))
        return json_path
    else:
        print(f'Warning !! {image_path} not exists')
        return ''


def make_ok_labelme_for_dir(root_dir):
    image_path_list = glob_images(root_dir)
    for image_path in tqdm(image_path_list):
        make_ok_labelme_json(image_path)
