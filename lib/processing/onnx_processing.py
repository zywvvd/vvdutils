# we assume that our users have not installed onnxruntime
# so we will recommend them only when they call the function that requires onnxruntime
# and all the functions in this file needs onnxruntime
import numpy as np
import cv2
from skimage.transform import resize
from ..utils import get_gpu_str_as_you_wish

def get_gpu_device_count(verbose=0):
    try:
        import pynvml
    except Exception as e:
        print('can not import pynvml.', e)
        print('please make sure pynvml is installed correctly.')
        print('a simple pip install nvidia-ml-py3 may help.')
        print('now a 0 will be return')
        return 0

    try:
        # 初始化工具
        pynvml.nvmlInit()
    except Exception as e:
        print('pynvml.nvmlInit failed:', e)
        print('now a 0 will be return')
        return 0
    # 驱动信息
    if verbose:
        print("GPU driver version: ", pynvml.nvmlSystemGetDriverVersion())
    # 获取Nvidia GPU块数
    gpuDeviceCount = pynvml.nvmlDeviceGetCount()
    return gpuDeviceCount

class OnnxSimpleRelease:
    mean=[
            123.675,
            116.28,
            103.53,
        ]

    std=[
        58.395,
        57.12,
        57.375,
    ]

    def __init__(self, model_path, specific_wh=None, mean=None, std=None, model_type='float32', gpu=True, device_id=None, bgr=False):
        import onnxruntime
        try:
            import pynvml
        except Exception as e:
            print('can not import pynvml.', e)
            print('please make sure pynvml is installed correctly.')
            print('a simple pip install nvidia-ml-py3 may help.')
            print('now a 0 will be return')
            raise RuntimeError("Onnx releaser need pynvml --- pip install nvidia-ml-py3")

        self.gpuDeviceCount = get_gpu_device_count()
        if self.gpuDeviceCount < 1:
            gpu = False

        if mean is not None:
            self.mean = mean

        if std is not None:
            self.std = std

        if specific_wh is not None:
            self.target_size = True
            self.width = int(specific_wh[0])
            self.height = int(specific_wh[1])
        else:
            self.target_size = False

        assert model_type in ['float32', 'float16'], "model_type must be 'float32' or 'float16'"
        self.model_type = model_type
        self.device_id = device_id
        self.bgr = bgr

        # 加载模型
        if gpu:
            CUDAExecutionProvider = {
                "cudnn_conv_algo_search": "DEFAULT",
                "cudnn_conv_use_max_workspace": '1'
            }
            if device_id is not None:
                CUDAExecutionProvider.update({'device_id': str(device_id)})
            else:
                gpu_index_str, gpu_index_picked_list = get_gpu_str_as_you_wish(1)
                CUDAExecutionProvider.update({'device_id': gpu_index_str})

            print(f"gpu ONNX runtime init: {CUDAExecutionProvider}.")
            self.ort_session = onnxruntime.InferenceSession(model_path, providers=[
                        ("CUDAExecutionProvider", CUDAExecutionProvider)])
        else:
            print(f"cpu ONNX runtime init.")
            self.ort_session = onnxruntime.InferenceSession(model_path)
        self.input_name = self.ort_session.get_inputs()[0].name


    def infer(self, rgb_image):
        if self.bgr:
            image = rgb_image[..., ::-1]
        else:
            image = rgb_image
        # 归一化
        processed_image = (image - self.mean) / self.std

        image_h, image_w = image.shape[:2]
        need_resize = False
        if self.target_size and (image_h != self.height or image_w != self.width):
            # 调整尺寸
            processed_image = cv2.resize(processed_image, (self.width, self.height))
            need_resize = True

        # 调整维度
        input_data = np.expand_dims(processed_image, axis=0).transpose(0, 3, 1, 2)
        if self.model_type == 'float16':
            input_data = input_data.astype(np.float16)
        elif self.model_type == 'float32':
            input_data = input_data.astype(np.float32)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # 运行模型
        outputs = self.ort_session.run(None, {self.input_name: input_data})

        if need_resize:
            new_outputs = list()
            for i in range(len(outputs)):
                resize_shape = [1, 1, image_h, image_w]
                new_outputs.append(resize(outputs[i], resize_shape, mode='constant', cval=0.0, order=0))
            outputs = new_outputs

        return outputs
    
    __call__ = infer