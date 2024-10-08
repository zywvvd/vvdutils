# we assume that our users have not installed onnxruntime
# so we will recommend them only when they call the function that requires onnxruntime
# and all the functions in this file needs onnxruntime
import numpy as np


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

    def __init__(self, model_path, mean=None, std=None, model_type='float32', gpu=True):
        import onnxruntime

        if mean is not None:
            self.mean = mean
        if std is not None:
            self.std = std

        assert model_type in ['float32', 'float16'], "model_type must be 'float32' or 'float16'"
        self.model_type = model_type

        # 加载模型
        if gpu:
            self.ort_session = onnxruntime.InferenceSession(model_path, providers=[
                        ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT", "cudnn_conv_use_max_workspace": '1'})
                    ])
        else:
            self.ort_session = onnxruntime.InferenceSession(model_path)
        self.input_name = self.ort_session.get_inputs()[0].name


    def infer(self, image):
        # 归一化
        processed_image = (image - self.mean) / self.std

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

        return outputs
    
    __call__ = infer