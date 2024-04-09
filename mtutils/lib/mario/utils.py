import json

from flask import request
from flask import jsonify
import requests
import numpy as np
from datetime import datetime
import cv2
from .. import MyEncoder


def tcp_encoder(image_list, **kwargs):
    headers = {}
    headers['time-before-format-encoding'] = str(datetime.now())

    # record image shapes and encode image list to a string
    image_shape_str = ''
    arr = list()
    for img in image_list:
        image_shape_str += str(img.shape)+';'
        arr.append(img.flatten())
    headers['image_shapes'] = image_shape_str.rstrip(';')        

    encoded_data = dict(
        data = np.hstack(arr).tostring(),
        headers = headers
    )
    headers['kwargs'] = json.dumps(kwargs, cls=MyEncoder)

    headers['time-before-sending'] = str(datetime.now())
    return encoded_data


def tcp_decoder(request):
    t_arrive = datetime.now()
    t_format_encode = datetime.fromisoformat(request.headers['time-before-format-encoding'])
    t_send   = datetime.fromisoformat(request.headers['time-before-sending'])

    image_str = request.data
    image_shape_str = request.headers['image_shapes']

    # decode image shape and recover image list
    img_data = np.fromstring(image_str, dtype='uint8')
    image_list = list()
    data_index = 0
    exception_info = list()
    for shape_str in image_shape_str.split(';'):
        try:
            image_shape = tuple(int(x) for x in shape_str[1:-1].split(','))
            image_size  = np.prod(image_shape)
            img = img_data[data_index:data_index+image_size].reshape(image_shape)
            data_index += image_size
            image_list.append(img)
        except Exception as e:
            print(e)
            exception_info.append("Fail to decode image from HTTP request")
        else:
            exception_info.append(None)

    assert len(img_data) == data_index

    # deformat kwargs in headers
    kwargs = json.loads(request.headers['kwargs'])

    t_format_decode = datetime.now()

    log = dict(
        format_encoding = (t_send-t_format_encode).total_seconds(),
        sending = (t_arrive-t_send).total_seconds(),
        format_decoding = (t_format_decode-t_arrive).total_seconds(),
        communication = (t_format_decode-t_format_encode).total_seconds(),
        before_sending = t_format_encode,
        exception_info = exception_info
    )

    return image_list, kwargs, log


def client_detection_infer(url, image, **kwargs):
    """ Infer single image """
    encoded_data = tcp_encoder([image], **kwargs)
    resp = requests.post(
        url='http://127.0.0.1:8000' + url,
        **encoded_data
    )
    results = resp.json()
    return results


def server_detection_infer(detection_func):
    """ Infer single image """
    if request.method == 'POST':
        image_list, kwargs, log = tcp_decoder(request)
        image = image_list[0]

        t_before_infer = datetime.now()
        results = detection_func(image, **kwargs)
        t_after_infer = datetime.now()

        print("\nSubmodule data exchanging costs:")
        print("----------------------------------")
        print("Format encoding: {}s".format(log['format_encoding']))
        print("Sending: {}s".format(log['sending']))
        print("Format decoding: {}s".format(log['format_decoding']))
        print("Communication cost: {}s".format(log['communication']))
        print("Inference: {}s".format((t_after_infer-t_before_infer).total_seconds()))
        print("Total cost: {}s".format((t_after_infer-log['before_sending']).total_seconds()))
        print("==================================\n")

        return jsonify(results)
