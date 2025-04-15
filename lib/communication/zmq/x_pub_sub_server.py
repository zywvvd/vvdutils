import zmq
import time
from loguru import logger


class XPubSubServer:
    def __init__(self, x_sub_port, x_pub_port):
        self.x_sub_port = str(x_sub_port)
        self.x_pub_port = str(x_pub_port)
        self.context = None

    def start(self):
        while True:
            try:
                self.context = zmq.Context()

                # 前端接收发布者消息（XSUB）
                frontend = self.context.socket(zmq.XSUB)
                frontend.bind("tcp://*:" + self.x_sub_port)  # 发布者连接到此端口
                logger.info(f"Listening for publishers on tcp://*:{self.x_sub_port}")

                # 后端发送给订阅者（XPUB）
                backend = self.context.socket(zmq.XPUB)
                backend.bind("tcp://*:" + self.x_pub_port)  # 订阅者连接到此端口
                logger.info(f"Listening for subscribers on tcp://*:{self.x_pub_port}")

                # 使用代理进行消息转发
                zmq.proxy(frontend, backend)
                logger.info("Proxy end.")

            except Exception as e:
                logger.exception(f"Error: {e}")


class XSubClient:
    def __init__(self, host, x_pub_port, filter=''):
        self.port = x_pub_port
        self.host = host
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://{host}:{x_pub_port}")
        self.socket.setsockopt(zmq.SUBSCRIBE, filter.encode('utf-8'))    #.encode('utf-8')
        time.sleep(0.001)

    def sub(self, verbose=False):
        while True: 
            logger.info(f" @@ host - {self.host} port - {self.port} Waiting for data...")
            data = self.socket.recv_string(encoding='utf-8')
            if verbose:
                logger.info(data)
            return data


class XPubClient:
    def __init__(self, host, x_sub_port):
        self.port = x_sub_port
        self.host = host
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.connect(f"tcp://{host}:{x_sub_port}")
        time.sleep(0.001)

    def pub(self, data):
        return self.socket.send_string(str(data), encoding='utf-8')


class SubClient:
    def __init__(self, sub_port, host="*", filter=''):
        self.port = sub_port
        self.host = host
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.bind(f"tcp://{host}:{sub_port}")
        self.socket.setsockopt(zmq.SUBSCRIBE, filter.encode('utf-8'))    #.encode('utf-8')
        time.sleep(0.001)

    def sub(self, verbose=False):
        while True: 
            logger.info(f" @@ host - {self.host} port - {self.port} Waiting for data...")
            data = self.socket.recv_string(encoding='utf-8')
            if verbose:
                logger.info(data)
            return data

class PubClient:
    def __init__(self, host, pub_port):
        self.port = pub_port
        self.host = host
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://{host}:{pub_port}")
        time.sleep(0.001)

    def pub(self, data):
        return self.socket.send_string(str(data), encoding='utf-8')

