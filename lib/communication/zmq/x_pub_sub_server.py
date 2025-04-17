import zmq
import time
from loguru import logger
import socket


class XPubSubServer:
    def __init__(self, x_sub_port, x_pub_port, host="*"):
        self.x_sub_port = str(x_sub_port)
        self.x_pub_port = str(x_pub_port)
        self.host = host
        self.context = None

    def start(self):
        while True:
            try:
                self.context = zmq.Context()

                # 前端接收发布者消息（XSUB）
                frontend = self.context.socket(zmq.XSUB)
                frontend.bind("tcp://" + self.host + ":" + self.x_sub_port)  # 发布者连接到此端口
                logger.info(f"Listening for publishers on tcp://*:{self.x_sub_port}")

                # 后端发送给订阅者（XPUB）
                backend = self.context.socket(zmq.XPUB)
                backend.bind("tcp://" + self.host + ":" + self.x_pub_port)  # 订阅者连接到此端口
                logger.info(f"Listening for subscribers on tcp://*:{self.x_pub_port}")

                # 使用代理进行消息转发
                zmq.proxy(frontend, backend)
                logger.info("Proxy end.")

            except Exception as e:
                logger.exception(f"Error: {e}")

def check_port(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex((host, int(port))) == 0  # ‌:ml-citation{ref="2" data="citationList"}


class XSubClient:
    def __init__(self, host, x_pub_port, filter='', reconnect_num = 100):
        self.port = x_pub_port
        self.host = host
        self.context = zmq.Context()
        self.reconnect_num = reconnect_num
        self.index = 0
        self.init_socket(host, x_pub_port, filter)

    def init_socket(self, host, x_pub_port, filter):
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.RECONNECT_IVL, 1000) # 设置重连间隔为 1000 毫秒
        self.socket.setsockopt(zmq.RECONNECT_IVL_MAX, 10000) # 设置最大重连间隔为 10000 毫秒
        self.socket.setsockopt(zmq.TCP_KEEPALIVE, 1) # 启用 TCP keepalive
        self.socket.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 40) # 设置 keepalive 消息间隔为 40 秒
        self.socket.connect(f"tcp://{host}:{x_pub_port}")

        self.socket.setsockopt(zmq.SUBSCRIBE, filter.encode('utf-8'))    #.encode('utf-8')
        time.sleep(0.001)

    def sub(self, verbose=False, timeout=10):
        if timeout is not None:
            timeout = int(timeout) * 1000
        self.socket.setsockopt(zmq.RCVTIMEO, timeout) if timeout is not None else None
        while True: 
            try:
                logger.info(f" @@ host - {self.host} port - {self.port} Waiting for data...")
                data = self.socket.recv_string(encoding='utf-8')
                if verbose:
                    logger.info(data)
                return data
            except zmq.Again:
                logger.debug("Timeout occurred")
            except Exception as e:
                logger.error(f"An error occurred: {e}")
            finally:
                self.index += 1
                if self.index % self.reconnect_num == 0:
                    self.reconnect()

    def close(self):
        self.socket.close()
        self.context.term()
        logger.info(f"Closed socket on host {self.host} port {self.port}")

    def reconnect(self):
        self.close()
        self.init_socket(self.host, self.port, self.filter)
        logger.info(f"Reconnected to host {self.host} port {self.port}")


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

