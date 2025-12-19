from ...loader import try_to_import
try_to_import('zmq', "please install pyzmq by 'pip install pyzmq'. ")

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
    def __init__(self, host, x_pub_port, filter='', reconnect_num=30, timeout=20):
        self.port = x_pub_port
        self.host = host
        self.filter = filter
        if timeout is not None:
            self.timeout = int(timeout) * 1000
        else:
            self.timeout = None
        self.reconnect_num = reconnect_num
        self.init_socket(host, x_pub_port, filter)

    def init_socket(self, host, x_pub_port, filter):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.RECONNECT_IVL, 1000) # 设置重连间隔为 1000 毫秒
        self.socket.setsockopt(zmq.RECONNECT_IVL_MAX, 10000) # 设置最大重连间隔为 10000 毫秒
        self.socket.setsockopt(zmq.TCP_KEEPALIVE, 1) # 启用 TCP keepalive
        self.socket.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 40) # 设置 keepalive 消息间隔为 40 秒
        self.socket.connect(f"tcp://{host}:{x_pub_port}")
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout) if self.timeout is not None else None
        self.socket.setsockopt(zmq.SUBSCRIBE, filter.encode('utf-8'))    #.encode('utf-8')
        time.sleep(0.001)

    def sub(self, verbose=False):
        reconnect_index = 0
        while True: 
            try:
                logger.info(f" @@ host - {self.host} port - {self.port} Waiting for data...")
                data = self.socket.recv_string(encoding='utf-8')
                if verbose:
                    logger.info(data)
                reconnect_index = 0
                return data
            except zmq.Again:
                logger.debug("Timeout occurred")
                reconnect_index += 1
            except Exception as e:
                logger.error(f"An error occurred: {e}")
            finally:
                if reconnect_index >= self.reconnect_num:
                    self.reconnect()
                    reconnect_index = 0
                time.sleep(0.001)

    def close(self):
        self.socket.close()
        self.context.term()
        logger.info(f"Closed socket on host {self.host} port {self.port}")

    def reconnect(self):
        self.close()
        self.init_socket(self.host, self.port, self.filter)
        logger.info(f"Reconnected to host {self.host} port {self.port}")


# class XPubClient:
#     def __init__(self, host, x_sub_port):
#         self.port = x_sub_port
#         self.host = host
#         self.context = zmq.Context()
#         self.socket = self.context.socket(zmq.PUB)
#         self.socket.connect(f"tcp://{host}:{x_sub_port}")
#         time.sleep(0.001)

#     def pub(self, data):
#         return self.socket.send_string(str(data), encoding='utf-8')

class XPubClient:
    def __init__(self, host, x_sub_port, reconnect_num=30, timeout=20):
        self.host = host
        self.port = x_sub_port
        self.reconnect_num = reconnect_num
        self.timeout = timeout * 1000  # 转换为毫秒
        self._init_socket()
        time.sleep(0.001)
        logger.info(f"XPUB client initialized for tcp://{host}:{x_sub_port}")

    def _init_socket(self):
        """ 初始化带有重连保护的Socket """
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        
        # 配置Socket参数
        self.socket.setsockopt(zmq.RECONNECT_IVL, 1000)          # 基础重连间隔 1秒
        self.socket.setsockopt(zmq.RECONNECT_IVL_MAX, 10000)     # 最大重连间隔 10秒
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout)  # 发送超时
        self.socket.setsockopt(zmq.TCP_KEEPALIVE, 1)             # 启用Keepalive
        self.socket.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 60)       # 空闲检测间隔
        
        try:
            self.socket.connect(f"tcp://{self.host}:{self.port}")
        except zmq.ZMQError as e:
            logger.error(f"Connection failed: {e}")
            self._safe_reconnect()
        time.sleep(1)  # 等待连接稳定

    def pub(self, data):
        """ 带重试机制的消息发送方法 """
        reconnect_index = 0
        while reconnect_index < self.reconnect_num:
            try:
                # 结构化数据发送 (避免字符串编码问题)
                if not isinstance(data, bytes):
                    data = str(data).encode('utf-8')
                return self.socket.send(data, copy=False)  # 零拷贝优化
                
            except zmq.ZMQError as e:
                logger.warning(f"Send failed (retry {reconnect_index+1}/{self.reconnect_num}): {e}")
                reconnect_index += 1
                self._safe_reconnect()
                time.sleep(1)

        logger.error(f"Publish failed after {self.reconnect_num} retries")

    def _safe_reconnect(self):
        """ 安全重连流程 """
        try:
            self.socket.close()
            self.context.term()
        except Exception as e:
            logger.debug(f"Cleanup error: {e}")
        finally:
            self._init_socket()
            self.reconnect_index = 0

    def close(self):
        """ 资源释放方法 """
        self.socket.setsockopt(zmq.LINGER, 0)  # 立即关闭
        self.socket.close()
        self.context.term()
        logger.info(f"PUB client closed for tcp://{self.host}:{self.port}")



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

