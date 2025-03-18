import zmq
import time


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
                print(f"Listening for publishers on tcp://*:{self.x_sub_port}")

                # 后端发送给订阅者（XPUB）
                backend = self.context.socket(zmq.XPUB)
                backend.bind("tcp://*:" + self.x_pub_port)  # 订阅者连接到此端口
                print(f"Listening for subscribers on tcp://*:{self.x_pub_port}")

                # 使用代理进行消息转发
                zmq.proxy(frontend, backend)
                print("Proxy end.")

            except Exception as e:
                print(f"Error: {e}")
                time.sleep(2)


class XSubClient:
    def __init__(self, host, x_pub_port, filter=''):
        self.port = x_pub_port
        self.host = host
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://{host}:{x_pub_port}")
        self.socket.setsockopt(zmq.SUBSCRIBE, filter.encode('utf-8'))    #.encode('utf-8')
        time.sleep(0.3)

    def sub(self, verbose=False):
        while True: 
            print(" @@ Waiting for data...")
            data = self.socket.recv_string(encoding='utf-8')
            if verbose:
                print(data)
            return data


class XPubClient:
    def __init__(self, host, x_sub_port):
        self.port = x_sub_port
        self.host = host
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.connect(f"tcp://{host}:{x_sub_port}")
        time.sleep(0.3)

    def pub(self, data):
        return self.socket.send_string(str(data), encoding='utf-8')
