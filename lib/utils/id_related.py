import uuid
import hashlib
import time
import os


def create_uuid():
    """ create a uuid (universally unique ID) """
    md5_hash = hashlib.md5(uuid.uuid1().bytes)
    return md5_hash.hexdigest()


def get_hash_code(file):
    """ get the md5 hash code of a given file """
    assert os.path.exists(file)
    md5_hash = hashlib.md5()
    with open(file, "rb") as fid:
        md5_hash.update(fid.read())
        digest = md5_hash.hexdigest()
    return digest


def is_black_distribution(distribution):
    """ Check if the distribution is implying a black instance (used in data.update()) """
    return all([x < 0 for x in distribution])


class Snowflake:  
    '''
    生成雪花算法ID

    worker_id = 1  
    data_center_id = 1  
    snowflake = Snowflake(worker_id, data_center_id)  
    snowflake.next_id()
    '''

    def __init__(self, worker_id, data_center_id):  
        ### 机器标识ID
        self.worker_id = worker_id  
        ### 数据中心ID
        self.data_center_id = data_center_id  
        ### 计数序列号
        self.sequence = 0  
        ### 时间戳
        self.last_timestamp = -1  
  
    def next_id(self):  
        timestamp = int(time.time() * 1000)  
        if timestamp < self.last_timestamp:  
            raise Exception("Clock moved backwards. Refusing to generate id for %d milliseconds" % abs(timestamp - self.last_timestamp))  
        if timestamp == self.last_timestamp:  
            self.sequence = (self.sequence + 1) & 4095  
            if self.sequence == 0:  
                timestamp = self.wait_for_next_millis(self.last_timestamp)  
        else:  
            self.sequence = 0  
        self.last_timestamp = timestamp  
        return ((timestamp - 1288834974657) << 22) | (self.data_center_id << 17) | (self.worker_id << 12) | self.sequence  
  
    def wait_for_next_millis(self, last_timestamp):  
        timestamp = int(time.time() * 1000)  
        while timestamp <= last_timestamp:  
            timestamp = int(time.time() * 1000)  
        return timestamp
