import os
import io

from os.path import exists

import pickle
import time
import json
from loguru import logger
import random 

from bson import ObjectId

from ...loader import lazy_import
from ...utils import dir_check
from ...utils import get_suffix
from ...utils import get_file_hash_code
from ...utils import get_dir_hash_code
from ...utils import md5
from ...utils import OS_join
from ...utils import create_uuid
from ...utils import zip_dir
from ...utils import unzip_dir
from ...utils import popular_image_extensions
from ...processing import read_mongodb_image


def make_mongo_connect_url(username, password, host, port, database):
    return f"mongodb://{username}:{password}@{host}:{port}/{database}"


class MongoGridFSConnection:
    MongoClient = None
    GridFS = None

    DATA_TYPES = ['file', 'dir', 'bytes', 'object', 'none']
    TEMP_DIR = './tmp/mongodb_tmp'

    def __init__(self, username, password, host, port, database, temp_dir=None,
                 connect_timeout=10, server_selection_timeout=20, max_retries=4):
        # from pymongo import MongoClient
        # from gridfs import GridFS
        if self.MongoClient is None:
            type(self).MongoClient = lazy_import('pymongo').MongoClient

        if self.GridFS is None:
            type(self).GridFS = lazy_import('gridfs').GridFS

        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password

        self.connect_timeout = connect_timeout  # 连接超时（秒）
        self.server_selection_timeout = server_selection_timeout  # 服务器选择超时（秒）
        self.max_retries = max_retries  # 最大重试次数
        self.conn = self.connect_with_retry()

        self.db = self.conn[self.database]
        self.fs = self.GridFS(self.db)

        if temp_dir is None:
            self.temp_dir = self.TEMP_DIR
        else:
            self.temp_dir = temp_dir

        assert dir_check(self.temp_dir), f"Temp directory {self.temp_dir} create failed."

    def parse(self, mongo_obj, save_dir='mongo_data', force_save=False):
        obj_type_code = mongo_obj._file['@type_code']
        obj_type = mongo_obj._file['@type']
        name = mongo_obj._file['@name']

        if obj_type_code == 0:
        # file 
            if obj_type in popular_image_extensions and not force_save:
                # image
                return read_mongodb_image(mongo_obj, obj_type)
            elif obj_type == 'json' and not force_save:
                # json
                return json.load(io.BytesIO(mongo_obj.read()))
            else:
                # other file
                dir_check(save_dir)
                save_file_path = OS_join(save_dir, name)
                with open(save_file_path, 'wb') as f:
                    f.write(mongo_obj.read())
                return save_file_path

        elif obj_type_code == 1:
        # dir
            dir_check(save_dir)
            save_zip_path = OS_join(self.temp_dir, name + '.zip')
            with open(save_zip_path, 'wb') as f:
                f.write(mongo_obj.read())

            unzip_dir(save_zip_path, save_dir)
            os.remove(save_zip_path)

            return OS_join(save_dir, name)

        elif obj_type_code == 2:
        # bytes
            return mongo_obj.read()

        elif obj_type_code == 3:
        # dict
            return mongo_obj._file

        elif obj_type_code == 4:
        # pickle object
            return pickle.loads(mongo_obj.read())

        raise NotImplementedError("Under Devolopment.")

    def connect_with_retry(self):
        """带重试机制的连接方法"""
        attempt = 0
        while attempt < self.max_retries:
            try:
                return self.connect()

            except Exception as e:
                attempt += 1
                wait_time = min(2 ** attempt, 10)  # 指数退避，最大10秒
                shake_wait_time = random.uniform(0, 0.5 * wait_time)

                logger.warning(f"Connection attempt {attempt}/{self.max_retries} failed. Retrying in {wait_time}s + {shake_wait_time}s. Error: {str(e)}")
                time.sleep(wait_time + shake_wait_time)  # 添加随机抖动时间，以避免多个客户端同时重试

        raise ConnectionError(f"Failed to connect after {self.max_retries} attempts")

    def create_client(self, auth=True):
        """创建MongoClient实例"""
        try:
            if auth:
                return self.MongoClient(
                    host=self.host,
                    port=self.port,
                    username=self.username,
                    password=self.password,
                    authSource=self.database,
                    connectTimeoutMS=self.connect_timeout * 1000,
                    serverSelectionTimeoutMS=self.server_selection_timeout * 1000
                )
            else:
                return self.MongoClient(
                    self.host, 
                    self.port, 
                    username=self.username, 
                    password=self.password,
                    connectTimeoutMS=self.connect_timeout * 1000,
                    serverSelectionTimeoutMS=self.server_selection_timeout * 1000
                )
        except Exception as e:
            logger.error(f" !! Mongo Client creation failed: {str(e)}")
            raise

    def auth_connect(self):
        conn = None
        try:
            # conn = self.MongoClient(self.host, self.port, username=self.username, password=self.password, authSource=self.database)
            # conn = self.MongoClient(make_mongo_connect_url(self.username, self.password, self.host, self.port, self.database))
            conn = self.create_client(auth=True)
            if self.verify_connection(conn):
                return conn
            return None
        except Exception as err:
            logger.exception(f"Auth connection failed: {err}")
            return None

    def no_auth_connect(self):
        conn = None
        try:
            # conn = self.MongoClient(self.host, self.port, username=self.username, password=self.password)
            conn = self.create_client(auth=False)
            if self.verify_connection(conn):
                return conn
            return None
        except Exception as err:
            logger.exception(f"No-auth connection failed: {err}")
            return None

    def verify_connection(self, conn):
        """验证连接是否有效"""
        try:
            # 使用更可靠的ping命令检查连接
            conn.admin.command('ping')
            logger.debug(f" Mongo Connection verified to {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.warning(f" !! Mongo Connection verification failed: {str(e)}")
            return False

    def connect(self):
        """连接主逻辑"""
        logger.info(f"Connecting to MongoDB at {self.host}:{self.port}...")
        
        # 1. 尝试认证连接
        conn = self.auth_connect()
        
        # 2. 尝试非认证连接
        if not conn:
            logger.info("Trying no-auth connection...")
            conn = self.no_auth_connect()
        
        if conn:
            logger.info(f"Connected to database '{self.database}'")
            return conn
        
        raise ConnectionError("All connection attempts failed")


    def __del__(self):
        self.close_connection()
    
    def __exit__(self):
        self.close_connection()

    def close_connection(self):
        if hasattr(self, "conn") and self.conn:
            try:
                self.conn.close()
                logger.debug("MongoDB connection closed")
            except Exception as err:
                logger.exception(f"Error closing connection: {err}")
            finally:
                self.conn = None

    def insert_file(self, path=None, dict_info=dict(), force=True, specific_data_id=None):
        """
        Insert data into GridFS
        file path points to a file or a directory to be inserted
        the file extension is used to determine the data type
        dict_info is a dictionary of additional information to be stored with the file
        force is a boolean that determines whether to overwrite an existing file
        if path is None, then the data is assumed to be an empty bytes object, and only dict_info will be saved

        """
        data_info = dict()
        hash_value = ""
        type_code = -1

        # calculate hash value and data type
        if not exists(path):
            raise  FileNotFoundError(f"Path {path} does not exist.")

        if os.path.isfile(path):
            type_code = 0
            data_type = get_suffix(path)
            if data_type == "":
                data_type = 'unknown'
            if not force:
                hash_value = get_file_hash_code(path, dict_info)
            data_name = os.path.basename(path)

        elif os.path.isdir(path):
            type_code = 1
            data_type = 'directory'
            if not force:
                hash_value = get_dir_hash_code(path, dict_info)
            data_name = os.path.basename(path)

        else:
            raise RuntimeError(f"Path {path} is not a file or directory.")


        # update data_info with hash value and data type
        data_info.update(
            {
                '@hash': hash_value,
                '@type': data_type,
                "@type_code": type_code,
                "@name": data_name
            })

        if specific_data_id is not None:
            specific_data_id = ObjectId(specific_data_id)
            if not self.fs.exists({'_id': specific_data_id}):
                data_info.update({'_id': specific_data_id})
            else:
                logger.error(f" !! mongo insert specific file id {specific_data_id} already exists.")
                raise RuntimeError(f" !! mongo insert specific file id {specific_data_id} already exists.")

        # update data_info with additional information
        if isinstance(dict_info, dict):
            data_info.update(dict_info)

        if not force and self.fs.exists({'@hash': hash_value}):
            # if hash value already exists, return the data_id
            files = self.fs.find({'@hash': hash_value})
            data_id = str(files[0]._id)

        else:
            # insert data into GridFS
            if type_code == 0:
                # path points to a file
                try:
                    with open(path, 'rb') as f:
                        data_obj_id = self.fs.put(f, **data_info)
                    data_id = str(data_obj_id)

                except Exception as e:
                    raise RuntimeError(f"Error occurred while saving file to MongoDB GridFS: {e}")

            elif type_code == 1:
                # path is a directory
                # package directory to a temporary tar file
                try:
                    temp_file_name = OS_join(self.temp_dir, create_uuid() + '.zip')
                    zip_dir(path, temp_file_name)
                    with open(temp_file_name, 'rb') as f:
                        data_obj_id = self.fs.put(f, **data_info)
                    data_id = str(data_obj_id)
                    os.remove(temp_file_name)

                except Exception as e:
                    raise RuntimeError(f"Error occurred while saving directory to MongoDB GridFS: {e}")
            else:
                raise RuntimeError(f"Unknown type code: {type_code} for file insert.")

        if specific_data_id is not None:
            assert str(data_id) == str(specific_data_id), f" !! mongo file insert specific data id {specific_data_id} not equal to {data_id}"

        return data_id

    def insert_file_by_id(self, data_id, path=None, dict_info=dict(), force=True):
        specific_data_id = ObjectId(data_id)

        if not self.fs.exists({'_id': specific_data_id}):
            return self.insert_file(path, dict_info, force, data_id)

        else:
            # data id already exists, update the data
            # 1. 获取旧数据（用于回滚）
            old_mongo_obj = self.get_by_id(data_id)

            if not old_mongo_obj:
                raise ValueError(" !! Mongo FS File not found by data_id {data_id}")
            else:
                mongo_data = old_mongo_obj.read()

            # 2. 删除旧数据
            self.delete_by_id(data_id)

            # 3. 插入新数据（使用相同的 _id）
            try:
                return self.insert_file(path, dict_info, force, data_id)

            except Exception as e:
                # 4. 失败时回滚
                self.fs.put(
                    mongo_data,
                    **old_mongo_obj._file
                )
                logger.error(f" !! Mongo Update failed, rolled back: {e}")
                return data_id

    def insert_data(self, data, dict_info=dict(), force=True, specific_data_id=None):

        data_info = dict()
        hash_value = ""
        type_code = -1

        if isinstance(data, bytes):
            # data is a bytes object
            data_type = 'bytes'
            hash_value = md5(data)
            type_code = 2

        elif data is None:
            type_code = 3
            data_type = 'dict'
            if not force:
                hash_value = md5(dict_info)

        else:
            # data is a object
            data_type = 'object'
            hash_value = md5(data)
            type_code = 4
        
        # update data_info with hash value and data type
        data_info.update(
            {
                '@hash': hash_value,
                '@type': data_type,
                "@type_code": type_code,
                "@name": hash_value
            })

        if specific_data_id is not None:
            specific_data_id = ObjectId(specific_data_id)
            if not self.fs.exists({'_id': specific_data_id}):
                data_info.update({'_id': specific_data_id})
            else:
                logger.error(f" !! mongo insert specific data id {specific_data_id} already exists.")
                raise RuntimeError(f" !! mongo insert specific data id {specific_data_id} already exists.")

        # update data_info with additional information
        if isinstance(dict_info, dict):
            data_info.update(dict_info)

        if not force and self.fs.exists({'@hash': hash_value}):
            # if hash value already exists, return the data_id
            files = self.fs.find({'@hash': hash_value})
            data_id = str(files[0]._id)

        else:
            # insert data into GridFS
            if type_code == 2:
                # data is a bytes object
                try:
                    data_obj_id = self.fs.put(data, **data_info)
                    data_id = str(data_obj_id)
                except Exception as e:
                    raise RuntimeError(f"Error occurred while saving bytes to MongoDB GridFS: {e}")

            elif type_code == 3:
                # data is a dict
                try:
                    data_obj_id = self.fs.put(b"", **data_info)
                    data_id = str(data_obj_id)

                except Exception as e:
                    raise RuntimeError(f"Error occurred while saving dict to MongoDB GridFS: {e}")

            elif type_code == 4:
                # data is a object
                pickle_data = pickle.dumps(data)

                try:
                    data_obj_id = self.fs.put(pickle_data, **data_info)
                    data_id = str(data_obj_id)
                except Exception as e:
                    raise RuntimeError(f"Error occurred while saving object to MongoDB GridFS: {e}")

            else:
                raise RuntimeError(f"Mongo indert fun got an Unknown type code: {type_code}")

        if specific_data_id is not None:
            assert str(data_id) == str(specific_data_id), f" !! mongo insert specific data id {specific_data_id} not equal to {data_id}"

        return data_id

    def insert_data_by_id(self, data_id, data, dict_info=dict(), force=True):
        specific_data_id = ObjectId(data_id)

        if not self.fs.exists({'_id': specific_data_id}):
            return self.insert_data(data, dict_info, force, data_id)

        else:
            # data id already exists, update the data
            # 1. 获取旧数据（用于回滚）
            old_mongo_obj = self.get_by_id(data_id)

            if not old_mongo_obj:
                raise ValueError(" !! Mongo FS File not found by data_id {data_id}")
            else:
                mongo_data = old_mongo_obj.read()

            # 2. 删除旧数据
            self.delete_by_id(data_id)

            # 3. 插入新数据（使用相同的 _id）
            try:
                return self.insert_data(data, dict_info, force, specific_data_id)

            except Exception as e:
                # 4. 失败时回滚
                self.fs.put(
                    mongo_data,
                    **old_mongo_obj._file
                )
                logger.error(f" !! Mongo Update failed, rolled back: {e}")
                return data_id

    def close(self):
        self.conn.close()

    def clear_data(self, declaration=""):
        if declaration == "I am sure to clear all data in the database.":
            return self.delete_by_condition(condition={})
        else:
            raise RuntimeError("Are you sure you want to clear all data in the database? Please enter 'I am sure to clear all data in the database.' to confirm.")

    def delete_by_condition(self, condition=None):
        if condition is None:
            return 0

        assert isinstance(condition, dict), f"condition {condition} must be a dict"
        # condition = {data_type_id: image_id}

        files_to_delete = self.get_by_condition(condition)
        del_data_num = 0

        # 删除每个找到的文件
        for file in files_to_delete:
            self.fs.delete(file._id)
            del_data_num += 1

        return del_data_num

    def delete_by_id(self, data_id):
        data_id = ObjectId(str(data_id))
        if self.fs.exists(data_id):
            self.fs.delete(data_id)

    def get_by_condition(self, condition):

        if condition is None:
            return None

        assert isinstance(condition, dict), f"condition {condition} must be a dict"

        files = self.fs.find(condition)
        return list(files)

    def get_by_id(self, data_id):
        if data_id is None:
            return None

        data_id = ObjectId(str(data_id))
        
        condition = {'_id': data_id}
        files = list(self.get_by_condition(condition))

        if len(files) == 0:
            return None
        elif len(files) > 1:
            raise RuntimeError(f"More than one file found with id {data_id}")
        else:
            return files[0]

    def get_and_parse(self, data_id, save_dir='mongo_data', force_save=False):
        mogo_obj = self.get_by_id(data_id)
        if mogo_obj is None:
            return None

        # data = pickle.loads(mogo_obj.read())
        return self.parse(mogo_obj, save_dir, force_save)

    def get_data_num(self):
        return len(list(self.get_by_condition({})))
    
    def save(self, data_id, dir_path=None, file_path=None):
        mogo_obj = self.get_by_id(data_id)

        if mogo_obj is None:
            return False

        save_path = None
        if file_path is not None:
            save_path = file_path
        elif dir_path is not None:
            assert dir_check(dir_path), f"dir_path {dir_path} is not a valid directory"
            save_path = os.path.join(dir_path, mogo_obj._file['@name'])
        else:
            raise RuntimeError("Either dir_path or file_path must be specified")
        
        assert save_path is not None, f"save_path {save_path} is not valid"

        # save data
        with open(save_path, 'wb') as f:
            f.write(mogo_obj.read())

        return True