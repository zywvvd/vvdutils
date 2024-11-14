import os
import io

from os.path import exists
from bson import ObjectId
import pickle
import time

from ...utils import lazy_import
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



class MongoGridFSConnection:
    MongoClient = None
    GridFS = None

    DATA_TYPES = ['file', 'dir', 'bytes', 'object', 'none']
    TEMP_DIR = './tmp/mongodb_tmp'

    def __init__(self, username, password, host, port, database, temp_dir=None):
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
        self.conn = self.connect()

        self.db = self.conn[self.database]
        self.fs = self.GridFS(self.db)

        if temp_dir is None:
            self.temp_dir = self.TEMP_DIR
        else:
            self.temp_dir = temp_dir

        assert dir_check(self.temp_dir), f"Temp directory {self.temp_dir} create failed."

    def parse(self, mongo_obj, save_dir='mongo_data'):
        obj_type_code = mongo_obj._file['@type_code']
        obj_type = mongo_obj._file['@type']
        name = mongo_obj._file['@name']

        if obj_type_code == 0:
        # file 
            if obj_type in popular_image_extensions:
                # image
                return read_mongodb_image(mongo_obj)
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

    def connect(self):
        conn = None
        try:
            conn = self.MongoClient(self.host, self.port, username=self.username, password=self.password)
        except Exception as err:
            raise RuntimeError(f" !! MongoDB connect error: {err}")

        time.sleep(0.5)

        if len(conn.nodes) == 0:
            raise RuntimeError(" !! MongoDB connect failed.")
        
        try:
            server_info = conn.server_info()
            print(f"MongoDB connect success, server info: {server_info['version']}")
        except Exception as err:
            raise RuntimeError(f" !! MongoDB Authentication failed: {err}")

        return conn

    def insert_file(self, path=None, dict_info=dict(), force=False, specific_data_id=None):
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
            assert not self.fs.exists({'_id': specific_data_id}), f"MongoDB insert_file: Data with id {specific_data_id} already exists."
            data_info.update({'_id': specific_data_id})

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

        return data_id

    def insert_data(self, data, dict_info=dict(), force=False, specific_data_id=None):

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
            assert not self.fs.exists({'_id': specific_data_id}), f"MongoDB insert_file: Data with id {specific_data_id} already exists."
            data_info.update({'_id': specific_data_id})

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
                raise RuntimeError(f"Unknown type code: {type_code}")

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

    def get_and_parse(self, data_id, save_dir='mongo_data'):
        mogo_obj = self.get_by_id(data_id)
        if mogo_obj is None:
            return None

        # data = pickle.loads(mogo_obj.read())
        return self.parse(mogo_obj, save_dir)

    def update_data_by_id(self, data_id, data, dict_info, data_type):
        raise RuntimeError("update_data_by_id is not implemented yet")

        assert data_type in self.DATA_TYPES, f"Unknown data type: {data_type} while updating data. "
        assert isinstance(dict_info, dict), f"dict_info {dict_info} must be a dict"

        data_id = ObjectId(str(data_id))

        # check data before update
        if data_type == 'file':
            assert os.path.isfile(data), f"Data {data} is not a file while updating data. "
        elif data_type == 'dir':
            assert os.path.isdir(data), f"Data {data} is not a directory while updating data. "
        elif data_type == 'bytes':
            assert isinstance(data, bytes), f"Data {data} is not bytes while updating data. "
        elif data_type == 'object':
            assert isinstance(data, object), f"Data {data} is not object while updating data. "
        elif data_type == 'none':
            assert data is None, f"Data {data} is not None while updating data. "
        else:
            raise RuntimeError(f"Unknown data type: {data_type} while updating data. ")
        
        # update data
        self.delete_by_id(data_id)
        if data_type == 'file' or data_type == 'dir' or data_type == 'none':
            self.insert_file(path=data, dict_info=dict_info, specific_data_id=data_id)
        elif data_type == 'bytes' or data_type == 'object':
            self.insert_data(data=data, dict_info=dict_info, specific_data_id=data_id)
        else:
            raise RuntimeError(f"Unknown data type: {data_type} while updating data. ")
        
        return data_id

    def get_data_num(self):
        return len(list(self.get_by_condition({})))