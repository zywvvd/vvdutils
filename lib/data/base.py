# -*- coding: utf-8 -*-
# @Author: Zhai Menghua
# @Date:   2020-07-17 11:01:04
# @Last Modified by:   Zhai Menghua
# @Last Modified time: 2020-07-21 20:59:00
#
# Data Manager Class
#
import copy
import random
import os
import json
import pickle

from tqdm import tqdm
from numpy.lib.function_base import iterable
from ..utils import MyEncoder


def _DEFAULT_KEY_CALLBACK(rec):
    return rec['info']['uuid']

class DataManagerBase(object):
    def __init__(self, record_list, class_list):
        assert isinstance(record_list, (list, tuple))
        self.record_list = record_list
        self.class_list = class_list

    def filter(self, condition_callback, verbose=False):
        """
        Filter data by the given condition
        Param:
            condition_callback: a callback function that returns a boolean value according to attributes
            verbose: if set True, then print information about the operation
        Return:
            a DataManager instance, whose record_list satisfies the given condition
        """
        try:
            if verbose:
                record_list = [copy.deepcopy(x) for x in tqdm(self.iterator(), total=len(self)) if condition_callback(x)]
            else:
                record_list = [copy.deepcopy(x) for x in self.iterator() if condition_callback(x)]
        except RuntimeError as e:
            print("condition_callback Error:", e)
        data = type(self)(record_list, copy.copy(self.class_list))
        if verbose:
            print("{} out of {} images are found meeting the given condition".format(len(record_list), len(self)))
        return data

    def __eq__(self, another_dataset):
        assert isinstance(another_dataset, type(self))
        # check class list
        if not all([x==y for x, y in zip(self.class_list, another_dataset.class_list)]):
            return False

        # check occurrence of uuids
        key_callback=lambda rec: rec['info']['uuid']
        uuid_dict1 = self.occurrence(key_callback)
        uuid_dict2 = another_dataset.occurrence(key_callback)
        if uuid_dict1 != uuid_dict2:
            return False

        return True


    def __len__(self):
        return len(self.record_list)

    def __getitem__(self, idx):
        return self.record_list[idx]

    def clone(self):
        return copy.deepcopy(self)

    def iterator(self, shuffle=False):
        random_ids = list(range(len(self)))
        if shuffle:
            random.shuffle(random_ids)
        for idx in random_ids:
            yield self.record_list[idx]

    def shuffle(self):
        """ return a shuffled iterator (record can be edit in place) """
        return self.iterator(shuffle=True)

    def record_shuffle(self):
        random.shuffle(self.record_list)

    def split(self, num_or_ratio, groupID_callback=None, random_seed=123):
        """
        Split data by the given condition
        Param:
            num_or_ratio: number (int) or ratio (float) of the first half of the splits (by groupID)
            groupID_callback: a callback function that returns a hashable object (groupID); if unset, split the dataset itemwise
        Return:
            two DataManager instances, the groupID computed from the record_list 
            of one differs from the values computed from the record_list of the other
        """
        # 1) put data in group (by their groupID)
        groups = dict()
        for ix, record in enumerate(self.iterator()):
            if groupID_callback == None:
                group_id = ix
            else:
                try:
                    group_id = groupID_callback(record)
                except RuntimeError as e:
                    print("groupID_callback Error:", e)
            if group_id not in groups:
                groups[group_id] = [record]
            else:
                groups[group_id].append(record)
        num_groups = len(groups)

        # 2) split data by groups
        if None: pass
        elif iterable(num_or_ratio):
            # support ratio split like split([7,2,1])
            ratios = list(num_or_ratio) + [0]
            datasets = list()
            data = self.clone()
            for idx in range(len(ratios)-1):
                r = ratios[idx] / sum(ratios[idx:])
                ds, data = data.split(r, groupID_callback, random_seed)
                datasets.append(ds)
            assert len(data) == 0
            return datasets
        elif isinstance(num_or_ratio, float):
            assert 0 <= num_or_ratio <= 1
            num_firsthalf = int(num_groups * num_or_ratio)
        elif isinstance(num_or_ratio, int):
            num_firsthalf = num_or_ratio

        assert num_firsthalf <= num_groups, "There're less than {} groups to split: {} < {}".format(num_firsthalf, num_groups, num_firsthalf)

        # split data
        groups = [groups[key] for key in sorted(groups.keys())]  # sort groups by key for repeatibility
        random.seed(random_seed)
        random.shuffle(groups)
        firsthalf, lasthalf = groups[:num_firsthalf], groups[num_firsthalf:]
        record_list1 = list()
        for grp in firsthalf:
            record_list1.extend([copy.deepcopy(x) for x in grp])
        record_list2 = list()
        for grp in lasthalf:
            record_list2.extend([copy.deepcopy(x) for x in grp])

        data1 = type(self)(record_list1, copy.copy(self.class_list))
        data2 = type(self)(record_list2, copy.copy(self.class_list))

        return data1, data2

    def union(self, another_dataset):
        """
        Fuse two datasets
        Param:
            another_dataset: a DataManager instance
        Return:
            a DataManager instance that fused self and another_dataset
        """
        assert isinstance(another_dataset, type(self))
        assert all([x==y for x, y in zip(self.class_list, another_dataset.class_list)])
        record_list_merged = [copy.deepcopy(x) for x in self.record_list+another_dataset.record_list]
        data_merged = type(self)(record_list_merged, copy.copy(self.class_list))
        return data_merged

    def merge(self, another_dataset):
        return self.union(another_dataset)

    def dump(self):
        """ Print all info of the dataset """
        import pprint
        for ix, record in enumerate(self.iterator()):
            print("\n[{}/{}] sample >>".format(ix+1, len(self)))
            pprint.pprint(record)
        print("\nClass Dict:")
        print(self.class_list)

    def extract_info(self, info_callback):
        """
        Extract infomation extracted by the info_callback function
        Param:
            info_callback: callback function that extracts info from record
        Return:
            a list of info items extracted by info_callback
        """
        return list(map(info_callback, self.iterator()))

    def occurrence(self, key_callback=_DEFAULT_KEY_CALLBACK):
        """
        Count the occurrence of record
        Param:
            key_callback: a callback function that returns a hashable object (ie. groupID) or a list of hashable objects
        Return:
            a dictionary: {key_callback(record1): occurrence1, ...}
        """
        occurrence = dict()
        for record in self.iterator():
            try:
                key = key_callback(record)
            except RuntimeError as e:
                print("key_callback Error:", e)
            # if the extract key is a list (or tuple)
            if isinstance(key, list) or isinstance(key, tuple):
                keys = key
            else:
                keys = [key]
            for key in keys:
                if not key in occurrence:
                    occurrence[key]  = 1
                else:
                    occurrence[key] += 1
        return occurrence

    def unique(self, key_callback=_DEFAULT_KEY_CALLBACK, verbose=False):
        """
        Remove duplicated record by its key (the removal choices are made randomly)
        Param:
            key_callback: a callback function that returns a hashable object (ie. groupID) or a list of hashable objects
            verbose: if set True, then print information about the operation
        Return:
            a new DataManager object, whose records have unique key returned by key_callback
        """
        class_list = copy.deepcopy(self.class_list)
        record_list = list()
        visited_key = set()
        for record in self.iterator():
            try:
                key = key_callback(record)
            except RuntimeError as e:
                print("key_callback Error:", e)
            if key not in visited_key:
                record_list.append(copy.deepcopy(record))
                visited_key.add(key)
        if verbose:
            print("{} out of {} duplicated records are found".format(len(self)-len(record_list), len(self)))
        return type(self)(record_list=record_list, class_list=class_list)

    def intersection(self, another_dataset, key_callback=_DEFAULT_KEY_CALLBACK, verbose=False):
        """
        Find the intersection set of two datasets, if two records share the same key, they are deemed as members of the intersection set
        Param:
            another_dataset: a DataManager instance
            key_callback: a callback function that returns a hashable object (ie. image uuid or image hash code)
            verbose: if set True, then print information about the operation
        Return:
            a DataManager instance that represents for the intersection of two datasets
        """
        assert isinstance(another_dataset, type(self))
        assert all([x==y for x, y in zip(self.class_list, another_dataset.class_list)])
        class_list = copy.deepcopy(self.class_list)
        intersection_keys = set(another_dataset.extract_info(info_callback=key_callback))
        record_list = list()
        for record in self.iterator():
            try:
                key = key_callback(record)
            except RuntimeError as e:
                print("key_callback Error:", e)
            if key in intersection_keys:
                record_list.append(copy.deepcopy(record))
        if verbose:
            print("{} out of {} overlapping records are found".format(len(record_list), len(self)))
        return type(self)(record_list=record_list, class_list=class_list)
        
    def difference(self, another_dataset, key_callback=_DEFAULT_KEY_CALLBACK, verbose=False):
        """
        Find the difference set of two datasets, whose members exist in self but not in another_dataset
        Param:
            another_dataset: a DataManager instance
            key_callback: a callback function that returns a hashable object (ie. image uuid or image hash code)
            verbose: if set True, then print information about the operation
        Return:
            a DataManager instance that represents for the difference set of two datasets
        """
        assert isinstance(another_dataset, type(self))
        assert all([x==y for x, y in zip(self.class_list, another_dataset.class_list)])
        class_list = copy.deepcopy(self.class_list)
        intersection_keys = set(another_dataset.extract_info(info_callback=key_callback))
        record_list = list()
        for record in self.iterator():
            try:
                key = key_callback(record)
            except RuntimeError as e:
                print("key_callback Error:", e)
            if key not in intersection_keys:
                record_list.append(copy.deepcopy(record))
        if verbose:
            print("{} out of {} different records are found".format(len(record_list), len(self)))
        return type(self)(record_list=record_list, class_list=class_list)


    def data_statistics(self):
        raise NotImplementedError()


    def save_json(self, json_file):
        """
        Save dataset to json file
        Param:
            json_file: path to the json file where you want to save your dataset
        """
        json_dir = os.path.dirname(json_file)
        if json_dir:
            os.makedirs(json_dir, exist_ok=True)

        data_statistics = self.data_statistics()
        json_object = {
            'data_statistics': self.data_statistics(),
            'record': self.record_list,
            'class_list': self.class_list,
        }

        # save json object
        with open(json_file, 'w') as fid:
            json.dump(
                json_object,
                fid,
                indent = 4,
                cls=MyEncoder
            )


    def save_pickle(self, pickle_file):
        """
        Save dataset to pickle file
        Param:
            pickle_file: path to the pickle file where you want to save your dataset
        """
        pickle_dir = os.path.dirname(pickle_file)
        if pickle_dir:
            os.makedirs(pickle_dir, exist_ok=True)

        data_statistics = self.data_statistics()
        pickle_object = {
            'data_statistics': self.data_statistics(),
            'record': self.record_list,
            'class_list': self.class_list,
        }

        # save json object
        with open(pickle_file, 'wb') as fid:
            pickle.dump(pickle_object, fid)

    @classmethod
    def from_json(cls, json_file):
        """
        Load dataset from json file
        Param:
            json_file: path to the json file where you want to read. Its format has to satisfy: {'class_list': ..., 'record': ...}
        Return:
            a DataManager instance
        """
        assert os.path.exists(json_file), "{} doesn't exist".format(json_file)
        with open(json_file) as fid:
            obj = json.load(fid)

        assert 'record' in obj, "invald json file to load: {}".format(json_file)
        if 'class_list' in obj:
            class_list = obj['class_list']
        elif 'class_dict' in obj:
            class_list = obj['class_dict']
        else:
            raise RuntimeError("invald json file to load: {}".format(json_file))

        record_list = list()
        for rec in obj['record']:
            record_list.append(rec)
        dataset = cls(record_list=record_list, class_list=class_list)

        return dataset

    @classmethod
    def from_pickle(cls, pickle_file):
        """
        Load dataset from pickle file
        Param:
            pickle_file: path to the json file where you want to read. Its format has to satisfy: {'class_list': ..., 'record': ...}
        Return:
            a DataManager instance
        """
        assert os.path.exists(pickle_file), "{} doesn't exist".format(pickle_file)
        with open(pickle_file, 'rb') as fid:
            obj = pickle.load(fid)

        assert 'class_list' in obj and 'record' in obj, "invald json file to load: {}".format(pickle_file)
        class_list = obj['class_list']
        record_list = list()
        for rec in obj['record']:
            record_list.append(rec)
        dataset = cls(record_list=record_list, class_list=class_list)

        return dataset

    @classmethod
    def load(cls, filepath):
        ext = os.path.basename(filepath).split('.')[-1].lower()
        if ext in ('pickle', 'pkl'):
            return cls.from_pickle(filepath)
        else:
            return cls.from_json(filepath)

    def save(self, filepath, *args):
        ext = os.path.basename(filepath).split('.')[-1].lower()
        if ext in ('pickle', 'pkl'):
            return self.save_pickle(filepath, *args)
        else:
            return self.save_json(filepath, *args)

    def batch(self, batch_size, shuffle=False, random_seed=123):
        """
        Divide the dataset into a series of batches, randomly
        Param:
            batch_size: #records this method will return per iteration
        Return:
            an iterator of DataManager objects
        """
        assert isinstance(batch_size, int), "batch_size should be a natural number"
        if batch_size > len(self):
            batch_size = len(self)
        assert 0 < batch_size
        ds = self.clone()
        record_list = ds.record_list
        if shuffle:
            random.seed(random_seed)
            random.shuffle(record_list)
        for idx in range(0, len(record_list), batch_size):
            records = record_list[idx:idx+batch_size]
            yield type(self)(record_list=records, class_list=ds.class_list)

    def chunk(self, chunk_num, shuffle=False, random_seed=123):
        """
        Chunk the dataset (divide it into a given number of chunks), randomly
        Param:
            chunk_num: #chunks this method will return
        Return:
            an iterator of DataManager objects
        """
        assert isinstance(chunk_num, int), "chunk_num should be a natural number"
        assert 0 < chunk_num <= len(self)
        batch_size = len(self) // chunk_num
        if len(self) % chunk_num > 0:
            batch_size += 1
        return self.batch(batch_size, shuffle, random_seed)

    def zip(self, *args, key_callback=_DEFAULT_KEY_CALLBACK):
        """
        Iterate over multiple datasets
        Param:
            args: other datasets
            key_callback: a callback function that takes a record and return a 
                          hashable object which is used to align elements between
                          different datasets. If there're more than one records
                          sharing the same key, return them as a list during iteration
        Return:
            an iterator of record tuples
        Usage:
            for rec1, rec2, rec3 in data1.zip(data2, data3):
                ...
        """
        def _pack_records(dataset):
            rec_dict = dict()
            for rec in dataset:
                key = key_callback(rec)
                if key not in rec_dict:
                    rec_dict[key] = [rec]
                else:
                    rec_dict[key].append(rec)
            return rec_dict

        # prepare data
        rec_dict_list = [_pack_records(self)]
        for dataset in args:
            rec_dict = _pack_records(dataset)
            assert len(rec_dict) == len(rec_dict_list[0]), "Unmatched dataset under current key_callback"
            rec_dict_list.append(rec_dict)
        
        # loop over records that shares the same key among dataset
        for key in rec_dict_list[0].keys():
            ret = list()
            for dataset in rec_dict_list:
                recs = dataset[key]
                if len(recs) == 1:
                    ret.append(recs[0])
                else:
                    ret.append(recs)
            yield ret

    def __mul__(self, obj):
        if isinstance(obj, (int, float)):
            number = int(obj)
            assert number > 0
            new_record_list = self.record_list * number
            return (type(self))(new_record_list, self.class_list)
        else:
            assert isinstance(obj, type(self)), type(obj)
            another_dataset = obj
            return self.intersection(another_dataset)
    
    # overwrite operator '+'
    __add__ = union
    __or__  = union
    __and__ = intersection
    __sub__ = difference
    __truediv__ = difference
