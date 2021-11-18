import numpy as np
import pandas as pd

import os
from os import listdir
from os.path import isfile, join


class Dataset:
    def __init__(self, features, targets, weights, next_state_id=None):
        self.features = features
        self.targets = targets
        self.weights = weights
        self.next_state_id = next_state_id


def read_files_in_dir(_dir: str)->list:
    return [os.path.join(_dir, f) for f in listdir(_dir) if isfile(join(_dir, f))]


def calc_weights(city_tiles: np.array, opp_city_tiles: np.array, turns: np.array, unit_ids: np.array, r) -> np.array:
#     target_fn=lambda my, opp: my * turns * 1e-5 #np.ones_like(my) / 500
#     target_fn=lambda my, opp: my * 1e-4
    target_fn=lambda my, opp: np.zeros_like(my)
    time_vals = target_fn(city_tiles, opp_city_tiles)
    assert r is not None
    last_turn_ids = np.where(turns == np.max(turns))
#     time_vals[np.where(turns == np.max(turns))] = my[np.where(turns == np.max(turns))]
    time_vals[last_turn_ids] = city_tiles[last_turn_ids] - opp_city_tiles[last_turn_ids]
    return time_vals * 1e-2


CONVERTER = { 'bcity': 0, 'p': 1, 'n': 2, 's': 3,  'e': 4, 'w': 5}
CAT_FEATURES_V3 = [2,3,4,6,11,16,17,22,28,29,30,31]
CAT_FEATURES_V4 = [2,3,4,8,14,16,22,24,25,32,38,39,40,41]


def calc_targets(actions: list) -> np.array:
    def _process(a: str) -> str:
        if a == None:
            return a
        a = a.split()
        if a[0] == "m":
            if a[-1] == "c":
                return "p"
            return a[-1]
        return a[0]
    
    targets = [CONVERTER[_process(a)] for a in actions]
    return np.array(targets)


def calc_next_state_id(unit_ids:np.array, turns:np.array):
    next_ids = np.ones_like(unit_ids) * (-1)
    last_id = dict()
    for i,_id in enumerate(unit_ids):
        if _id in last_id:
            next_ids[last_id[_id]] = i
        last_id[_id] = i
    return next_ids


def get_dataset_from_file(f,r=None) -> Dataset:
    df = pd.read_csv(f, sep='\t')
    targets = calc_targets(df.values[:, -5])
    turns = df.values[:, -2]
    unit_ids = df.values[:, -1]
    weights = calc_weights(df.values[:, -4], df.values[:, -3], turns, unit_ids, r)
    features = df.values[:, :-5]
    next_state_id = calc_next_state_id(unit_ids, turns)
    return Dataset(features, targets, weights, next_state_id)


def read_columns_from_random_file(_dir: str)->list:
    files = read_files_in_dir(_dir)
    return list(enumerate(pd.read_csv(files[0], sep='\t').columns))



def read_datasets_from_dir(_dir: str) -> list:
    files = read_files_in_dir(_dir)
    return [get_dataset_from_file(f) for f in files]


def concat_datasets(_d: list)-> Dataset:
    features = np.concatenate([d.features for d in _d], axis=0)
    targets = np.concatenate([d.targets for d in _d], axis=0)
    weights = np.concatenate([d.weights for d in _d], axis=0)
    assert _d.next_state_id is None
    return Dataset(features, targets, weights)


def read_partitioned_dataset(_dir: str) -> Dataset:
    d = read_datasets_from_dir(_dir)
    return concat_datasets(d)
