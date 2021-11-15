import numpy as np
import os
import sys
import math

import catboost

from lux.constants import Constants

from routines import unit_cargo_total, unit_cargo_fuel_total

FEATURE_LOG_PATH = None
FEATURE_CACHE = dict()

MODEL = None
TURN = 0


def create_or_get_cb_model(model_path: str):
    global MODEL
    if MODEL is not None:
        return MODEL
    MODEL = catboost.CatBoostClassifier().load_model(model_path)
    return MODEL


import torch
from torch import nn


MAP_F = 32 * 32 * 7

MAP_F = 32 * 32 * 7


class NNWithCustomFeatures(nn.Module):
    def __init__(self, INPUT_F, DROP_P, H, A=6):
        super().__init__()
        INPUT_F_C = INPUT_F + 128
        self.model_q = nn.Sequential(
            nn.Dropout(DROP_P),
            nn.Linear(INPUT_F_C, H),
            nn.LayerNorm(H),
            nn.ReLU(),
            nn.Dropout(DROP_P),
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Dropout(DROP_P),
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Linear(H, A),
            nn.ReLU()
            #             nn.Sigmoid()
        )

        self.model_p = nn.Sequential(
            nn.Dropout(DROP_P),
            nn.Linear(INPUT_F_C, H),
            nn.LayerNorm(H),
            nn.ReLU(),
            nn.Dropout(DROP_P),
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Dropout(DROP_P),
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Linear(H, A)
            #             nn.Sigmoid()
        )

        self.map_model = nn.Sequential(
            nn.Conv2d(7, 64, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # after -> (16,16)
            nn.Conv2d(64, 128, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # after -> (8, 8)
            nn.Conv2d(128, 256, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # after -> (4, 4)
            #             nn.Conv2d(128, 256, 3),
            #             nn.ReLU(inplace=True),
            #             nn.MaxPool2d(2),  # after -> (1, 1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.proj = nn.Sequential(
            nn.Dropout(p=DROP_P),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DROP_P),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        L = x.shape[1]
        cur_r = self.forward_impl(x[:, :L // 2])
        next_r = self.forward_impl(x[:, L // 2:])
        return torch.cat([cur_r, next_r], dim=1)

    def forward_impl(self, x):
        mapp = x[:, -MAP_F:].reshape(-1, 32, 32, 7)
        rest = x[:, :-MAP_F]
        mapp = torch.transpose(mapp, 1, -1)
        mapp = self.avgpool(self.map_model(mapp))
        mapp = torch.flatten(mapp, 1)
        mapp_f = self.proj(mapp)
        #         print(mapp_f.shape)
        input_x = torch.cat([rest, mapp_f], dim=1)
        #         print(input_x.shape)
        #         return self.model_q(input_x)
        return torch.cat([self.model_q(input_x), self.model_p(input_x)], dim=1)


class NNModelWrapper:
    def __init__(self, model):
        self.model = model

    def predict_proba(self, x: np.array) -> np.array:
        with torch.no_grad():
            _l = self.model.forward(torch.Tensor(x).reshape(1, -1)).cpu().detach().numpy().reshape(-1)
            _l = np.clip(_l, -30, 30)
            assert _l.shape[0] == 12
            _l = _l[6:]
            p = np.exp(_l) / np.sum(np.exp(_l))
            return p.reshape(1, -1)


def create_or_get_nn_model(model_path: str):
    global MODEL
    if MODEL is not None:
        return MODEL
    model = NNWithCustomFeatures(83, 0.05, 64)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    MODEL = NNModelWrapper(model)
    return MODEL


def create_or_get_file(log_path: str, features_keys, env):
    global FEATURE_LOG_PATH
    if FEATURE_LOG_PATH is not None:
        return FEATURE_LOG_PATH
    dp = env.get("log_path_file_name", None)
    if dp is None:
        dp = "features_" + str(np.random.random()) + ".txt"
    path = os.path.join(log_path, dp)
    FEATURE_LOG_PATH = path
    with open(path, "w") as f:
        f.write("\t".join(features_keys) + "\taction\tmy_tiles\topp_tiles\tturn\tunit_id\n")
    print("FEATURE_LOG_PATH=" + FEATURE_LOG_PATH, file=sys.stderr)
    return path


def calc_city_tiles(player) -> list:
    city_tiles: list = []
    for city in player.cities.values():
        for city_tile in city.citytiles:
            city_tiles.append(city_tile)
    return city_tiles


def calc_nearest_city_features(player, unit, city_tiles, prefix_="") -> dict:
    closest_dist = 1e9
    closest_city_tile = None
    move_dir = None
    fuel = None
    light_upkeep = None
    city_size = None
    tx, ty = None, None
    is_survive = None
    for city_tile in np.random.permutation(city_tiles):
        dist = city_tile.pos.distance_to(unit.pos)
        if dist < closest_dist:
            closest_dist = dist
            closest_city_tile = city_tile
    if closest_city_tile is not None:
        move_dir = unit.pos.direction_to(closest_city_tile.pos)
        city = player.cities[closest_city_tile.cityid]
        fuel = city.fuel
        light_upkeep = city.light_upkeep
        city_size = len(city.citytiles)
        tx = closest_city_tile.pos.x
        ty = closest_city_tile.pos.y
        is_survive = light_upkeep * 10 <= fuel
    res = {
        "near_city_dist": closest_dist,
        "near_city_dir": move_dir,
        "near_city_fuel": fuel,
        "near_city_light_upkeep": light_upkeep,
        "city_size": city_size,
        "near_city_coord_x": tx,
        "near_city_coord_y": ty,
        "near_city_survive_at_night": is_survive
    }
    return {prefix_ + k: v for k, v in res.items()}


def calc_nearest_resource_features(player, unit, resource_tiles, prefix_="") -> dict:
    closest_dist = 1e9
    closest_tile = None
    move_dir = None
    type = None
    amount = None
    tx, ty = None, None
    for tile in np.random.permutation(resource_tiles):
        if tile.resource.type == Constants.RESOURCE_TYPES.COAL and not player.researched_coal(): continue
        if tile.resource.type == Constants.RESOURCE_TYPES.URANIUM and not player.researched_uranium(): continue
        dist = tile.pos.distance_to(unit.pos)
        if dist < closest_dist:
            closest_dist = dist
            closest_tile = tile
    if closest_tile is not None:
        move_dir = unit.pos.direction_to(closest_tile.pos)
        type = closest_tile.resource.type
        amount = closest_tile.resource.amount
        tx = closest_tile.pos.x
        ty = closest_tile.pos.y
    res = {
        "near_resource_dist": closest_dist,
        "near_resource_dir": move_dir,
        "near_resource_type": type,
        "near_resource_amount": amount,
        "near_resource_coord_x": tx,
        "near_resource_coord_y": ty
    }
    return {prefix_ + k: v for k, v in res.items()}


def calc_common_features(city_tiles, opp_city_tiles, turn, width, height, player, opponent) -> dict:
    return {
        "my_city_count": len(city_tiles),
        "opp_city_count": len(opp_city_tiles),
        "turn": turn,
        "is_night": turn % 40 >= 30,
        "time_to_night": max(0, 30 - turn % 40),
        "width": width,
        "height": height,
        "my_research": player.research_points,
        "opp_research": opponent.research_points,
        "my_research_coal": player.researched_coal(),
        "opp_research_coal": opponent.researched_coal(),
        "my_research_uran": player.researched_uranium(),
        "opp_research_uran": opponent.researched_uranium(),
    }


def calc_resource_tiles(height, width, game_state):
    resource_tiles: list = []
    for y in range(height):
        for x in range(width):
            cell = game_state.map.get_cell(x, y)
            if cell.has_resource():
                resource_tiles.append(cell)
    return resource_tiles


from enum import Enum

MAP_STATE = dict()

def calc_map_features(game_state, my_cities, op_cities, width, height, player, opp, turn):
    global MAP_STATE
    if turn in MAP_STATE:
        return MAP_STATE[turn]
    class Feature:
        TREE = 0
        COAL = 1
        URAN = 2
        MY_CITY = 3
        OPP_CITY = 4
        MY_UNIT = 5
        OPP_UNIT = 6
    fmap = np.zeros((32, 32, 7), dtype=float)   #[is_tree, is_coal, is_uran, is_my_city,is_opp_city, is_my_unit, is_opp_unit]
    for t in my_cities:
        city = player.cities[t.cityid]
        fuel = city.fuel
        fmap[t.pos.x][t.pos.y][Feature.MY_CITY] = fuel

    for t in op_cities:
        city = opp.cities[t.cityid]
        fuel = city.fuel
        fmap[t.pos.x][t.pos.y][Feature.OPP_CITY] = fuel

    for y in range(height):
        for x in range(width):
            cell = game_state.map.get_cell(x, y)
            if cell.has_resource():
                amount = cell.resource.amount
                type = cell.resource.type
                coord = None
                if type == "wood":
                    coord = Feature.TREE
                if type == "coal":
                    coord = Feature.COAL
                if type == "uranium":
                    coord = Feature.URAN
                fmap[x][y][coord] = amount

    for unit in player.units:
        if unit.is_worker():
            fmap[unit.pos.x][unit.pos.y][Feature.MY_UNIT] = 1 + unit_cargo_fuel_total(unit)

    for unit in opp.units:
        if unit.is_worker():
            fmap[unit.pos.x][unit.pos.y][Feature.OPP_UNIT] = 1 + unit_cargo_fuel_total(unit)

    MAP_STATE[turn] = fmap
    return fmap




def calc_features(unit, game_state, turn, player, opponent, width, height, unit_routine, last_action, log_path):
    global FEATURE_CACHE, TURN
    TURN = turn
    if (unit.id, turn) in FEATURE_CACHE:
        return FEATURE_CACHE[(unit.id, turn)]
    city_tiles = calc_city_tiles(player)
    opp_city_tiles = calc_city_tiles(opponent)
    resource_tiles = calc_resource_tiles(height, width, game_state)

    near_city_features = calc_nearest_city_features(player, unit, city_tiles)
    near_opp_city_features = calc_nearest_city_features(opponent, unit, opp_city_tiles, "opp_")
    near_resource_features = calc_nearest_resource_features(player, unit, resource_tiles)
    common_features = calc_common_features(city_tiles, opp_city_tiles, turn, width, height, player, opponent)
    assert len(near_city_features) == 8, len(near_city_features)
    assert len(near_opp_city_features) == 8, len(near_opp_city_features)
    assert len(common_features) == 13, len(common_features)
    assert len(near_resource_features) == 6, len(near_resource_features)

    map_features = calc_map_features(game_state, city_tiles, opp_city_tiles, width, height, player, opponent, turn)
    assert np.prod(map_features.shape) == 32 * 32 * 7
    map_features_buffer = np.ones((32,32,7), dtype=float) * 1e9
    for x in np.arange(width):
        for y in np.arange(height):
            map_features_buffer[x][y] = map_features[x][y]
    map_features_buffer = map_features_buffer.reshape(-1)
    map_features_dict = {"map_i_" + str(k): v for k, v in enumerate(map_features_buffer)}

    features = dict()
    features["cargo_vol_total"] = unit_cargo_total(unit)
    features["cargo_fuel_total"] = unit_cargo_fuel_total(unit)
    features["unit_can_build"] = unit.can_build(game_state.map)
    features["unit_routine"] = unit_routine.value if unit_routine is not None else None
    features["unit_last_action"] = get_action_type(last_action)
    features["unit_coords_x"] = unit.pos.x
    features["unit_coords_y"] = unit.pos.y
    assert len(features) == 7, len(features)

    features.update(near_city_features)
    features.update(near_opp_city_features)
    features.update(near_resource_features)
    features.update(common_features)
    features.update(map_features_dict)
    assert len(features) == 32 + 10 + 32 * 32 * 7, len(features)
    FEATURE_CACHE[(unit.id, turn)] = features
    return features


def log_features(unit, game_state, action, turn, player, opponent, width, height, unit_routine, last_action, log_path, env):
    features = calc_features(unit, game_state, turn, player, opponent, width, height, unit_routine, last_action, log_path)

    city_tiles = calc_city_tiles(player)
    opp_city_tiles = calc_city_tiles(opponent)

    row = list(features.values()) + [action, len(city_tiles), len(opp_city_tiles), turn, unit.id]

    path = create_or_get_file(log_path, features.keys(), env)

    with open(path, "a") as f:
        f.write("\t".join(map(str, row)) + "\n")


CONVERTER = ['bcity', 'p', 'n', 's',  'e', 'w']
# CAT_FEATURES = [2, 3, 4, 6, 11, 16, 17, 22, 28, 29, 30, 31]
CAT_FEATURES = [2, 3, 4, 8, 14, 16, 22, 24, 25, 32, 38, 39, 40, 41]
FLOAT_FEATURES = [i for i in np.arange(42 + 32 * 32 * 7) if i not in CAT_FEATURES]
OHE = None


from sklearn.preprocessing import OneHotEncoder


def get_or_read_ohe(ohe_path: str):
    global OHE
    import pickle
    with open(ohe_path, "rb") as f:
        OHE = pickle.load(f)
    return OHE


def prepare_features(features: list, env):
    OHE = get_or_read_ohe(env.get("ohe_path", None))
    features = np.array(features, dtype=object).reshape(1, -1)
    cf = features[:, CAT_FEATURES]
    ff = features[:, FLOAT_FEATURES]
    cf[cf == "False"] = False
    cf[cf == "True"] = True
    cf[cf == None] = "None"
    # cf[cf == "0"] = 0
    cf[cf == "1"] = 1
    cf[cf == "2"] = 2
    cf[cf == "3"] = 3
    ff[ff == "None"] = 0
    # print(cf, file=sys.stderr)
    cf_o = OHE.transform(cf)
    return np.array(np.concatenate([cf_o, ff], axis=1), dtype=np.float)


def apply_model(model_path: str, unit, game_state, ENV) -> str:
    features = FEATURE_CACHE.get((unit.id, TURN), None)
    assert features is not None
    features = np.array(list(map(str,features.values()))).reshape(1, -1)
    if ENV.get("is_neural", False):
        features = prepare_features(features, ENV)
        model = create_or_get_nn_model(model_path)
    else:
        model = create_or_get_cb_model(model_path)
    if ENV.get("use_policy", False):
        probs = model.predict_proba(features)[0]
        classes = CONVERTER
        res = None
        for _ in np.arange(20):
            res = np.random.choice(classes, p=probs)
            if res == 'bcity' and not unit.can_build(game_state.map):
                continue
            if res not in ['bcity', 'p']:
                next_cell = unit.pos.translate(res, 1)
                if next_cell.x < 0 or next_cell.x >= game_state.map.width or next_cell.y < 0 or next_cell.y >= game_state.map.height:
                    continue
            break
        # print(features, probs, res, convert_prediction(res, unit.id), file=sys.stderr)
    else:
        probs = model.predict_proba(features)[0]
        idx = np.argmax(probs)
        res = CONVERTER[idx]
    return convert_prediction(res, unit.id)


def make_random_action(_id: str) -> str:
    idx = np.random.choice(6)
    return convert_prediction(CONVERTER[idx], _id)


def convert_prediction(p: str, _id: str) -> str:
    if p == "c":
        p = "p"
    if p in ["bcity", "p"]:
        return p + " " + _id
    if p in ["n", "e", "w", "s", "c"]:
        return "m " + _id + " " + p
    assert False


def get_action_type(a: str) -> str:
    if a is None:
        return a
    a = a.split()
    if a[0] == "m":
        if a[-1] == "c":
            return "p"
        return a[-1]
    return a[0]
