import math, sys
from lux.game import Game
from lux.game_map import Cell, RESOURCE_TYPES
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
from lux import annotate
from routines import perform_routine, Routine

import numpy as np

DIRECTIONS = Constants.DIRECTIONS
game_state = None
future_workers_pos = dict()


def worker_try_go_to_near_resource(unit, player, resource_tiles: list) -> str:
    global future_workers_pos
    """
    :param unit: represents worker unit
    :param player: represents player info
    :param resource_tiles: coordinates of resources
    :return: action or None
    """
    closest_dist = math.inf
    closest_resource_tile = None
    action = None
    for resource_tile in np.random.permutation(resource_tiles):
        if resource_tile.resource.type == Constants.RESOURCE_TYPES.COAL and not player.researched_coal(): continue
        if resource_tile.resource.type == Constants.RESOURCE_TYPES.URANIUM and not player.researched_uranium(): continue
        dist = resource_tile.pos.distance_to(unit.pos)
        if dist < closest_dist:
            closest_dist = dist
            closest_resource_tile = resource_tile
    if closest_resource_tile is not None:
        move_dir = unit.pos.direction_to(closest_resource_tile.pos)
        tile = unit.pos.translate(move_dir, 1)
        if not worker_go_in_tile(player, tile):
            action = unit.move(move_dir)
            future_workers_pos[unit.id] = tile
    return action

def worker_try_return_to_city(unit, player) -> str:
    global future_workers_pos
    """
    :param unit: represents worker unit
    :param player: represents player info
    :return: action or None
    """
    action = None
    if len(player.cities) > 0:
        closest_dist = math.inf
        closest_city_tile = None
        cities = player.cities.items()
        for k, city in np.random.permutation(list(cities)):
            for city_tile in np.random.permutation(city.citytiles):
                # dist = city_tile.pos.distance_to(unit.pos)
                alfa = 10
                dist = player.cities[city_tile.cityid].fuel + alfa * city_tile.pos.distance_to(unit.pos)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_city_tile = city_tile
        if closest_city_tile is not None:
            move_dir = unit.pos.direction_to(closest_city_tile.pos)
            action = unit.move(move_dir)
            future_workers_pos[unit.id] = unit.pos.translate(move_dir, 1)
    return action

def worker_try_to_build_a_city(unit) -> str:
    global future_workers_pos
    """
    :param unit: represents worker unit
    :return: action or None
    """
    action = None
    # assert False
    assert game_state is not None, "Please check if global_state inited"
    if unit.can_build(game_state.map):
        return unit.build_city()
    # assert False
    closest_dist = math.inf
    closest_empty_tile = None
    width, height = game_state.map.width, game_state.map.height

    tiles = set()
    for player in [0, 1]:
        p = game_state.players[player]
        for city in p.cities.values():
            for city_tile in city.citytiles:
                tiles.add((city_tile.pos.x, city_tile.pos.y))

    for y in np.random.permutation(height):
        for x in np.random.permutation(width):
            cell = game_state.map.get_cell(x, y)
            if not cell.has_resource() and (cell.pos.x, cell.pos.y) not in tiles:
                dist = cell.pos.distance_to(unit.pos)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_empty_tile = cell

    assert closest_empty_tile is not None
    move_dir = unit.pos.direction_to(closest_empty_tile.pos)
    closest_dist = math.inf
    for direction in np.random.permutation(Constants.ALL_DIRECTIONS):
        if direction == DIRECTIONS.CENTER:
            continue
        next_pos = unit.pos.translate(direction, 1)
        if (next_pos.x, next_pos.y) not in tiles and 0 <= next_pos.y < game_state.map.height and 0 <= next_pos.x < game_state.map.width:
            dist = closest_empty_tile.pos.distance_to(next_pos)
            if dist < closest_dist:
                closest_dist = dist
                move_dir = direction
    action = unit.move(move_dir)
    future_workers_pos[unit.id] = unit.pos.translate(move_dir, 1)

    return action


def append_if_not_none(actions: list, action: str):
    if action is not None:
        actions.append(action)


def worker_go_in_tile(player, tile) -> bool:
    for unit in player.units:
        if unit.is_worker():
            pos = future_workers_pos.get(unit.id, None)
            if pos is not None and pos == tile:
                return True
    return False


def agent(observation, _):
    global game_state

    ### Do not edit ###
    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation.player
    else:
        game_state._update(observation["updates"])
    
    actions = []

    ### AI Code goes down here! ### 
    player = game_state.players[observation.player]
    opponent = game_state.players[(observation.player + 1) % 2]
    width, height = game_state.map.width, game_state.map.height

    resource_tiles: list = []
    for y in range(height):
        for x in range(width):
            cell = game_state.map.get_cell(x, y)
            if cell.has_resource():
                resource_tiles.append(cell)

    city_tiles: list = []
    for city in player.cities.values():
        for city_tile in city.citytiles:
            city_tiles.append(city_tile)

    # we iterate over all our units and do something with them
    for unit in player.units:
        if unit.is_worker() and unit.can_act():
            # if unit.get_cargo_space_left() > 0:
            #     # if the unit is a worker and we have space in cargo,
            #     # lets find the nearest resource tile and try to mine it
            #     append_if_not_none(actions, worker_try_go_to_near_resource(unit, player, resource_tiles))
            # else:
            #     # if unit is a worker and there is no cargo space left, and we have cities, lets return to them
            #     append_if_not_none(actions, worker_try_return_to_city(unit, player))

            action = perform_routine(unit, city_tiles, {
                Routine.GO_NEAREST_RESOURCE: lambda: worker_try_go_to_near_resource(unit, player, resource_tiles),
                Routine.GO_NEAREST_CITY: lambda: worker_try_return_to_city(unit, player),
                Routine.GO_BUILD_CITY: lambda: worker_try_to_build_a_city(unit)
            })
            if action is None:
                action = unit.pillage()
            actions.append(action)

    num_workers = sum(unit.is_worker() for unit in player.units)
    for city in player.cities.values():
        for city_tile in city.citytiles:
            if city_tile.can_act():
                if num_workers < len(city_tiles) and not worker_go_in_tile(player, city_tile.pos):
                    actions.append(city_tile.build_worker())
                    num_workers += 1
                else:
                    actions.append(city_tile.research())

    # you can add debug annotations using the functions in the annotate object
    # actions.append(annotate.circle(0, 0))
    
    return actions
