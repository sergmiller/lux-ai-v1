import math, sys
from lux.game import Game
from lux.game_map import Cell, RESOURCE_TYPES
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
from lux import annotate

DIRECTIONS = Constants.DIRECTIONS
game_state = None



def worker_try_go_to_near_resource(unit, player, resource_tiles: list) -> str:
    """
    :param unit: represents worker unit
    :param player: represents player info
    :param resource_tiles: coordinates of resources
    :return: action or None
    """
    closest_dist = math.inf
    closest_resource_tile = None
    action = None
    for resource_tile in resource_tiles:
        if resource_tile.resource.type == Constants.RESOURCE_TYPES.COAL and not player.researched_coal(): continue
        if resource_tile.resource.type == Constants.RESOURCE_TYPES.URANIUM and not player.researched_uranium(): continue
        dist = resource_tile.pos.distance_to(unit.pos)
        if dist < closest_dist:
            closest_dist = dist
            closest_resource_tile = resource_tile
    if closest_resource_tile is not None:
        action = unit.move(unit.pos.direction_to(closest_resource_tile.pos))
    return action

def worker_try_return_to_city(unit, player) -> str:
    """
    :param unit: represents worker unit
    :param player: represents player info
    :return: action or None
    """
    action = None
    if len(player.cities) > 0:
        closest_dist = math.inf
        closest_city_tile = None
        for k, city in player.cities.items():
            for city_tile in city.citytiles:
                dist = city_tile.pos.distance_to(unit.pos)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_city_tile = city_tile
        if closest_city_tile is not None:
            move_dir = unit.pos.direction_to(closest_city_tile.pos)
            action = unit.move(move_dir)
    return action


def append_if_not_none(actions: list, action: str):
    if action is not None:
        actions.append(action)


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


    # we iterate over all our units and do something with them
    for unit in player.units:
        if unit.is_worker() and unit.can_act():
            if unit.get_cargo_space_left() > 0:
                # if the unit is a worker and we have space in cargo,
                # lets find the nearest resource tile and try to mine it
                append_if_not_none(actions, worker_try_go_to_near_resource(unit, player, resource_tiles))
            else:
                # if unit is a worker and there is no cargo space left, and we have cities, lets return to them
                append_if_not_none(actions, worker_try_return_to_city(unit, player))

    for city in player.cities.values():
        for city_tile in city.citytiles:
            if city_tile.can_act():
                actions.append(city_tile.research())

    # you can add debug annotations using the functions in the annotate object
    # actions.append(annotate.circle(0, 0))
    
    return actions
