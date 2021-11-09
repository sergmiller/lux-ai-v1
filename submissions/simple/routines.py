from enum import Enum
import numpy as np

import sys


ENV = dict()

class Routine(Enum):
    FREE = 0
    GO_NEAREST_CITY = 1
    GO_NEAREST_RESOURCE = 2
    GO_BUILD_CITY = 3


ROUTINES = dict()

ROUTINE_RULES = {
    Routine.FREE: [[Routine.GO_NEAREST_RESOURCE], [1.0]],
    Routine.GO_NEAREST_CITY: [[Routine.GO_NEAREST_RESOURCE], [1.0]],
    Routine.GO_NEAREST_RESOURCE: [[Routine.GO_NEAREST_CITY, Routine.GO_BUILD_CITY], [0.5, 0.5]],
    Routine.GO_BUILD_CITY: [[Routine.GO_NEAREST_RESOURCE], [1.0]]
}

ROUTINE_EXIT_CONDITIONS = {
    Routine.FREE: lambda _, __: True,
    Routine.GO_NEAREST_CITY: lambda unit, city_tiles: True if len(city_tiles) == 0 else max(unit.pos == city.pos for city in city_tiles) > 0,
    Routine.GO_NEAREST_RESOURCE: lambda unit, _: unit_cargo_total(unit) > ENV.get("stop_mining_threshold", 79),
    Routine.GO_BUILD_CITY: lambda unit, __: unit_cargo_total(unit) < 100 if ENV.get("use_old_units_cargo_rules", True) else unit_cargo_fuel_total(unit) < 100
}


def unit_cargo_total(unit):
    return unit.cargo.wood + unit.cargo.coal + unit.cargo.uranium


def unit_cargo_fuel_total(unit):
    return unit.cargo.wood + unit.cargo.coal * 10 + unit.cargo.uranium * 40


def perform_routine(env, turn, unit, city_tiles, routine_actions: dict):
    global ROUTINES, ENV
    ENV = env
    state = update_state(unit, city_tiles, turn)
    action = routine_actions[state]()
    if env.get('debug', False):
        print("DEBUG:", turn, unit.id, state, action, unit.pos, unit.cargo, ROUTINES, file=sys.stderr)
    return action


def update_state(unit, city_tiles, turn):
    global ROUTINES
    state = ROUTINES.get(unit.id, Routine.FREE)
    if ROUTINE_EXIT_CONDITIONS[state](unit, city_tiles):
        possibilities, weights = ROUTINE_RULES[state]
        if state == Routine.GO_NEAREST_RESOURCE:
            weights = ENV.get("go_resource_next_action_probs", [0.1, 0.9])
        weights = np.array(weights)
        if ENV.get("norm_probs_to_city_tiles", False):
            weights = weights ** (1 / (1 + len(city_tiles)) ** 0.5)
            weights /= np.sum(weights)
        ROUTINES[unit.id] = np.random.choice(possibilities, 1, p=weights)[0]
        if state == Routine.GO_NEAREST_CITY and len(city_tiles) == 0:
            ROUTINES[unit.id] = Routine.GO_BUILD_CITY
        if ENV.get("skip_mine_and_build_loop", False):
            if state == Routine.GO_NEAREST_RESOURCE and ROUTINE_EXIT_CONDITIONS[Routine.GO_BUILD_CITY](unit, city_tiles):
                ROUTINES[unit.id] = Routine.GO_NEAREST_CITY
    if ENV.get("go_to_city_at_night", True):
        if turn % 40 >= 30:
            ROUTINES[unit.id] = Routine.GO_NEAREST_CITY
    return ROUTINES[unit.id]
