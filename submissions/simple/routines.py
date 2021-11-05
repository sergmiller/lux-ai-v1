from enum import Enum
import numpy as np

import sys


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
    Routine.GO_NEAREST_RESOURCE: lambda unit, _: unit.get_cargo_space_left() == 0,
    Routine.GO_BUILD_CITY: lambda unit, __: unit.get_cargo_space_left() > 0
}


def perform_routine(unit, city_tiles, routine_actions: dict):
    global ROUTINES
    state = update_state(unit, city_tiles)
    action = routine_actions[state]()
    # print("DEBUG:", unit, state, action, unit.pos, unit.cargo, file=sys.stderr)
    return action


TURN = 0


def update_state(unit, city_tiles):
    global ROUTINES, TURN
    TURN += 1
    # if TURN  == 10:
    #     assert False
    state = ROUTINES.get(unit.id, Routine.FREE)
    # assert False
    # if state == Routine.GO_BUILD_CITY:
    #     assert False
    # if state == Routine.GO_NEAREST_RESOURCE:
    #     assert False
    # assert False
    if ROUTINE_EXIT_CONDITIONS[state](unit, city_tiles):
        possibilities, weights = ROUTINE_RULES[state]
        ROUTINES[unit.id] = np.random.choice(possibilities, 1, p=weights)[0]
        if state == Routine.GO_NEAREST_CITY and len(city_tiles) == 0:
            ROUTINES[unit.id] = Routine.GO_BUILD_CITY
        # print("DEBUG2:", ROUTINES[unit.id], possibilities, file=sys.stderr)
    return ROUTINES[unit.id]
