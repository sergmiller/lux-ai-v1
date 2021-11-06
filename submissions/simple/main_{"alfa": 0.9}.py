from typing import Dict
import sys
import json

try:
    from agent import agent
except:
    from .agent import agent

raw_env = """{"alfa": 0.9}"""

ENV = dict()

if isinstance(raw_env, str):
    ENV = json.loads(raw_env)

if __name__ == "__main__":
    
    def read_input():
        """
        Reads input from stdin
        """
        try:
            return input()
        except EOFError as eof:
            raise SystemExit(eof)
    step = 0
    class Observation(Dict[str, any]):
        def __init__(self, player=0) -> None:
            self.player = player
            # self.updates = []
            # self.step = 0
    observation = Observation()
    observation["updates"] = []
    observation["step"] = 0
    player_id = 0
    while True:
        inputs = read_input()
        observation["updates"].append(inputs)
        
        if step == 0:
            player_id = int(observation["updates"][0])
            observation.player = player_id
        if inputs == "D_DONE":
            actions = agent(observation, ENV)
            observation["updates"] = []
            step += 1
            observation["step"] = step
            print(",".join(actions))
            print("D_FINISH")