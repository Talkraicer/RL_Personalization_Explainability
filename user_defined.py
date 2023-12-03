from highway_env.utils import lmap

action_to_num = {"LANE_LEFT": 0, "IDLE": 1, "LANE_RIGHT": 2, "FASTER": 3, "SLOWER": 4}


def policy_fast(state):
    return action_to_num["FASTER"] if state["speed"] < 29 else 5


def policy_slow(state):
    return action_to_num["SLOWER"] if state["speed"] > 21 else 5


def scale_speed(speed):
    return lmap(speed, [20, 30], [0, 1])


def value_fast(state, action):
    speed = state["speed"]
    return scale_speed(speed)


def value_slow(state, action):
    speed = state["speed"]
    return 1 - scale_speed(speed)


def value_fast_left(state, action):
    speed = state["speed"]
    lane = state["lane"]
    return scale_speed(speed) - lane / 3


def value_slow_right(state, action):
    speed = state["speed"]
    lane = state["lane"]
    return 1 - scale_speed(speed) + lane / 3
