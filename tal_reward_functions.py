import math

import gym
import numpy as np
from highway_env import utils

from highway_env.envs import HighwayEnv, Action
from gym.envs.registration import register

from highway_env.utils import lmap
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split

class Truck(HighwayEnv):
    """rewarded for driving in the right lane, penalized for driving in other lanes
        also penalized for changing lanes, rewarded just a little for speed"""

    def __init__(self):
        super().__init__()
        self.config["keep_distance_reward"] = 1
        self.config["lane_change_reward"] = 2

    def _reward(self, action: Action) -> float:
        obs = self.observation_type.observe()
        other_cars = obs[1:]
        # punish for changing lanes
        lane_change_left = -1 if action == 0 else 0
        lane_change_right = 1 if action == 2 else 0

        scaled_speed = lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])

        # safety distance from car in same lane
        dist_closest_car_in_lane = [x[1] for x in other_cars if x[1] > 0 and abs(x[2]) <= 0.05]
        if not dist_closest_car_in_lane or dist_closest_car_in_lane[0] > 0.01:
            keeping_distance = 1
        else:
            keeping_distance = -1

        reward = \
            + self.config["collision_reward"] * self.vehicle.crashed \
            + self.config["keep_distance_reward"] * keeping_distance \
            + self.config["lane_change_reward"] * lane_change_left \
            + self.config["lane_change_reward"] * lane_change_right \
            + self.config["high_speed_reward"] * scaled_speed
        reward = -10 if not self.vehicle.on_road else reward
        return reward


register("Truck-v0",
         entry_point=__name__ + ':Truck')


class FastLeft(HighwayEnv):
    """ rewarded for driving fast and left, never leaves left lane"""

    def __init__(self):
        super().__init__()
        self.config["lane_change_reward"] = 2
        self.config["high_speed_reward"] = 2
        self.config["collision_reward"] = -20
        self.config["keep_distance_reward"] = 1
        self.config["gas_reward"] = 0

    def _reward(self, action: Action) -> float:
        """
        Prioritize fast and left
        """
        obs = self.observation_type.observe()
        other_cars = obs[1:]

        # find my lane
        my_lane = self.vehicle.lane_index[2]

        scaled_speed = lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])

        # safety distance from car in same lane
        dist_closest_car_in_lane = [x[1] for x in other_cars if x[1] > 0 and abs(x[2]) <= 0.05]
        if not dist_closest_car_in_lane or dist_closest_car_in_lane[0] > 0.01:
            keeping_distance = 1
        else:
            keeping_distance = -1

        is_gas = 1 if action == 3 else 0
        reward = \
            + self.config["collision_reward"] * self.vehicle.crashed \
            - self.config["lane_change_reward"] * abs(my_lane) \
            + self.config["high_speed_reward"] * scaled_speed \
            + self.config["keep_distance_reward"] * keeping_distance  \
            + self.config["gas_reward"] * is_gas
        reward = -100 if not self.vehicle.on_road else reward
        return reward



register(
    id='FastLeft-v0',
    entry_point=__name__+':FastLeft',
)

class Fast(HighwayEnv):
    """ rewarded for driving fast and left, never leaves left lane"""

    # def _create_road(self) -> None:
    #     """Create a road composed of straight adjacent lanes."""
    #     self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=25),
    #                      np_random=self.np_random, record_history=self.config["show_trajectories"])

    def __init__(self):
        super().__init__()
        self.config["high_speed_reward"] = 2
        self.config["collision_reward"] = -20
        self.config["keep_distance_reward"] = 1
        self.config["gas_reward"] = 0

    def _reward(self, action: Action) -> float:
        """
        Prioritize fast
        """
        obs = self.observation_type.observe()
        other_cars = obs[1:]

        scaled_speed = lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])

        dist_closest_car_in_lane = [x[1] for x in other_cars if x[1] > 0 and abs(x[2]) <= 0.05]
        if not dist_closest_car_in_lane or dist_closest_car_in_lane[0] > 0.01:
            keeping_distance = 1
        else:
            keeping_distance = -1

        is_gas = 1 if action == 3 else 0

        reward = \
            + self.config["collision_reward"] * self.vehicle.crashed \
            + self.config["high_speed_reward"] * scaled_speed \
            + self.config["keep_distance_reward"] * keeping_distance \
            + self.config["gas_reward"] * is_gas
        reward = -100 if not self.vehicle.on_road else reward
        return reward


register(
    id='Fast-v0',
    entry_point=__name__+':Fast',
)

class Saver(HighwayEnv):
    """ rewarded for driving fast and left, never leaves left lane"""

    def _reward(self, action: Action) -> float:

        # punish if break
        break_reward = -1 if action == 4 else 0

        scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
        reward = \
            + self.config["collision_reward"] * self.vehicle.crashed \
            + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)
        reward = utils.lmap(reward,
                          [self.config["collision_reward"],
                           self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                          [0, 1])
        reward += break_reward
        reward = -2 if not self.vehicle.on_road else reward
        return reward


register(
    id='Saver-v0',
    entry_point=__name__+':Saver',
)
