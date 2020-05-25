from __future__ import annotations

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import json
from pprint import pprint

import math
import numpy as np

from typing import List, Iterator, Callable

import config as cfg


class Highway:
    def __init__(self,
                 name: str = 'None',
                 length: float = 0,
                 hp: float = 0,
                 traffic: float = 0,
                 status: str = 'None',
                 remain: float = math.inf,
                 id: int = None,
                 new_traffic: Callable = None):
        self.name: str = name
        self.id = id
        self.len: float = length  # in km

        self._hp: float = hp
        self.traffic: float = traffic
        self.new_traffic = new_traffic

        self.status: str = status
        self.remain: float = remain

        self.default_status: str = status
        self.default_remain: float = remain

    @staticmethod
    def get_aging(traffic: float):
        return cfg.traffic_to_aging_speed(traffic)

    @staticmethod
    def init_from_cfg(highway: dict) -> Iterator[Highway]:
        hp: float = highway['hp'] if 'hp' in highway else cfg.DEFAULT_HP
        traffic: float = highway['traffic'] if 'traffic' in highway else cfg.DEFAULT_TRAFFIC
        new_traffic: Callable = highway['new_traffic'] if 'new_traffic' in highway else cfg.DEFAULT_NEW_TRAFFIC

        for i in range(highway['total_len'] // cfg.UNIT_LEN_KM):
            yield Highway(name=highway['name'],
                          length=cfg.UNIT_LEN_KM,
                          hp=hp,
                          traffic=traffic,
                          status=cfg.DEFAULT_STATUS,
                          remain=cfg.DEFAULT_REMAIN,
                          id=i,
                          new_traffic=new_traffic)

    def set_status(self,
                   status: str,
                   remain: float = math.inf):
        self.status = status
        self.remain = remain

    @property
    def hp(self):
        return self._hp

    def set_hp(self, new_hp: float):
        if new_hp >= 0:
            self._hp = new_hp
        else:
            self._hp = 0

    def update(self, period:int):
        assert isinstance(period, int)
        while period > 0:
            self.update_one_day()
            period -= 1

    def update_one_day(self):
        # line
        self.traffic = self.new_traffic(self.remain, self.traffic)
        aging = cfg.traffic_to_aging_speed(self.traffic)

        self._hp -= aging
        if self._hp < 0:
            self._hp = 0

        self.remain -= 1
        if self.remain <= 0:
            self.status = self.default_status
            self.remain = self.default_remain


class Action:
    def __init__(self,
                 name,
                 new_hp: Callable,
                 cost,
                 time):
        self.name = name
        self.update_function = new_hp
        self.cost = cost
        self.remain_time = time

    def get_new_hp(self, old_hp):
        return self.update_function(old_hp)


class HMEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # init highways
        self.highways: List[Highway] = []
        for highway_dict in cfg.HIGHWAYS:
            for highway in Highway.init_from_cfg(highway_dict):
                self.highways.append(highway)
        # set TIME
        self.cur_time = 0
        self.time_interval = cfg.TIME_INTERVAL_DAY
        # set ACTIONS
        self.actions = [Action(**d_act) for d_act in cfg.ACTIONS]
        # (action tag, highway tag)
        self.action_space = gym.spaces.MultiDiscrete([len(self.actions), len(self.highways)])
        # set OBSERVATIONS
        self.observation_space = gym.spaces.Box(low=0,
                                                high=cfg.DEFAULT_HP,
                                                shape=(len(self.highways),))
        # set REWARDS
        self.reward_range = (-float('inf'), float('inf'))
        self.reward_gamma = cfg.REWARD_GAMMA
        self.reward_weight_sum = 0
        # LOOP CONTROL
        self.should_stop = False
        self.max_time = cfg.MAX_TIME_DAY

    def step(self, action: np.ndarray):
        """ take a step

        Parameters
        ----------
        action : array
            (action_tag, highway_tag), both are in type int

        Returns
        -------
        result : tuple
            (obs, rewards, dones, info)
        """

        # 1-th: take action
        act_idx, hw_idx = action[0], action[1]  # action name and target
        # update status
        chosen_highway, chosen_action = self.highways[hw_idx], self.actions[act_idx]
        chosen_highway.set_status(chosen_action.name, chosen_action.remain_time)
        chosen_highway.set_hp(chosen_action.get_new_hp(chosen_highway.hp))
        # 2-th update
        for hw in self.highways:
            hw.update(self.time_interval)
        # 3-th calculate the REWARD
        # TODO
        reward = cfg.reward(sum(hw.traffic for hw in self.highways),
                            chosen_action.cost,
                            sum(hw.hp for hw in self.highways))
        # 4-th time goes by
        self.cur_time += self.time_interval
        if self.cur_time > self.max_time:
            self.should_stop = True
        # 5-th RETURN
        obs = [hw.hp for hw in self.highways]
        info = 'action name: {}'.format(chosen_action.name)
        return obs, reward, self.should_stop, info

    def reset(self):
        self.__init__()

    def render(self, mode='human'):
        pass

    def close(self):
        pass
