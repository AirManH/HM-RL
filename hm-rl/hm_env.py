import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.registration import register

import json
from pprint import pprint

import math
import numpy as np

from typing import List, Iterator, Callable

import config as cfg


class Action:
    def __init__(self,
                 name,
                 new_hp: Callable,
                 new_traffic: Callable,
                 cost,
                 time,
                 stoppable: bool):
        self.name = name
        self.new_hp: Callable = new_hp
        self.cost = cost
        self.remain_time = time
        self.new_traffic: Callable = new_traffic
        self.stoppable: bool = stoppable


class Status:
    def __init__(self,
                 name: str,
                 remain: float,
                 stoppable: bool):
        self.name: str = name
        self.remain: float = remain
        self.stoppable: bool = stoppable

    def __str__(self):
        return str({
            "name": self.name,
            "remain": self.remain,
            "stoppable": self.stoppable
        })

    def __eq__(self, other: 'Status'):
        return self.name == other.name and \
               self.remain == other.remain and \
               self.stoppable == other.stoppable


class Emergency:
    def __init__(self,
                 name: str,
                 new_hp: Callable,
                 frequency: float):
        self.name: str = name
        self.new_hp: Callable = new_hp
        self.frequency: float = frequency


class Highway:
    def __init__(self,
                 name: str = 'None',
                 hp: float = 0,
                 traffic: int = 0,
                 toll: float = 0,
                 status: Status = None,
                 tag: int = None,
                 new_traffic: Callable = None):
        self.name: str = name
        self.tag = tag

        self._hp: float = hp
        # only be assigned when take actions, should be reset to None after action ends
        self.new_hp_to_set: float = None
        self.traffic: int = traffic
        self.new_traffic: Callable = new_traffic
        self.toll = toll

        self.status: Status = status
        self.default_status: Status = status

    def take_action(self,
                    status: Status,
                    new_traffic: Callable,
                    new_hp: Callable):
        self.status = status
        self.new_traffic = new_traffic
        self.new_hp_to_set = new_hp(self._hp)

    def update(self, period: int):
        assert isinstance(period, int)
        while period > 0:
            self.update_one_day()
            period -= 1

    def update_one_day(self):
        # line
        self.traffic = self.new_traffic(self.status.remain, self.traffic)
        # TODO ugly design here
        # if is being repaired, freeze hp
        if self.status.name == self.default_status.name:
            aging = cfg.traffic_to_aging_speed(self.traffic)
            self.set_hp(self._hp - aging)

        self.status.remain -= 1
        if self.status.remain <= 0:
            # this means the action finally ends
            # we should set new hp
            self.status = self.default_status
            self.set_hp(self.new_hp_to_set)
            self.new_hp_to_set = None

    @staticmethod
    def get_aging(traffic: float):
        return cfg.traffic_to_aging_speed(traffic)

    @staticmethod
    def init_from_cfg(highway: dict) -> 'Highway':
        hp: float = highway['hp'] if 'hp' in highway else cfg.DEFAULT_HP
        traffic: int = highway['traffic'] if 'traffic' in highway else cfg.DEFAULT_TRAFFIC
        new_traffic: Callable = highway['new_traffic'] if 'new_traffic' in highway else cfg.DEFAULT_NEW_TRAFFIC
        return Highway(name=highway['name'],
                       hp=hp,
                       traffic=traffic,
                       toll=highway['toll'],
                       status=Status(**cfg.DEFAULT_STATUS),
                       tag=0,
                       new_traffic=new_traffic)

    def __str__(self):
        d = {
            "name": self.name,
            "tag": self.tag,
            "hp": self._hp,
            "traffic": self.traffic,
            "status": str(self.status)
        }
        return str(d)

    @property
    def hp(self):
        return self._hp

    def set_hp(self, new_hp: float):
        if new_hp < 0:
            self._hp = 0
        elif new_hp > cfg.DEFAULT_HP:
            self._hp = cfg.DEFAULT_HP
        else:
            self._hp = new_hp


class HMEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # init highways
        self.highways: List[Highway] = [Highway.init_from_cfg(d) for d in cfg.HIGHWAYS]
        # set TIME
        self.cur_time = 0
        self.time_interval = cfg.TIME_INTERVAL_DAY
        # set ACTIONS
        self.actions = [Action(**d_act) for d_act in cfg.ACTIONS]
        # (action tag, highway tag)
        self.action_space = gym.spaces.MultiDiscrete([len(self.actions)] * len(self.highways))
        # set OBSERVATIONS
        """ observation_space
        [[       hp_1,        hp_2, ...,        hp_n],
         [stoppable_1, stoppable_2, ..., stoppable_n]]
        """
        self.observation_space = gym.spaces.Box(0, cfg.DEFAULT_HP, (2, len(self.highways)))
        # set REWARDS
        self.reward_range = (-float('inf'), float('inf'))
        self.reward_gamma = cfg.REWARD_GAMMA
        self.reward_ewma = 0
        # LOOP CONTROL
        self.should_stop = False
        self.max_time = cfg.MAX_TIME_DAY

    def step(self, action: np.ndarray):
        """ take a step

        Parameters
        ----------
        action : array

        Returns
        -------
        result : tuple
            (obs, rewards, done, info)
        """

        # 1-th: Take Action, Calculate reward
        rewards = []
        for hw_idx, act_idx in enumerate(action):
            chosen_highway, chosen_action = self.highways[hw_idx], self.actions[act_idx]
            if chosen_highway.status.stoppable:
                # update status
                # TODO ugly design here
                new_status = Status(chosen_action.name, chosen_action.remain_time, chosen_action.stoppable)
                chosen_highway.take_action(new_status, chosen_action.new_traffic, chosen_action.new_hp)
            # reward
            rewards.append(
                cfg.reward(chosen_highway.traffic, chosen_highway.toll, chosen_highway.hp, chosen_action.cost))

        # 2-th update
        for hw in self.highways:
            hw.update(self.time_interval)
        # 3-th time goes by
        self.cur_time += self.time_interval
        if self.cur_time > self.max_time:
            self.should_stop = True
        # 4-th RETURN
        obs = np.array([[hw.hp for hw in self.highways],
                        [int(hw.status.stoppable) for hw in self.highways]])
        info: dict = {}
        return obs, np.average(rewards), self.should_stop, info

    def reset(self):
        self.__init__()
        obs = np.array([[hw.hp for hw in self.highways],
                        [int(hw.status.stoppable) for hw in self.highways]])
        return obs

    def render(self, mode='human'):
        pass

    def close(self):
        pass


register(id='PccNs-v0', entry_point='network_sim:SimulatedNetworkEnv')
