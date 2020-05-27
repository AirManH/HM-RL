from hm_env import HMEnv
from pprint import pprint

from stable_baselines.common.policies import MlpPolicy, CnnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

import datetime

import matplotlib.pyplot as plt

import config as cfg

from tqdm import tqdm

def print_env(e: HMEnv):
    for hw in e.highways:
        print(hw)
    print('=' * 10)


def train(total_timesteps):
    env = HMEnv()
    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    # save
    cur_time = datetime.datetime.now()
    filename = "{}{}{}-{}{}{}-step_{}".format(cur_time.year, cur_time.month, cur_time.day,
                                              cur_time.hour, cur_time.minute, cur_time.second,
                                              total_timesteps)
    model.save('../models/{}'.format(filename))


def test(path: str):
    model = PPO2.load(path)

    data = [[[], [], cfg.HIGHWAYS[i]['name']] for i in range(len(cfg.HIGHWAYS))]
    max_time = 3 * cfg.DAY_PER_YEAR
    record_interval = 1

    env = HMEnv()
    obs = env.reset()
    for step in tqdm(range(max_time)):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        # record
        if step % record_interval == 0:
            for i in range(len(cfg.HIGHWAYS)):
                data[i][0].append(step)
                data[i][1].append(env.highways[i].hp)
    # plot
    for single_data in data:
        x, y, note = single_data
        plt.plot(x, y, label=note)
    plt.legend()
    plt.show()

def foo():
    env = HMEnv()
    act = [3] * len(cfg.HIGHWAYS)
    for i in range(10):
        obs, reward, done, info = env.step(act)
        print(reward)


if __name__ == "__main__":
    # train(100000)
