import gym
import numpy as np
import os
from gym import spaces
import pandas as pd
from gym.utils import seeding
from arguments import parse_args
args = parse_args()

REWARD_SCALING = args.reward_shorten
from config.name_fee import name_fee

class TestTradingEnv(gym.Env):

    def __init__(self, df, obs_dim=23, day=0, episode_window=480, last_episode_action=0):
        super(TestTradingEnv, self).__init__()
        self.day = day
        self.df = df
        self.obs_dim = obs_dim
        self.observation_space = spaces.Box(low=-5, high=5, shape=(self.obs_dim,))
        self.action_space = spaces.Discrete(3)
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        self.episode_window = episode_window
        self.state = self.data.historySpreadNormalized + [0] + [self.data.MACDNormalized] + [self.data.RSINormalized]

        self.action_memory = []
        self.profit_memory = []
        self.date_memory = []
        self.seed()
        self.log_step = 0
        self.last_episode_action = last_episode_action

    def reset(self):
        self.log_step = self.log_step + 1
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.state = self.data.historySpreadNormalized + [0] + [self.data.MACDNormalized] + [self.data.RSINormalized]
        return self.state

    def reset_task(self, task):
        pass

    def step(self, action):
        action -= 1
        self.reward = 0
        self.action_memory.append(action)
        trade_action = action


        pre_action = self.last_episode_action if self.day == 0 else self.action_memory[self.day - 1]

        lastPrice = self.data.close
        begin_total_asset = lastPrice * trade_action

        self.day += 1
        self.data = self.df.loc[self.day, :]
        end_total_asset = self.data.close * trade_action



        profit = (end_total_asset - begin_total_asset) - \
                 (args.fee_time) * name_fee[args.name] * abs(trade_action - pre_action) * lastPrice

        self.profit_memory.append(profit)


        self.state = self.data.historySpreadNormalized + [self.action_memory[self.day - 1]] + \
                     [self.data.MACDNormalized] + [self.data.RSINormalized]
        self.date_memory.append(self.data.datadate)


        self.terminal = (self.day >= (self.episode_window-1))










        self.reward = profit
        self.reward = self.reward * REWARD_SCALING
        return self.state, self.reward, self.terminal, {}










