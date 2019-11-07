'''
This script defines training environment of stock price model in OpenAI gym. 
'''
import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces

import matplotlib.pyplot as plt

'''
Data Preprocessing
'''
data_1 = pd.read_csv('/Users/Yitao/Anaconda3/envs/myenv/Lib/site-packages/gym/envs/stock/Data_Daily_Stock_Dow_Jones_30/fulldata.csv')

# filter out stocks with less than 2476 data points
equal_2476_list = list(data_1.tic.value_counts() == 2476)
names = data_1.tic.value_counts().index
select_stocks_list = list(names[equal_2476_list])

data_2 = data_1[data_1.tic.isin(select_stocks_list)][~data_1.datadate.isin(['20010912','20010913'])]
data_3 = data_2[['iid','datadate','tic','prccd','ajexdi']]

data_3['adjcp'] = data_3['prccd'] / data_3['ajexdi']

train_data = data_3[(data_3.datadate > 20100000) & (data_3.datadate < 20160000)]
train_daily_data = []

for date in np.unique(train_data.datadate):
    train_daily_data.append(train_data[train_data.datadate == date])

# global constant
iteration = 0
select_stocks_size = len(select_stocks_list)
terminal_day = len(train_daily_data) - 1
max_share = 5
observation_size = select_stocks_size * 2 + 1
initial_fund = 10000

'''
Environment definition
'''
class StockEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, day = 0, money = 10, scope = 1):
        self.day = day
        self.size = len(train_daily_data)
        # buy or sell maximum 5 shares
        self.action_space = spaces.Box(low = -max_share, high = max_share,shape = (select_stocks_size,),dtype=np.int8) 

        # [money]+[prices 1-28]+[owned shares 1-28]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape = (observation_size,))

        self.data = train_daily_data[self.day]
        
        self.terminal = False
        
        self.state = [initial_fund] + self.data.adjcp.values.tolist() + [0 for i in range(select_stocks_size)]
        self.reward = 0
        
        self.asset_memory = [initial_fund]

        self.reset()
        self._seed()


    def _sell_stock(self, index, action):
        if self.state[index+29] > 0:
            self.state[0] += self.state[index+1]*min(abs(action), self.state[index+29])
            self.state[index+29] -= min(abs(action), self.state[index+29])
        else:
            pass
    
    def _buy_stock(self, index, action):
        available_amount = self.state[0] // self.state[index+1]
        self.state[0] -= self.state[index+1]*min(available_amount, action)
        self.state[index+29] += min(available_amount, action)
        
    def step(self, actions):
        self.terminal = self.day >= terminal_day

        if self.terminal:
            np.savetxt("/Users/Yitao/Documents/baseline.csv", np.array(self.asset_memory), delimiter=",")

            plt.plot(self.asset_memory,'r')
            plt.savefig('/Users/Yitao/Documents/iteration_{}.png'.format(iteration))
            plt.close()
            print("total_reward:{}".format(self.state[0] + sum(np.array(self.state[1:29])*np.array(self.state[29:]))- initial_fund))
            return self.state, self.reward, self.terminal,{}

        else:
            # print(np.array(self.state[1:29]))
            begin_total_asset = self.state[0]+ sum(np.array(self.state[1:29])*np.array(self.state[29:]))
            # print("begin_total_asset:{}".format(begin_total_asset))
            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                # print('take sell action'.format(actions[index]))
                self._sell_stock(index, actions[index])

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                self._buy_stock(index, actions[index])

            self.day += 1
            self.data = train_daily_data[self.day]         


            # print("stock_shares:{}".format(self.state[29:]))
            self.state =  [self.state[0]] + self.data.adjcp.values.tolist() + list(self.state[29:])
            end_total_asset = self.state[0]+ sum(np.array(self.state[1:29])*np.array(self.state[29:]))
            # print("end_total_asset:{}".format(end_total_asset))
            
            self.reward = end_total_asset - begin_total_asset            
            # print("step_reward:{}".format(self.reward))

            self.asset_memory.append(end_total_asset)

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.asset_memory = [initial_fund]
        self.day = 0
        self.data = train_daily_data[self.day]
        self.state = [initial_fund] + self.data.adjcp.values.tolist() + [0 for i in range(select_stocks_size)]
        
        return self.state
    
    def render(self, mode='human'):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]