import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def utility(x, gamma):
    return - np.exp(-gamma * x)

def calculate_reward(pnl_new, pnl_last, q_new, S, flag, gamma, util_flag=True):
    if flag:
        # if we end, we need to penalty the maker for having large inventory
        if util_flag:
            return -np.exp(-gamma * (pnl_new - q_new * S))
        return -np.abs(q_new) * S
    else:
        if util_flag:
            U_new = utility(pnl_new, gamma)
            U_old = utility(pnl_last, gamma)
            return U_new - U_old
        return pnl_new - pnl_last

class Environment():
    #
    # our actions are: i * spread
    # might be: short, long, nothing
    #
    
    def __init__(self, data_maker, T, A, k, K, cur, gamma, util_flag=True, mid_flag=False):
        '''
        Make and environment for market interaction.
        
        Parameters:
        data_maker (DataMaker): Data emulator.
        T (int): Number of ticks to the end of the trading day.
        cur (int): The start moment.
        gamma (float): Risk aversion coefficient.
        mid_price (bool): True-price flag. If true - determined by middle price, else by weighted.
        util_flag (bool): Reward policy flag. If true - determined by utility policy, else by P&L.
        '''
        self.observation_space = np.array([0., 0.])
        self.n_actions = K
        
        self.data_maker = data_maker
        self.cur = cur # current pos
        self.T = T #  number of ticks to end 
        self.gamma = gamma # risk aversion
        self.A = A # A param for Poisson
        self.k = k # k param for Poisson
        self.K = K # Number of actions supported
        
        self.short_positions = [] # all short prices
        self.long_positions = [] # all long prices
        
        self.q = np.zeros(T + 2) # current inventory
        self.x = np.zeros(T + 2) # current wealth
        self.pnl = np.zeros(T + 2) # current P&L
        
        self.actions = [] # actions list
        
        self.util_flag = util_flag
        self.mid_flag = mid_flag
    
    def step(self, action):
        
        i = self.cur
        pnl_mem = 0.
        
        
        # _________________________________________________ #
        
        v_a, v_b, p_a, p_b, flag = self.data_maker.get()
        
        add_for_nothing = 0
        
        S = (p_a + p_b) / 2.# mid price to get current value of portfollio

        if self.mid_flag:
            w = v_a * 1./ (v_a + v_b)
            S = w * p_a + (1 - w) * p_b
            
        action_ask = np.floor_divide(action, self.K) + 1
        action_bid = np.mod(action, self.K) + 1
        
        self.actions.append((action_ask, action_bid))  # save for future
        
        spread_tick = np.abs(p_a - p_b) * 2.
        
        delta_ask = spread_tick * action_ask
        delta_bid = spread_tick * action_bid
        
        r_ask = S + spread_tick * action_ask
        r_bid = S - spread_tick * action_bid
        
        lambda_ask = self.A * np.exp(-self.k * delta_ask)
        lambda_bid = self.A * np.exp(-self.k * delta_bid)

        # _________________________________________________ #
        
        add_ask, add_bid = 0, 0

        dt = 1.0 / self.T
        
        if np.random.rand() < lambda_ask * dt:
            add_ask = 1
        if np.random.rand() < lambda_ask * dt:
            add_bid = 1
            
        self.q[i + 1] = self.q[i] - add_ask + add_bid
        self.x[i + 1] = self.x[i] + r_ask * add_ask - r_bid * add_bid
        self.pnl[i + 1] = self.x[i + 1] + self.q[i + 1] * S
        
        reward = calculate_reward(self.pnl[i + 1], self.pnl[i], self.q[i + 1], S,
                                  flag, self.gamma, self.util_flag)

        # _________________________________________________ #
        
        # check that there is no current data
        if self.data_maker.is_empty():
            flag = True
            
        is_done = flag
        
        info = 'Done' if is_done else 'Continue'
        
        self.cur += 1
                
        return (self.q[i + 1], self.cur / (self.T + 1)), reward, is_done, info
    
    def reset(self):
        
        self.data_maker.step()
        self.cur = 0

        self.long_positions = []
        self.short_positions = []
        self.pnl_hist = []
        self.q_hist = []
        self.actions = []
        
        self.q = np.zeros(self.T + 2)
        self.pnl = np.zeros(self.T + 2)
        self.x = np.zeros(self.T + 2)
        
        self.util_flag = self.util_flag
        self.mid_flag = self.mid_flag
        
        return (self.q[0], self.cur / (self.T + 1))

    def render(self, mode='human', close=False):
        pass

    def take_action(self, action):
        pass
