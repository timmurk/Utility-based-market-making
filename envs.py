import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def utility(S, q, gamma, x_0 = 0):
    return - np.exp(-gamma * (x_0 + q * S))

def calculate_reward(q_new, q_old, pnl_new, pnl_last,
                     S, S_old, flag, gamma):
    if flag:
        # if we end, we need to penalty the maker for having large inventory
        print(pnl_new, q_new, q_old, S)
        return -np.exp(-gamma * (pnl_new - q_new * S))
    else:
#        V_1 = -np.exp(-gamma * (pnl_new + q_new * S)) + np.exp(-gamma * (pnl_last + q_new * S)) - 0.1 * np.sign(np.abs(q_new) - #np.abs(q_old))
#        return V_1
        return pnl_new - pnl_last

class Environment():
    #
    # our actions are: buy, sell, hold
    # might be: short, long, nothing
    #
    
    def __init__(self, data_maker, T, cur, gamma, mid_price=True, all_close=True):
        self.observation_space = np.array([0.])
        self.n_actions = 3
        
        self.data_maker = data_maker
        self.positions = []
        self.cur = cur
        self.gamma = gamma
        self.short_positions = []
        self.long_positions = []
        
        self.q = 0
        self.q_hist = [0]
        
        self.S_prev = 0.
        self.all_close = all_close
        self.prev_pnl = 0.
        self.mid_price = mid_price
        self.pnl_hist = []
        self.actions = []
        self.PnL = 0.
        
    
    def step(self, action):
        pnl_add = 0.
        pnl_mem = 0.
        self.actions.append(action)
        
        if self.mid_price:
            _, _, p_a, p_b, flag = self.data_maker.get()
            S = 0.5 * (p_a + p_b)
        else:
            v_a, v_b, p_a, p_b, flag = self.data_maker.get()
            w_a = v_a / (v_a + v_b)
            w_b = 1. - w_a
            S = w_a * p_a + w_b * p_b
    
#        S = np.log(S / np.exp(self.S_prev))
        add_for_nothing = 0
        
        if action == 0:
            if len(self.short_positions) == 0: # no shorts
                self.long_positions.append(S)
                self.q += 1
            else:
                if self.all_close: 
                    self.q += len(self.short_positions)
                    pnl_mem = np.sum(self.short_positions - S)
                    self.short_positions = []
                else:
                    pass
        elif action == 1:
            if len(self.long_positions) == 0: # no longs
                self.short_positions.append(S)
                self.q -= 1
            else:
                if self.all_close: 
                    self.q -= len(self.long_positions)
                    pnl_mem = np.sum(S - self.long_positions)
                    self.long_positions = []
                else:
                    pass
        else:
            add_for_nothing = -1.0 / 10
            
        self.prev_pnl = self.PnL
        self.PnL += pnl_mem
        
        # check that there is no current data
        if self.data_maker.is_empty():
            flag = True
            
        is_done = flag
        
        reward = calculate_reward(self.q, self.q_hist[-1], self.PnL, self.prev_pnl, S, self.S_prev, flag, self.gamma) + add_for_nothing 
        
        info = 'Done' if is_done else 'Continue'
        
        self.prev_S = S
        
        self.pnl_hist.append(self.PnL)
        self.q_hist.append(self.q)
        
        self.cur += 1
        
        if flag == True:
            self.pnl_hist[-1] += np.sum(S - self.long_positions) + np.sum(self.short_positions - S)
        
        return self.q, reward, is_done, info
    
    
    def reset(self):
        self.long_positions = []
        self.short_positions = []
        self.cur = 0
        self.data_maker.step()
#        self.pnl_hist = []
        self.q_hist = [0]
#        self.actions = []
        self.q = 0
        self.S_prev = 0
        
        return self.q

    def render(self, mode='human', close=False):
        pass

    def take_action(self, action):
        pass