import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
import os
import subprocess
import glob
import cv2
import torch.nn as nn
import torch
import random
from tqdm import trange

from agents import DQNAgent
from envs import Environment
from loss import loss_f
from buffers import ExpReplay, ReplayBuffer

class Streamer():
    def __init__(self, value=0., dt=0.1):
        self.value = value
        self.dt = dt
        
        self.process = None
        self.new = True
        self.flag = False
        
    def set_dt(self, dt):
        self.dt = dt
        
    def set_value(self, value):
        self.value = value
        self.new = False
        
    def start(self, process=None, flag=False, asset=None):
        self.process = process
        self.flag = flag
        
        value = process(flag, self, asset)
        
        self.set_value(value)
        
    def step(self, asset):
        value = self.process(self.flag, self, asset)
        
        self.set_value(value)

class Asset():
    def __init__(self, S_0, r, sigma):
        self.S_0 = S_0
        self.r = r
        self.sigma = sigma
        
        self.last_price = None
        self.W = None
        
    def price(self, price, W):
        self.last_price = price
        self.W = W
        
    def get(self):
        return self.last_price, self.W
            
def simulate_path_infinite(
    geom: bool=False, streamer: Streamer = None, asset: Asset = None
):
    '''
    Simulate asset price over time for online work.
    '''
    
    if streamer.new == True:
        W = np.random.normal(0, 1, 1)
        S_new = asset.sigma * W
        
        if geom:
            S_new = asset.S_0 * (asset.sigma * W + asset.r)
        
        asset.price(S_new, W)
        
        return S_new
    
    else:
        
        S_last, W_last = asset.get()
        
        W = np.random.normal(0, 1, 1)
        S_new = asset.sigma * (W - W_last)
        
        if geom:
            S_new = S_last * (asset.sigma * (W - W_last) + asset.r)
            
        return S_new
        
    
    
def simulate_path_finite(
    N: int=10_000, geom: bool=False,
    S_0: float = 100, vol: float=0.2, r: float=0.05, T: float=1.0
):
    '''
    Simualate asset price over time. 
    dS_t = sigma dW_t.
    '''
    
    path = np.zeros(N)
    path[0] = S_0
    
    time = np.linspace(0., T, N)
    
    process = np.random.normal(0, 1, size=N)
    
    for i in range(1, N):
        dt = time[i] - time[i - 1]
        if geom:
            path[i] = path[i - 1] * r * dt + path[i - 1] * dt * vol * (process[i] - process[i - 1])
        else:
            path[i] = vol * process[i] * dt
            
    return path


class DataMaker:
    def __init__(self, data, times):
        self.data = data
        self.timestamps = times
        self.cur = 0
        self.ends_of_trading = []
        self.previous_day = -1
        
    def is_empty(self):
        return self.cur == len(self.timestamps)
        
    def step(self):
        if self.cur != len(self.timestamps):
            self.previous_day = self.timestamps[self.cur].hour
        else:
            self.cur = np.random.choice(self.ends_of_trading)
    
    def get(self):
        if self.cur == len(self.timestamps):
            self.cur = np.random.choice(self.ends_of_trading)
            return *self.data[self.cur - 1], True
        if self.timestamps[self.cur].hour == self.previous_day:
            self.cur += 1
            return *self.data[self.cur - 1], False
        else:            
            self.cur += 1
            self.ends_of_trading.append(self.cur)
            return *self.data[self.cur - 1], True
        
    
class Logger():
    def __init__(self, data_path=None, data_name=None, file_type='csv'):
        self.data_path = data_path
        self.data_name = data_name
        self.data_storage = None
        self.file_type = file_type
        
        self.output = None
        
        self.images = []
        self.video_flag = False
        
        self.cnt = 0
        
    def get_file(self, to_path=None, type_file=False):
        name_f = self.file_type
        if name_f == 'csv':
            self.data = pd.read_csv(name_f)
        elif name_f == 'parquet':
            self.data = pd.read_parquet(name_f)
        elif name_f == 'excell':
            self.data = pd.read_excel(name_f)
        else:
            raise NotImplemetedError
        
#    def read(self):
#        if self.data_path is None and self.data_name is not None:
            
    def send(self, list_of_data):
        self.output = list_of_data
        
    def plot(self, video_flag=False, path=None):
        list_of_data = self.output

        N = len(list_of_data)
            
        if video_flag:
            self.video_flag = True
            self.images = []
            
            fig, ax = plt.subplots(1, N, figsize=(5 * N, N))
            for i in range(N):
                ax[i].set_title(list_of_data[i][1])
                ax[i].plot(np.arange(len(list_of_data[i][0])), list_of_data[i][0], label=list_of_data[i][1])
                ax[i].grid()
                
            plt.savefig(path + "/file%02d.png" % self.cnt)
            
            plt.close(fig)
            
            self.cnt += 1
            
        else:
            fig, ax = plt.subplots(1, N, figsize=(5 * N, N))
            for i in range(N):
                ax[i].set_title(list_of_data[i][1])
                ax[i].plot(np.arange(len(list_of_data[i][0])), list_of_data[i][0], label=list_of_data[i][1])
                ax[i].grid()
                
            plt.show()
        clear_output(True)
    
    def generate_video(self, path):
        video_name = 'video.avi'

        images = [img for img in sorted(os.listdir(path)) if img.endswith(".png")]
        frame = cv2.imread(os.path.join(path, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_name, 0, 1, (width,height))

        for image in images:
            video.write(cv2.imread(os.path.join(path, image)))

        cv2.destroyAllWindows()
        video.release()
        
        os.rmdir(path)
        
def evaluate(env, agent, n_games=1, greedy=False, t_max=100):
    """ Plays n_games full games. If greedy, picks actions as argmax(qvalues). Returns mean reward. """
    rewards = []
    for _ in range(n_games):
        s = env.reset()
        reward = 0
        for _ in range(t_max):
            qvalues = agent.get_qvalues(np.array([s]))
#            print( qvalues.argmax(axis=-1), agent.sample_actions(qvalues))
            #Only greedy
            #print(qvalues)
            action = qvalues.argmax(axis=-1)
            
            s, r, done, _ = env.step(action)
            reward += r
            if done:
                break

        rewards.append(reward)
    #print(rewards)
    return np.mean(rewards)
        
def make_env(env_params):
    return Environment(**env_params)

def train(agent, target_network, device, env, env_params, num_of_steps):
    seed = 11
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    target_network.load_state_dict(agent.state_dict())

    REPLAY_BUFFER_SIZE = 10**6
    N_STEPS = 100

#    exp_replay = ExpReplay(REPLAY_BUFFER_SIZE)
    exp_replay = ReplayBuffer(REPLAY_BUFFER_SIZE)

    timesteps_per_epoch = 1
    batch_size = 32
    total_steps = num_of_steps
    decay_steps = 2 * 10**6

    opt = torch.optim.Adam(agent.parameters(), lr=1e-4)

    init_epsilon = 1
    final_epsilon = 0.1

    loss_freq = 10000
    refresh_target_network_freq = 5000
    eval_freq = 5000

    max_grad_norm = 50

    n_lives = 5

    beta = 0.5

    mean_rw_history = []
    td_loss_history = [0]
    grad_norm_history = []
    initial_state_v_history = []
    step = 0
    
    state = env.reset()
    beta = 0.5
    episode_reward = 0
    
    with trange(step, total_steps + 1) as progress_bar:
        for step in progress_bar:
            cur = step

            action = agent.get_action(state)

            next_state, reward, done, _ = env.step(action)
            exp_replay.add(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            if done:
                state = env.reset()
                episode_reward = 0

            if len(exp_replay) > batch_size:
                beta = beta + step * (1000 - beta) / total_steps
                loss_f(td_loss_history, exp_replay, opt, agent, target_network, batch_size, beta, 0.99, device)
            
            #mean_rw_history.append(episode_reward)
            
            if step % refresh_target_network_freq == 0:
                # Load agent weights into target_network
                target_network.load_state_dict(agent.state_dict())
            
            #print(env.q_hist, env.pnl_hist, env.cur)
            if step % 10000 == 0:
                clear_output(True)
                
                fig, ax = plt.subplots(2, 2, figsize=(20, 12))
                ax[0, 0].plot(np.array(td_loss_history)[np.arange(0, step, 10000)])
                ax[0, 0].set_title("Loss.")
                ax[0, 0].grid()
                ax[0, 1].set_title("Mean reward.")
                #mean_rw_history.append(evaluate(
                #    make_env(env_params), agent, n_games=10, greedy=True)
                #)
                ax[0, 1].hist(np.array(env.actions), density=True)
                ax[0, 1].grid()
                ax[1, 0].plot(np.array(env.pnl_hist))
                ax[1, 0].grid()
                ax[1, 1].plot(np.array(env.q_hist))
                ax[1, 1].grid()
            
                plt.show()