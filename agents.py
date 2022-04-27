import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import random


action_space = {
    0: 'buy',
    1: 'sell',
    2: 'hold'
}

class DQNAgent(nn.Module):
    def __init__(self, state_shape, n_actions, device, epsilon=0):
        super().__init__()
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.state_shape = state_shape
        self.device = device
        
        self.lin = nn.Linear(state_shape, 256)
        
        self.advantage = nn.Sequential(
            nn.Linear(256, n_actions)
        )
        self.value = nn.Sequential(
            nn.Linear(256, 1)
        )
        

    def forward(self, state_t):
        out = self.lin(state_t)
        out = F.relu(out)
        
        V = self.value(out)
        A = self.advantage(out)
        
        qvalues = V + A - A.mean()

        return qvalues

    def get_qvalues(self, states):
        
        model_device = next(self.parameters()).device
        states = torch.tensor(states, device=model_device, dtype=torch.float32)
        qvalues = self.forward(states)
        
        return qvalues.data.cpu().numpy()

    def sample_actions(self, qvalues):
        """pick actions given qvalues. Uses epsilon-greedy exploration strategy. """
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape

        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)

        should_explore = np.random.choice(
            [0, 1], batch_size, p=[1-epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)
    
    def get_action(self, state):
        state = np.array([state])
        state = torch.from_numpy(state).type(torch.float32)
        q_value = self.forward(state)
        
        random_actions = np.random.choice(self.n_actions)

        if random.random() > self.epsilon:
            action  = q_value.max(1)[1].data[0].cpu()
            return action
        
        return random_actions