import torch
import numpy as np

def loss_f(losses, buffer, opt, agent, target_network, batch_size, gamma=1., device='cpu'):
    """ Compute td loss using torch operations only."""
    states, actions, rewards, next_states, is_done= buffer.sample(batch_size)

    states = torch.tensor(states.astype('float32'), device=device)    # shape: [batch_size, *state_shape]
    actions = torch.tensor(actions, device=device)    # shape: [batch_size]
    rewards = torch.tensor(rewards, device=device)  # shape: [batch_size]
#    weights = torch.tensor(weights, device=device)
    next_states = torch.tensor(next_states.astype('float32'), device=device)
    
    
    is_done = torch.tensor(
        is_done.astype('float32'),
        device=device,
        dtype=torch.float32,
    )  # shape: [batch_size]
    
    is_not_done = 1 - is_done

    
    qvalues = agent(states)  # shape: [batch_size, n_actions]
    next_qvalues = target_network(next_states)  # shape: [batch_size, n_actions]

#    print(qvalues.shape)
    
    real_q_values = qvalues.gather(1, actions.unsqueeze(1)).squeeze(1)
    target_q_values = next_qvalues.max(1)[0]
    expected_q_value = rewards + gamma * target_q_values * is_not_done

    loss  = (real_q_values - expected_q_value.detach()).pow(2) 
#    w = loss
    loss  = loss.mean()
    
    opt.zero_grad()
    loss.backward()
#    buffer.update_priorities(indices, w.data.cpu().numpy())
    opt.step()

    losses.append(loss.data.item())