import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTOR_LAYER_SIZES = [256, 128]
CRITIC_LAYER_SIZES = [256, 128]

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, actor_layer_sizes=ACTOR_LAYER_SIZES):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            actor_layer_sizes (list of int): Number of nodes in each layer of the network
        """
        
#         print(f"Actor: state_size={state_size}, action_size={action_size}, actor_layer_sizes={actor_layer_sizes}")
        
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, actor_layer_sizes[0])
        self.fc2 = nn.Linear(actor_layer_sizes[0], actor_layer_sizes[1])
        self.fc3 = nn.Linear(actor_layer_sizes[1], action_size)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.leaky_relu(self.fc1(state))
        x = F.leaky_relu(self.fc2(x))
        return F.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, critic_layer_sizes=CRITIC_LAYER_SIZES):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            critic_layer_sizes (list of int): Number of nodes in each layer of the network
        """

#         print(f"Critic: state_size={state_size}, action_size={action_size}, critic_layer_sizes={critic_layer_sizes}")

        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, CRITIC_LAYER_SIZES[0])
        self.fc2 = nn.Linear(CRITIC_LAYER_SIZES[0]+action_size, CRITIC_LAYER_SIZES[1])
        self.fc3 = nn.Linear(CRITIC_LAYER_SIZES[1], 1)        

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.leaky_relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.leaky_relu(self.fc2(x))
        return self.fc3(x)
