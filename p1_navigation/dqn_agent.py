import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)     # replay buffer size
BATCH_SIZE = 64            # minibatch size
GAMMA = 0.99               # discount factor
TAU = 1e-3                 # for soft update of target parameters
LR = 5e-4                  # learning rate 
UPDATE_EVERY = 4           # how often to update the network
EPSILON_PRIORITY = 1e-7    # minimum TD-error for prioritized replay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, ddqn=False, prioritized_replay=False):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            ddqn (bool): Use Double DQN
            prioritized_replay (bool): use prioritized replay
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.ddqn = ddqn
        self.prioritized_replay = prioritized_replay
        
        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, prioritized_replay=prioritized_replay)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0        
    
    def step(self, state, action, reward, next_state, done, alpha=0.0, beta=0.0):
        """Update the agent based on the taken step.
        
        Params
        =====
            state (array_like): state in which an action was taken
            action (array_like): the action that was taken
            reward (float): the reward that was received for the taken action
            next_state (array_like): the state arrived at after having taken the action
            done (bool): whether the episode is complete
            alpha (float): alpha power used for importance sampling priorities
            beta (float): beta power used for importance sampling weights
        """
        
        # Save experience in replay memory
        if self.prioritized_replay:
            # Set the error for this experience to be equal to the highest error in the replay buffer.
            error = self.memory.get_highest_error()
        else:
            error = None
        self.memory.add(state, action, reward, next_state, done, error)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get a random batch and learn
            if len(self.memory) > BATCH_SIZE:
                experiences, indices, priorities = self.memory.sample(alpha=alpha)
                self.learn(experiences, indices, priorities, GAMMA, beta=beta)

    def act(self, state, eps=0.0):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """        
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))        

    def learn(self, experiences, indices, priorities, gamma, beta=0.0):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done, error) tensors 
            indices (array of int): index of experiences in the replay buffer
            priorities (array of float): priority of experiences in the replay buffer
            gamma (float): discount factor
            beta (float): beta power used for importance sampling weights
        """
        states, actions, rewards, next_states, dones = experiences

        self.qnetwork_target.eval()
        
        if not self.ddqn:
            with torch.no_grad():
                target_next_action_values = self.qnetwork_target(next_states)        
            target_best_next_action_value, _ = torch.max(target_next_action_values, dim=1, keepdim=True)
            
        else:
            # The local network of DDQN is in eval mode for computing the best next action.
            with torch.no_grad():
                self.qnetwork_local.eval()
                
                # Select the best next action using the local network
                local_next_action_values = self.qnetwork_local(next_states)
                _, local_best_next_action = torch.max(local_next_action_values, dim=1, keepdim=True)
                
                # Evaluate the best next actions in the target network
                target_next_action_values = self.qnetwork_target(next_states)
                target_best_next_action_value = torch.gather(target_next_action_values, 1, local_best_next_action)
            
        target_best_next_action_value = target_best_next_action_value * (1.0 - dones)
        
        # Adjust the target_best_next_action_value by importance sampling weights
        if self.prioritized_replay:
            weights = (len(self.memory) * priorities) ** -beta
            maximum_weight, _ = torch.max(weights, dim=0, keepdim=True)
            weights /= maximum_weight
            target_best_next_action_value *= weights
            
        estimated_q_values = rewards + target_best_next_action_value * gamma
        
        self.qnetwork_local.train()
        self.optimizer.zero_grad()
        action_values = self.qnetwork_local(states)
        q_values = torch.gather(action_values, 1, actions)
        
        # Update errors of sampled experiences
        if self.prioritized_replay:
            errors = torch.abs(q_values - estimated_q_values) + EPSILON_PRIORITY
            self.memory.update_errors(indices, errors)        

        loss = F.mse_loss(q_values, estimated_q_values)
        loss.backward()
        self.optimizer.step()
        
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, prioritized_replay=False):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            prioritized_replay (bool): whether importance sampling is enabled
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.prioritized_replay = prioritized_replay
        if prioritized_replay:
            self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "error"])
        else:
            self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])           
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done, error=0.0):
        """Add a new experience to memory."""
        if self.prioritized_replay:
            e = self.experience(state, action, reward, next_state, done, error=error)
        else:
            e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self, alpha=0.0):
        """Randomly sample a batch of experiences from memory.
        
        Params
        ======
            alpha (float): alpha power for importance sampling priorities
        """
        
        if not self.prioritized_replay:
            experiences = random.sample(self.memory, k=self.batch_size)
            indices = None
            priorities = None
        else:
            errors = np.array([x.error for x in self.memory]) ** alpha # error
            partition_function = np.sum(errors)
            errors_probability = errors / partition_function
            indices = random.choices(range(len(self.memory)), weights=errors_probability, k=self.batch_size)
            experiences = [self.memory[index] for index in indices]
            priorities = errors_probability[indices]
                  
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        if self.prioritized_replay:
            priorities = torch.from_numpy(np.vstack(priorities)).float().to(device)
        else:
            priorities = None
        return (states, actions, rewards, next_states, dones), indices, priorities
    
    def get_highest_error(self):
        """Return the highest error of all experiences in the replay buffer."""
        if not self.prioritized_replay or len(self.memory) == 0:
            return EPSILON_PRIORITY
        else:
            return max(map((lambda x: x[-1]), self.memory))
        
    def update_errors(self, indices, errors):
        """Updates TD-errors for sampled replay buffer experiences.
        
        Params
        ======
            indices (int): replay buffer locations that need to be updated.
            errors (float): new values for the error field of the replay buffer locations
        """
        if self.prioritized_replay:
            for i in range(len(indices)):
                self.memory[indices[i]] = self.memory[indices[i]]._replace(error=errors[i].item())

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)