import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic
from sum_tree import SumTree

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 512        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
TAU_INCREASE = 1.001    # rate of increase of tau
LR_ACTOR = 0.5e-4       # learning rate of the actor 
LR_CRITIC = 0.5e-5      # learning rate of the critic
WEIGHT_DECAY = 0.0      # L2 weight decay
NOISE_THETA = 0.15      # OUNoise theta
NOISE_SIGMA = 0.2       # OUNoise sigma
ALPHA = 0.6             # alpha parameter of prioritized replay
EPSILON_ERROR = 1e-7    # minimum TD-error for prioritized replay
MAXIMUM_ERROR = 1e4     # default error for experiences newly added to the replay buffer

RANDOM_ACTION_PERIOD = 1500         # length of the exploration period in early training
MINIMUM_RANDOM_ACTION_PROB = 0.01   # probability of taking random actions during the exploration period

ACTOR_LAYER_SIZES = [256, 128]
CRITIC_LAYER_SIZES = [256, 128]

def if_print(condition, item):
    if condition:
        print(item)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def override_config(config):
    """Overrides the above global parameters used by the Agent."""
    
    global BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, TAU_INCREASE, LR_ACTOR, LR_CRITIC 
    global WEIGHT_DECAY, NOISE_THETA, NOISE_SIGMA, ALPHA, EPSILON_ERROR, MAXIMUM_ERROR
    global RANDOM_ACTION_PERIOD, MINIMUM_RANDOM_ACTION_PROB
    global ACTOR_LAYER_SIZES, CRITIC_LAYER_SIZES, EPSILON_ERROR
    
    BUFFER_SIZE = config["buffer_size"]
    BATCH_SIZE = config["batch_size"]
    GAMMA = config["gamma"]
    TAU = config["tau"]             
    TAU_INCREASE = config["tau_increase"]
    LR_ACTOR = config["lr_actor"]    
    LR_CRITIC = config["lr_critic"]
    WEIGHT_DECAY = config["weight_decay"]
    NOISE_THETA = config["noise_theta"]
    NOISE_SIGMA = config["noise_sigma"]
    ALPHA = config["alpha"]
    EPSILON_ERROR = config["epsilon_error"]
    MAXIMUM_ERROR = config["maximum_error"]
    RANDOM_ACTION_PERIOD = config["random_action_period"]
    MINIMUM_RANDOM_ACTION_PROB = config["minimum_random_action_prob"]
    ACTOR_LAYER_SIZES = config["actor_layer_sizes"]
    CRITIC_LAYER_SIZES = config["critic_layer_sizes"]

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed, 
                 critic_combines_state_action=True,
                 prioritized_replay=False, use_ounoise=True, parallel_agents=1,
                 train_every=20, train_steps=10):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            critic_combines_state_action (bool): if True, the critic's state includes the other agent's action
            prioritized_replay (bool): if True, use prioritized replay. Otherwise don't
            use_ounoise (bool): if True, uses Ornstein-Uhlenbeck processes to add noise to the output of the policy
            parallel_agents (int): number of agents running in parallel
            train_every (int): number of steps to take before switching to train mode
            train_steps (int): number of times to update the network in train mode
        """
        
#         print(f"Agent: state_size={state_size}, action_size={action_size}")
#         print(f"Actor_layer_sizes={ACTOR_LAYER_SIZES}, Critic_layer_sizes={CRITIC_LAYER_SIZES}")
#         print(f"lr_actor={LR_ACTOR}, lr_critic={LR_CRITIC}")
#         print(f"critic_combines_state_action={critic_combines_state_action}")
#         print(f"train_every={train_every}, train_steps={train_steps}")        
#         print(f"buffer_size={BUFFER_SIZE}, batch_size={BATCH_SIZE}")
#         print(f"gamma={GAMMA}, tau={TAU}, weight_decay={WEIGHT_DECAY}")
#         print(f"use_ounoise={use_ounoise}, noise_theta={NOISE_THETA}, noise_sigma={NOISE_SIGMA}")
#         print(f"prioritized_replay={prioritized_replay}")
#         print("\n")

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.critic_combines_state_action = critic_combines_state_action
        self.prioritized_replay = prioritized_replay
        self.use_ounoise = use_ounoise
        self.parallel_agents = parallel_agents
        self.train_every = train_every
        self.train_steps = train_steps
        self.tau = TAU
        
        # Note that with a sum_tree implementation, alpha has to be fixed
        # and alpha is applied at the time an item is added or updated.
        self.alpha = ALPHA
        
        actor_state_size = self.state_size
        actor_state_size *= 2
            
        critic_state_size = self.state_size * 2
        critic_action_size = action_size * 2
        if critic_combines_state_action:
            critic_state_size += self.action_size
            critic_action_size = action_size

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(actor_state_size, action_size, random_seed,
                                actor_layer_sizes=ACTOR_LAYER_SIZES).to(device)
        self.actor_target = Actor(actor_state_size, action_size, random_seed,
                                 actor_layer_sizes=ACTOR_LAYER_SIZES).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(critic_state_size, critic_action_size, random_seed, 
                                   critic_layer_sizes=CRITIC_LAYER_SIZES).to(device)
        self.critic_target = Critic(critic_state_size, critic_action_size, random_seed,
                                   critic_layer_sizes=CRITIC_LAYER_SIZES).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        
        # Copy the local networks into the target networks
        self.soft_update_targets(tau=1.0)

        # Noise process
        self.noise = OUNoise(action_size * parallel_agents, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed, 
                                   prioritized_replay=prioritized_replay)    

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
        self.reset()
                        
    def reset(self, theta=NOISE_THETA, sigma=NOISE_SIGMA):
        """Reset the noise generator given the theta and sigma parameters."""        
        self.noise.reset(theta, sigma)
       
    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy.
        Noise can be added to the action proposed by the policy to
        encourage exploration. Moreover, in the initial steps of training, 
        the actions proposed by the policy may be randomly overriden 
        by completely random actions, to encourage even more diversified
        exploration, which is controlled by two hyper-parameters
        RANDOM_ACTION_PERIOD and MINIMUM_RANDOM_ACTION_PROB.
        
        Params
        ======
            add_noise (bool): if True, add noise to the action proposed by the actor
        """
        
        a1_state = state
        a2_state = state[::-1]
        
        total_state = np.concatenate((a1_state, a2_state), axis=1)            
        total_state = torch.from_numpy(total_state).float().to(device)
        
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(total_state).cpu().data.numpy()
        self.actor_local.train()
        
        if add_noise:
            if self.use_ounoise:
                action += self.noise.sample().reshape((-1, self.action_size))[:len(action)]
            else:
                action += np.random.standard_normal(action.shape)

        action = np.clip(action, -1, 1)
        
        # Flip a coin to see if you better take a random action instead.
        # Reduce the randomness as time goes by
        for i in range(action.shape[0]):
            random_action_prob = max((RANDOM_ACTION_PERIOD - self.t_step)/RANDOM_ACTION_PERIOD, 
                                     MINIMUM_RANDOM_ACTION_PROB)
            if (np.random.random() <= random_action_prob):
                # take a random action
                action[i] = np.random.uniform(-1.0, 1.0, action.shape[1])

        return action
    
    def step(self, state, action, reward, next_state, done, beta=0.0):
        """Update the agent based on the taken step.
        Save experience in replay memory.
        Use random sample from buffer to learn.
        
        Params
        =====
            beta (float): beta power used for importance sampling weights
        """
        
        # Choose an error for these experiences
        if self.prioritized_replay:
            # Set the error for this experience to be equal to the highest error in the replay buffer.
            error = self.memory.get_highest_error()
        else:
            error = None
            
        other_state = state[::-1]
        combined_state = np.concatenate((state, other_state), axis=1)
        
        other_action = action[::-1]
        combined_action = np.concatenate((action, other_action), axis=1)
        
        other_next_state = next_state[::-1]
        combined_next_state = np.concatenate((next_state, other_next_state), axis=1)
                
        combined_reward = reward
                        
        # Save experiences into replay buffer, along their errors
        for s, a, r, n_s, d in zip(combined_state, combined_action, combined_reward, combined_next_state, done):
            self.memory.add(s, a, r, n_s, d, error)
                    
        # Learn every train_every time steps. 
        if (self.t_step + 1) % self.train_every == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                # learn for train_steps times
                for i in range(self.train_steps):
                    experiences, indices, priorities = self.memory.sample()
                    self.learn(experiences, indices, priorities, GAMMA, beta=beta)
        self.t_step += 1

    def learn(self, experiences, indices, priorities, gamma, beta):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done, error) tuples 
            indices (int): index of experiences in the replay buffer
            priorities (float): priority of experiences in the replay buffer
            gamma (float): discount factor
            beta (float): beta power used for importance sampling weights
        """
        
        states, actions, rewards, next_states, dones = experiences
        
        # extract the states, next_states, and actions from the perspective
        # of each of the two agents: a1, and a2
        a1_a2_states = states
        a1_states = states[:, :self.state_size]
        a2_states = states[:, self.state_size:]
        a2_a1_states = torch.cat((a2_states, a1_states), dim=1)
        
        a1_a2_next_states = next_states
        a1_next_states = next_states[:, :self.state_size]
        a2_next_states = next_states[:, self.state_size:]
        a2_a1_next_states = torch.cat((a2_next_states, a1_next_states), dim=1)
        batched_a1a2_a2a1_next_states = torch.cat((a1_a2_next_states, a2_a1_next_states), dim=0)
        batched_a1_a2_next_states = torch.cat((a1_next_states, a2_next_states), dim=0)
                
        a1_a2_actions = actions
        a1_actions = actions[:, :self.action_size]
        a2_actions = actions[:, self.action_size:]
        
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        self.actor_target.eval()
        self.critic_target.eval()

        with torch.no_grad():
            batched_a1_a2_next_actions = self.actor_target(batched_a1a2_a2a1_next_states)
            a1_next_actions = batched_a1_a2_next_actions[:BATCH_SIZE, :]
            a2_next_actions = batched_a1_a2_next_actions[BATCH_SIZE:, :]
            if not self.critic_combines_state_action:                
                a1_a2_next_actions = torch.cat((a1_next_actions, a2_next_actions), dim=1)
                Q_targets_next = self.critic_target(a1_a2_next_states, a1_a2_next_actions)
            else:
                a1_a2_next_states_a2_next_actions = torch.cat((a1_a2_next_states, a2_next_actions), dim=1)
                Q_targets_next = self.critic_target(a1_a2_next_states_a2_next_actions, a1_next_actions)
        
        # Adjust the target_best_next_action_value by importance sampling weights
        if self.prioritized_replay:
            weights = (len(self.memory) * priorities) ** -beta
            maximum_weight, _ = torch.max(weights, dim=0, keepdim=True)
            weights /= maximum_weight
            Q_targets_next *= weights

        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # Compute critic loss
        if not self.critic_combines_state_action:
            Q_expected = self.critic_local(a1_a2_states, a1_a2_actions)
        else:
            a1_a2_states_a2_actions = torch.cat((a1_a2_states, a2_actions), dim=1)
            Q_expected = self.critic_local(a1_a2_states_a2_actions, a1_actions)
        
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
    
        # Update errors of sampled experiences
        if self.prioritized_replay:
            errors = (torch.abs(Q_expected - Q_targets).squeeze().detach().to('cpu').numpy() + EPSILON_ERROR) ** self.alpha
            self.memory.update_errors(indices, errors)
            
        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss        
        a1_actions = self.actor_local(a1_a2_states)
        a2_actions = self.actor_local(a2_a1_states)
        
        if not self.critic_combines_state_action:
            a1_a2_actions = torch.cat((a1_actions, a2_actions), dim=1)
            actor_loss = -self.critic_local(a1_a2_states, a1_a2_actions).mean()
        else:
            a1_a2_states_a2_actions = torch.cat((a1_a2_states, a2_actions), dim=1)
            actor_loss = -self.critic_local(a1_a2_states_a2_actions, a1_actions).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.tau = min(5e-1, self.tau * TAU_INCREASE)
        self.soft_update_targets(tau=self.tau)  # Note, this now needs to be done by the caller!

    def soft_update_targets(self, tau=TAU):
        self.soft_update(self.critic_local, self.critic_target, tau)
        self.soft_update(self.actor_local, self.actor_target, tau)         

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=NOISE_THETA, sigma=NOISE_SIGMA):
        """Initialize parameters and the noise processes."""
        self.size = size
        self.mu = mu * np.ones(size)
        self.seed = random.seed(seed)
        self.reset(theta, sigma)

    def reset(self, theta=NOISE_THETA, sigma=NOISE_SIGMA):
        """Reset the internal state (= noise) to mean (mu), 
        and decay the theta and sigma parameters."""
        self.state = copy.copy(self.mu)
        self.theta = theta
        self.sigma = sigma

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        noise = np.random.standard_normal(self.size)
        dx = self.theta * (self.mu - x) + self.sigma * noise
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed, prioritized_replay=False):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            prioritized_replay (bool): if True, use prioritized replay
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.prioritized_replay = prioritized_replay
        if not prioritized_replay:
            self.memory = deque(maxlen=buffer_size)
        else:
            self.memory = [None] * buffer_size
            self.sum_tree = SumTree(buffer_size)
            self.data_index = 0
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])             
    
    def add(self, state, action, reward, next_state, done, error=0.0):
        """Add a new experience to memory."""

        e = self.experience(state, action, reward, next_state, done)
        if not self.prioritized_replay:
            self.memory.append(e)
        else:
            self.memory[self.data_index] = e
            self.sum_tree.add(error)
            self.data_index += 1
            if self.data_index == self.buffer_size:
                self.data_index = 0
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        
        if not self.prioritized_replay:
            experiences = random.sample(self.memory, k=self.batch_size)
            indices = None
            priorities = None
        else:
            indices, priorities = self.sum_tree.weighted_sample(self.batch_size)
            total_sum = self.sum_tree.total_sum()
            priorities = np.array(priorities) / total_sum
            experiences = [self.memory[index] for index in indices]

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        if self.prioritized_replay:
            priorities = torch.from_numpy(np.vstack(priorities)).float().to(device)
        else:
            priorities = None
        return (states, actions, rewards, next_states, dones), indices, priorities

    def get_highest_error(self):
        """Return the highest error of all experiences in the replay buffer.
        Used only for prioritized replay.
        """
        
        assert(self.prioritized_replay == True)
        largest_probability = self.sum_tree.largest_item()
        if largest_probability == 0.0:
            return (MAXIMUM_ERROR)
        else:
            return largest_probability
                
    def update_errors(self, indices, errors):
        """Update the TD-errors of sampled replay buffer experiences.
        Used only for prioritized replay.
        
        Params
        ======
            indices (int): replay buffer locations that need to be updated.
            errors (float): new values for the error field of the replay buffer locations
        """
        assert(self.prioritized_replay == True)
        for i in range(len(indices)):
            self.sum_tree.update(i, errors[i])

    def __len__(self):
        """Return the current size of internal memory."""
        if not self.prioritized_replay:
            return len(self.memory)
        else:
            return len(self.sum_tree)
