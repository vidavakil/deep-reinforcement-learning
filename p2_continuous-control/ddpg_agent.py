import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 512        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 0.5e-4       # learning rate of the actor 
LR_CRITIC = 0.5e-5      # learning rate of the critic
WEIGHT_DECAY = 0.0      # L2 weight decay
NOISE_THETA = 0.15      # OUNoise theta
NOISE_SIGMA = 0.2       # OUNoise sigma
EPSILON_PRIORITY = 1e-7 # minimum TD-error for prioritized replay

ACTOR_LAYER_SIZES = [256, 128]        # layer sizes in the actor model
CRITIC_LAYER_SIZES = [256, 256, 128]  # layer sizes in the critic model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def override_config(config):
    """Overrides the above global parameters used by the Agent."""
    
    global BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR_ACTOR, LR_CRITIC, WEIGHT_DECAY, NOISE_THETA
    global NOISE_SIGMA, EPSILON_PRIORITY, ACTOR_LAYER_SIZES, CRITIC_LAYER_SIZES
    
    BUFFER_SIZE = config["buffer_size"]
    BATCH_SIZE = config["batch_size"]
    GAMMA = config["gamma"]**config["unroll_steps_every"]
    TAU = config["tau"]             
    LR_ACTOR = config["lr_actor"]    
    LR_CRITIC = config["lr_critic"]
    WEIGHT_DECAY = config["weight_decay"]
    NOISE_THETA = config["noise_theta"]
    NOISE_SIGMA = config["noise_sigma"]
    EPSILON_PRIORITY = config["epsilon_priority"]
    ACTOR_LAYER_SIZES = config["actor_layer_sizes"]
    CRITIC_LAYER_SIZES = config["critic_layer_sizes"]


class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed, 
                 prioritized_replay=False, parallel_agents=1,
                 train_every=20, train_steps=10):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            prioritized_replay (bool): if True, use prioritized replay. Otherwise don't
            parallel_agents (int): number of agents running in parallel
            train_every (int): number of steps to take before switching to train mode
            train_steps (int): number of times to update the network in train mode
        """       
#         print(f"Agent: state_size={state_size}, action_size={action_size}")
#         print(f"prioritized_replay={prioritized_replay}")
#         print(f"Actor_layer_sizes={ACTOR_LAYER_SIZES}, Critic_layer_sizes={CRITIC_LAYER_SIZES}")        
#         print(f"train_every={train_every}, train_steps={train_steps}")        
#         print(f"lr_actor={LR_ACTOR}, lr_critic={LR_CRITIC}")
#         print(f"buffer_size={BUFFER_SIZE}, batch_size={BATCH_SIZE}")
#         print(f"gamma={GAMMA}, tau={TAU}, weight_decay={WEIGHT_DECAY}")
#         print(f"noise_theta={NOISE_THETA}, noise_sigma={NOISE_SIGMA}")
#         print("\n")

        self.state_size = state_size
        self.action_size = action_size
        self.parallel_agents = parallel_agents
        self.seed = random.seed(random_seed)
        self.prioritized_replay = prioritized_replay
        self.train_every = train_every
        self.train_steps = train_steps

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed,
                                actor_layer_sizes=ACTOR_LAYER_SIZES).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed,
                                 actor_layer_sizes=ACTOR_LAYER_SIZES).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed, 
                                   critic_layer_sizes=CRITIC_LAYER_SIZES).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed,
                                   critic_layer_sizes=CRITIC_LAYER_SIZES).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        
        # Copy the local networks into the target networks
        self.soft_update_targets(tau=1.0)

        # Noise process
        self.noise = OUNoise(action_size * parallel_agents, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed, prioritized_replay=prioritized_replay)    

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
        self.reset()
                        
    def reset(self, theta=NOISE_THETA, sigma=NOISE_SIGMA):
        """Reset the noise generator, and its parameters."""        
        self.noise.reset(theta, sigma)
       
    def act(self, state):
        """Returns actions for given state as per current policy."""        
        state = torch.from_numpy(state).float().to(device)
         
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        action += self.noise.sample().reshape((-1, self.action_size))[:len(action)]
        return np.clip(action, -1, 1)
    
    def step(self, state, action, reward, next_state, done, alpha=0.0, beta=0.0):
        """Update the agent based on the taken step.
        Save experience in replay memory.
        Use random sample from buffer to learn.
        
        Params
        =====
            alpha (float): the alpha parameter used in importance sampling
            beta (float): the beta parameter used in importance sampling
        """                
        # Choose an error for these experiences
        if self.prioritized_replay:
            # Set the error for this experience to be equal to the highest error in the replay buffer.
            error = self.memory.get_highest_error()
        else:
            error = None
            
        # Save experiences into replay buffer, along their errors
        for s, a, r, n_s, d in zip(state, action, reward, next_state, done):
            self.memory.add(s, a, r, n_s, d, error)
                    
        # Learn every train_every time steps.
        self.t_step = (self.t_step + 1) % self.train_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                # learn for train_steps times
                for _ in range(self.train_steps):
                    experiences, indices, priorities = self.memory.sample(alpha=alpha)
                    self.learn(experiences, indices, priorities, GAMMA, beta=beta)

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
            beta (float): the beta parameter used in importance sampling
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        self.actor_target.eval()
        self.critic_target.eval()
        with torch.no_grad():
            actions_next = self.actor_target(next_states)
            Q_targets_next = self.critic_target(next_states, actions_next)
        
        # Adjust the target_best_next_action_value by importance sampling weights
        if self.prioritized_replay:
            weights = (len(self.memory) * priorities) ** -beta
            maximum_weight, _ = torch.max(weights, dim=0, keepdim=True)
            weights /= maximum_weight
            Q_targets_next *= weights

        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
    
        # Update errors of sampled experiences
        if self.prioritized_replay:
            errors = torch.abs(Q_expected - Q_targets) + EPSILON_PRIORITY
            self.memory.update_errors(indices, errors)        

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update_targets()

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
        """Initialize parameters and noise process.
        
        Params
        ======
            size (int): number of independent noise processes
            seed (int): random seed
            mu (float): mu parameter of the noise
            theta (flaot): theta parameter of the noise
            sigma (float): sigma parameter of the noise
        """        
        self.size = size
        self.mu = mu * np.ones(size)
        self.seed = random.seed(seed)
        self.reset(theta, sigma)

    def reset(self, theta=NOISE_THETA, sigma=NOISE_SIGMA):
        """Reset the internal state (= noise) to mean (mu). 
        Reset the theta and sigma parameters of the noise.
        """        
        self.state = copy.copy(self.mu)
        self.theta = theta
        self.sigma = sigma

    def sample(self):
        """Update the internal state and return it as a noise sample."""        
        x = self.state
        noise = np.random.standard_normal(self.size)
        dx = self.theta * (self.mu - x) + self.sigma * noise
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size circular buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed, prioritized_replay=False):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            prioritized_replay (bool): if True, use prioritized replay. Otherwise don't.
        """        
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.prioritized_replay = prioritized_replay

        if prioritized_replay:
            self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "error"])
        else:
            self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])           
    
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
            alpha (float): the alpha parameter used in importance sampling
        """        
        if not self.prioritized_replay:
            experiences = random.sample(self.memory, k=self.batch_size)
            indices = None
            priorities = None
        else:
            errors = np.array([x.error for x in self.memory]) ** alpha
            partition_function = np.sum(errors)
            errors_probability = errors / partition_function
            indices = random.choices(range(len(self.memory)), weights=errors_probability, k=self.batch_size)
            experiences = [self.memory[index] for index in indices]
            priorities = errors_probability[indices]

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
        """Return the highest TD-error of all experiences in the replay buffer."""
        
        if not self.prioritized_replay or len(self.memory) == 0:
            return EPSILON_PRIORITY
        else:
            return max(map((lambda x: x[-1]), self.memory))
        
    def update_errors(self, indices, errors):
        """Updates TD-errors for sampled replay buffer experiences.
        
        Params
        ======
            indices (int): replay buffer locations that need to be updated
            errors (float): new values for the error field of the replay buffer locations
        """
        if self.prioritized_replay:
            for i in range(len(indices)):
                self.memory[indices[i]] = self.memory[indices[i]]._replace(error=errors[i].item())

    def __len__(self):
        """Return the current size of internal memory."""
        
        return len(self.memory)