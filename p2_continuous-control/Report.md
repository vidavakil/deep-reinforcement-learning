# Project Description

The goal of this project is to train a double-jointed arm in the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

In this environment, the double-jointed arm can move to target locations. A reward is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible. The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to the torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The version of the Unity environment used for this project is a distributed learning environment containing 20 identical agents, each with its own copy of the environment.

[//]: # (Image References)

[image0]: ./DDPG_algorithm.png
[image1]: ./plot_of_rewards.png


# Learning Algorithm

The main algorithm used here for training the given agent is DDPG. DDPG extends the use of Deep Q-Learning to the continuous action domain, which is what we need for training the Reacher agent. Please refer to the original [paper](https://arxiv.org/abs/1509.02971) by DeepMind for details of DDPG. 

DDPG is an “Actor-Critic” method. It uses two networks. The actor network is a policy network, that learns the best (optimal) deterministic continuous action for any given state. The critic network learns the action-value function of the policy; i.e., given a state/action pair it learns to estimate ```Q(s, a)```. Similar to DQN, there are two copies of each of the actor and critic networks: a local network, and a target network which is a less frequently updated version of the local network. The use of target networks help stabilize the The local actor (policy) network is used to act in the environment and collect experiences that are stored in a replay buffer, which are used many times for off-line training, similar to the DQN algorithm. In the learning phase of the algorithm, random batches of  experiences from the replay buffer are used to update the local critic network, much like in DQN. That is, for each experience ```(s, a, r, s’, done)```, the TD-error of the experience as shown below is used to train the local critic (using SGD and by minimizing the TD-error over batch of experiences):
```
TD-error = Q(s, a) - (r + gamma * max(a’)(Q(s’, a’)))
TD-error = local_critic(s, a) - (r + gamma * target_critic(s’, target_actor(s’)))
```
The local critic is in turn used to provide estimated values for actions proposed by the local actor, which are then used to push the local actor to producing higher value actions. This is done by back propagating the local-critic estimates of the values of actions proposed by the local actor through the local critic and back to the local actor. The local actor and critic networks are then used to update their corresponding target networks using a soft-update algorithm.  

The following shows a description of the original DDPG algorithm.

![alt text][image0]

DDPG combines a policy-based algorithm and a value-based algorithm. Policy-based algorithms alone (such as Policy Gradient) are high variance, but low bias; since they use actual rollouts of experiences they are low bias, but different rollouts can result different results and hence the high variance. Value-based only methods (such as DQN) on the other hand are low variance, but high bias; since they use an estimate of the value of the next state that is more stable they have low variance, but because of using such an estimate, they are high bias. Since DDPG uses a combination of the two, it helps better balance both bias and variance. 

For this project, we have access to 20 agents that share the same policy, and can independently act in the environment (they are not interacting with each other), and collect experiences that are then added to a shared replay buffer. This helps experience and collect in parallel many more experiences than a single agent could in the same amount of time. Following ```train_every``` number of steps of the parallel agents acting and collecting experiences, a learning phase starts in which for ```train_steps``` number of times randomly selected batches of episodes are sampled from the replay buffer and used to update the local actor and critic networks. After each such update, the corresponding target networks are updated using a soft-update algorithm controlled by a parameter ```tau```. 

Since the policy is deterministic, in order to help the agent explore the environment, noise is added to the actions proposed by the actor/policy network. The Ornstein-Uhlenbeck process used to generate the noise is controlled by two parameters ```noise_theta``` and ```noise_sigma```, and each of the 20 agents uses its own noise process that are all reset at the end of each episode. To control the amount of exploration as the training makes progress, the two parameters of the noise process can be decayed using two other hyperparameters, ```theta_decay``` and ```sigma_decay```, upon reset of the noise process.


# Model Architecture

The network architecture used for the local/target actor is a fully connected network with two hidden layers. It takes as input a real valued state vector of length 33, and generates 4 real values, one for each of the four actions required by the agent. The first fully connected layer has 256 units, and the second layer has 128 units. ```Leaky_ReLU``` activation functions are used after each hidden layer. The output layer of the network is a fully connected layer with 4 units each with a ```Tanh``` activation function, producing one continuous value in the range of (-1, +1) for each of the four actions supported by the agent.

The network architecture used for the local/target critic is a fully connected network with two hidden layers. The network takes as input the real valued state vector of length 33 that it feeds to its first fully connected layer that has 256 units. It also takes a real valued action vector of length 4 that is fed, together with the output of the first layer, to the second fully connected layer of the network that has 128 units. Both hidden layers use ```Leaky-ReLU``` activation functions. The output layer of the network is a fully connected layer with a single unit (and no activation function) that generates a real value for the given input state/action pair.


# Hyperparameters

The DDPG algorithm used and implemented for this project has a number of hyper-parameters that can be tuned for faster convergence and more stable training. These include the following:

- ```train_every```: the number of experiences that are explored in the sampling phase and added to the replay buffer before switching to the learning phase. 
- ```train_steps```: the number of training steps in each learning phase. In each training step, a single mini-batch of experiences randomly sampled from the replay buffer is used for one step of training the local networks and immediately updating the target networks using a soft-update algorithm. 
- ```buffer_size```: the size of the circular replay buffer
- ```batch_size```: the mini-batch size used in training the local networks
- ```lr_actor```: the learning rate used for training the local actor network
- ```lr_critic```: the learning rate used for training the local critic network
- ```weight_decay```: the L2 regularization parameter for critic’s network
- ```gamma```: the discount factor in the update rule of the critic (Q-Learning) that discounts future rewards
- ```tau```: used for updating each of the two actor and critic target networks as a weighted average of the old target network and the updated local network.
```target_network_parameters = TAU * - local_network_parameters + (1.0 - TAU) * target_network_parameters```
- ```noise_theta```, ```noise_sigma```: ```theta``` and ```sigma``` parameters of the Ornstein-Uhlenbeck process
- ```sigma_decay```, ```theta_decay```: used to multiplicatively decay the corresponding noise parameters at the end of each collective episode, to reduce the amount of exploration over the course of training

Using the following values for the above hyperparameters, an average score of 30.099 (over the latest 100 episodes) was achieved after 173 episodes of training (an episode ends when the environment returns a done = True in response to the last action taken by an agent).
```
train_every = 20
train_steps = 10 
buffer_size = int(1e6)
batch_size = 512
lr_actor = 1e-4
lr_critic = 1e-3
weight_decay = 0.0
gamma = 0.99
tau = 1e-3
noise_theta = 0.15
noise_sigma = 0.2
theta_decay = 0.99
sigma_decay = 0.99
```


# Plot of Rewards 

After each episode, the raw rewards received by each agent are added up to compute a score for the agent. This yields 20 (potentially different) scores. The average of these 20 scores is then taken, yielding an average score for each episode (where the average is over all 20 agents).

The following shows the plot of rewards per episode of training the agents with a goal of reaching an average score of 30. An average score of 30.099 (over the latest 100 episodes) was achieved after 173 episodes. Checkpoints for the actor and critic networks are available in files ```checkpoint_actor.pth``` and ```checkpoint_critic.pth```.

![alt text][image1]


# Ideas for Future Work

Distributed Distributional Deterministic Policy Gradient ([DDDPG](https://openreview.net/forum?id=SyZipzbCb)) uses additional techniques to improve the performance of DDPG, including N-step returns, and prioritized experience replay. These techniques are described below:

Prioritized Experience Replay (PER): Allows for important experiences/transitions to be sampled with higher probability from the replay buffer which is otherwise sampled uniformly. For example, one can use the absolute value of the TD-error of experiences to prioritize them for sampling, thus allowing experiences that had larger errors to be sampled with higher probability from the replay buffer, and used more often for training, and thus better utilized. This algorithm has two additional hyper parameters,```alpha``` and ```beta```. ```alpha``` controls the amount of prioritization (versus random sampling), and ```beta``` controls another term that adjusts for sampling from the prioritized distribution instead of the real distribution. These parameters themselves are typically starting with zero and slowly increased towards 1.0.

N-step returns: the TD-error used in the vanilla DDPG algorithm (and DQN) uses a one-step return, where the immediate reward and discounted maximum action-value in the next state is used to update the action-value of the current state/action pair. In N-step return, the discounted rewards of a sequence of N steps and a discounted estimate of the action-value in the Nth subsequent state is used to update the action-value of the current state/action pair. This algorithm helps better trade off the bias and variance of the critic, as larger N reduces the bias but increases the variance. 

For this project, basic implementations of the above two algorithms were coded (included in this repo). However, the vanilla DDPG was already able to solve the problem after some hyperparameter tuning. The basic implementation of the PER present in the code has a high complexity (e.g., sampling complexity of ```O(n)```, where n is the size of the replay buffer), which significantly slows down the training, instead of providing any improvement. 

For future work, the implementation of the PER better be revised to use a sum-tree for the replay buffer as proposed in the original Prioritized Replay Buffer [paper](https://arxiv.org/pdf/1511.05952.pdf). The sum-tree data structure proposed in the above paper is similar to the array representation of a binary heap, but instead of the usual heap property where the value of a parent is the maximum value of its two children, in the sum-tree the value of a parent node is the sum of its children. Leaf nodes store the priorities and the internal nodes are intermediate sums, with the parent node containing the sum over all priorities. 

Beating the performance of the vanilla DDPG using the above two algorithms also requires tuning of the hyperparameters of these algorithms, and is left as future work.
