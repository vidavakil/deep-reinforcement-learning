# Project Description

The goal of this project is to train an agent to navigate and collect bananas in a large, square world. 

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

[//]: # (Image References)

[image0]: ./DQN_algorithm.png
[image1]: ./plot_of_rewards.png


# Description of the Implementation

The main algorithm used here for training the banana-collecting agent is Deep Q-Learning, using a DQN (Deep Q-Network). Please refer to the original [paper](htps://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) by DeepMind for further details.

DQN uses a Deep Neural Network to learn the optimal action-value function for Q-Learning. Recall that in Q-Learning, the update rule for the action-value of a state/action pair ```(s, a)``` that produces a reward ```r``` and a next state ```s'``` is given by the following formula, where ```0 < alpha <= 1``` is used to smooth the learning process. 

```Q(s, a) = (1 - alpha) Q(s, a) + alpha * (r + gamma * max(a)(Q(s', a))```

Q-Learning uses a table to represent the action-value function. This is useful for problems with discrete state/action spaces where the state/action space sizes are not too large. As for problems with continuous state/action spaces, using the tabular form of Q-Learning requires discretizing the state/action space that in turn may produce nevertheless large (or still infinite) space sizes. DQN uses a deep network, instead of a table, to represent and estimate the Q function. 

The input to the DQN network is the state of the environment ```s```, and the network produces one output for each possible action ```a```, representing the estimated value of ```Q(s, a)```. An epsilon-greedy policy is used to take actions, where with a probability of ```(1 - epsilon)``` the action with the highest value is taken, and otherwise a random action is taken to provide for exploration versus exploitation. 

Training DQNs with the same algorithm that is used for the tabular form of Q-Learning would be notoriously unstable. The following describes the main sources of instability if we simply replaced the Q-table with a deep network with no other modification to the learning algorithm.

Instability due to having a moving target: With the tabular form of Q-Learning, an epsilon-greedy algorithm using the current Q-table is used to collect experiences of the form ```(s, a, r, s’, done)``` and then use the TD error ```((r + gamma * max(a)(Q(s', a)) - Q(s, a))``` to update ```Q(s, a)``` (```gamma``` is the discount factor discounting future rewards). Such update does not affect other entries of the Q-table. In contrast, updating a DQN for a single ```(s, a)``` pair will change the value of all state/action pairs, making the target in the update rule a moving target, and thus making the training process unstable. 

Instability due to correlation between experiences within an episode: Because a DQN is a deep network trained with SGD, we need random, uncorrelated, and independent mini-batches of experiences sampled according to the policy over all states and actions for the SGD to work properly. When mini-batches of experiences are coming from sequentially running the current policy, they will be highly correlated and biased, leading to convergence problems in the training. 

To solve these issues, DQN uses two key features: a target network, and a replay buffer.

- The target network is a less frequently updated version of the trained network. Because it is less frequently updated, it provides for a more stable target and thus more stable training.

- The replay buffer is a circular buffer that holds the latest ```N``` experience tuples. The DQN algorithm has two phases. In the algorithm’s sampling phase, an epsilon-greedy policy using the local Q-network is run to collect experience tuples that are added to the end of the replay buffer. In the algorithm’s learning phase, mini-batches of experiences are randomly sampled from the replay buffer for training the network, thus addressing the issue of correlation among the experiences in a mini-batch. Using the replay buffer the network can also learn from its past experiences many times before they get discarded.

The following shows a description of the DQN algorithm.

![alt text][image0]

# Hyperparameters

The DQN algorithm used and implemented for this project has a number of hyper-parameters that can be tuned for faster convergence and more stable training. These include the following:

- ```eps``` (epsilon parameter of the epsilon-greedy policy) and how it is decayed as the training makes progress: starting with ```eps_start```, ```eps``` is multiplicatively decayed after each episode by ```eps_decay``` until it reaches ```eps_end```. This schedule encourages more exploration in the early steps of training and more exploitation later on
- ```BUFFER_SIZE```: the size of the circular replay buffer
- ```BATCH_SIZE```: the mini-batch size used in training the local Q-network
- ```UPDATE_EVERY```: the number of experiences that are explored in the sampling phase and added to the replay buffer before switching to the learning phase 
One may use another hyper-parameter for the number of training steps in each learning phase. For this project, each learning phase performs a single step of training. That is, a single mini-batch of experiences randomly sampled from the replay buffer is used for one step of training the local Q-network and immediately updating the target network using a soft-update algorithm.
- ```TAU```: used for updating the target Q-Network as a weighted average of the old target Q-network and the updated local Q-network. 
```target_network_parameters = TAU * local_network_parameters + (1.0 - TAU) * target_network_parameters```
- ```GAMMA```: the discount factor in the update rule of Q-Learning that discounts future rewards
- ```LR```: the learning rate used for training the local Q-network

Using the following values for the above hyperparameters, an average score of 13.52 (over the latest 100 episodes) was achieved after 394 episodes of training (an episode ends when the environment returns a True value in response to the last action taken by the agent).

```
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 1e-3
LR = 5e-4
UPDATE_EVERY = 4
```

# Model Architecture

The network architecture used for the local/target Q-networks is a fully connected network with two hidden layers. The first fully connected layer has 256 units, and the second layer has 128 units. ReLU activation functions are used after each hidden layer, followed by a DropOut layer with a dropout rate of 0.2. The output layer of the network is a fully connected layer with 4 units, producing one output per each of the four possible actions.


# Training Results

The following shows the plot of rewards per episode of training the agent with a goal of reaching an average score of 13.5. 
An average score of 13.52 (over the latest 100 episodes) was achieved after 394 episodes.
A model checkpoint is available in file ```checkpoint.pth```.

![alt text][image1]

# Ideas for Future Work

The following are two of the suggested improvements to the DQN algorithm:

Double DQN (DDQN): addresses DQN's issue of over-approximation of the action-value function via using two networks to compute the target value. The local network is used to compute the action ```a’``` that has the maximum estimated value at the next state ```s’```, while a separate network (e.g, the target network already used in DQN) is used to compute ```Q(s’, a’)```. 

Prioritized Experience Replay: Allows for important experiences/transitions to be sampled with higher probability from the replay buffer which is otherwise sampled uniformly. For example, one can use the TD-errors of experiences to prioritize them for sampling, thus allowing experiences that had larger errors to be sampled with higher probability from the replay buffer, and used more often for training, and thus better utilized. This algorithm has two additional hyper parameters, ```alpha``` and ```beta```. ```alpha``` controls the amount of prioritization (versus random sampling), and ```beta``` controls another term that adjusts for sampling from the prioritized distribution instead of the real distribution. These parameters themselves are typically starting with zero and slowly increased towards 1.0.

For this project, both of the above algorithms were implemented (included in the code). However, the vanilla DQN was already able to solve the problem with a bit of hyperparameter tuning. Beating the performance of the vanilla DQN using the above two algorithms requires further tuning of the hyperparameters of these algorithms, and is left as future work.
