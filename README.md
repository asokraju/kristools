# kristools

In this I impleted a custum multi-agent reinforcement learning algithm proposed in [[1]](#1). We develop a reinforcement learning based distributed control design approach that exploits the dissipativity property of individual subsystems to guarantee stability of the entire networked system. Our proposed approach can be summarized as follows. We first 
use a control barrier function(CBF) to characterize the set of controllers that enforce a dissipativity condition at each subsystem. We impose a minimal energy perturbation on the control input learned by the RL algorithm to project it to an input in this set. Together, these results guarantee the stability of the entire networked system even when the subsystems utilize potentially heterogeneous RL algorithms to design their local controllers. The over all algorithm is described as follows:

<img src="https://github.com/asokraju/kristools/blob/d23bbb49d2ac67d4750a55c728d9b631ef4633b3/results/algorithm.PNG" width="2000" align="right">

Please see [[1]](#1) for Notation. Our approach of utilizing a control barrier function (CBF) to impose the constraint that the controller designed for each subsystem using RL preserves the dissipativity of the subsystem in the closed loop parallels the use of CBFs to enforce safety in RL algorithms. CBFs guarantee the existence of control inputs under which a super-level set of a function (typically representing specifications like safety) is forward invariant under a given dynamics. However, their use to impose input-output properties such as dissipativity is less studied. Here, we utilize CBFs to characterize the set of dissipativity ensuring controllers, and then learn a dissipativity ensuring controller for each subsystem from this set.


The code repository also contains tools for implementing custor multi-agent reinforment leaning algorithms for systems described in Gym environments. 
This contains basic tools for implementing Reinforcement Learning algorithms and gym environments. Mainly aiming for systems with continious state space and action space.


## References
<a id="1">[1]</a> 
Kosaraju, K. C., Sivaranjani, S., Suttle, W., Gupta, V., & Liu, J. (2021). 
Reinforcement learning based distributed control of dissipative networked systems. 
IEEE Transactions on Control of Network Systems, 9(2), 856-866.

## gym environments:
- [DC-DC buck converter](rl/gym_env/buck.py)
- [DC-DC boost converter](rl/gym_env/boost.py)
- [four node buck (DC) microgrid](rl/gym_env/buck_microgrid.py)
## RL algorithms
- ```buck_ddpg``` run DDPG on a simple buck converter environment.
![DC-DC buck converter](results/results_plot_nice.png)

# How to use?
```python buck_ddpg --gamma=0.9 --max_episodes=100 --actor_lr=0.0001 --critic_lr=0.01 summary_dir='./results_buck_ddps'```
will run the ddpg algorithm on buck converter, with discount factor = 0.9, for 100 episodes, and actor and critic learning rates 0.0001, 0.01, respectively. Finally saves the results in the folder = './results_buck_ddps' (the folder should be available)

# Complete argument list:

Use argparse to set the parameters of the desired experiment. Running buck_ddpg.py as a script will then output the results to a named and dated directory in the results folder.

```summary_dir``` folder path to load and save the model. Saved all the results in .mat format.

```save_model``` (```bool```) if ```True``` saves the model in the ```summary_dir```

```load_model``` (```bool```) if ```True``` loads the model in the ```summary_dir```

```random_seed```  (```int```)  seeding the random number generator (NOT completely implemented)

```buffer_size``` (```int```) replay buffer size

```max_episodes``` (```int```) max number of episodes for training

```max_episode_len``` (```int```) Number of steps per epsiode

```mini_batch_size``` (```int```) sampling batch size drawn from replay buffer

```actor_lr``` (```float```) actor network learning rate

```critic_lr``` (```float```) critic network learning rate

```gamma``` (```float```) models the long term returns (discount factor)

```noise_var``` (```float```) starting variance of the exploration noise at each episode, and decreased as the episode progress

```scaling```  (```bool```) If ```True``` scales the states before using for training

```state_dim``` (```int```) state dimension of environment

```action_dim``` (```int```) action space dimension

```action_bound``` (```float```) upper and lower bound of the actions

```discretization_time``` (```float```) discretization time used for the environment

### Actor and Critic network is implemented using LSTM's + two hidden layers

```time_steps``` (```int```) Number of time-steps for rnn (LSTM)

```actor_rnn``` (```int```) actor network rnn layer paramerters

```actor_l1``` (```int```) actor network layer 1 parameters

```actor_l2``` (```int```) actor network layer 2 parameters



```critic_rnn``` (```int```) critic network rnn layer paramerters

```critic_l1``` (```int```) critic network layer 1 parameters

```critic_l2``` (```int```) critic network layer 2 parameters

```tau```  (```float```)  target network learning rate



# Dependencies

Written in TensorFlow 2.0 (Keras)

Requires the following PiPy packages
```
import matplotlib.pyplot as plt
import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model, Sequential, layers
import datetime
from scipy.io import savemat
import os
import argparse
import pprint as pp
```
