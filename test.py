from rl.utils import Scaler, OrnsteinUhlenbeckActionNoise, ReplayBuffer
from rl.gym_env.buck import Buck_Converter_n
from rl.agents.ddpg import ActorNetwork, CriticNetwork, train

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

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

test_env = Buck_Converter_n(Vs = 400, L = 1, C = 1, R = 1, G = 0.1, Vdes = 230, dt = 1e-2)
env = Buck_Converter_n(Vs = 400, L = 1, C = 1, R = 1, G = 0.1, Vdes = 230, dt = 1e-2)

test_s = test_env.reset()
test_obs=[]
test_steps = 10**2
test_episodes = 200
for _ in range(test_episodes):
    u = np.random.uniform(-1,1)
    for _ in range(test_steps):
        s, _,_,_ = test_env.step(u)
        test_obs.append(s)
scaler = Scaler(2)
scaler.update(np.concatenate(test_obs).reshape((test_steps*test_episodes,env.observation_space.shape[0])))
var, mean = scaler.get()
print(var, mean)
obs = []
for _ in range(200):
    s, _, _, _ = test_env.step(0.4)
    s_scaled = np.float32((s - mean) * var)
    obs.append(s_scaled)


#plt.plot(np.concatenate(obs).reshape((200,env.observation_space.shape[0])))
#plt.show()

##-----------------------------------------------------------
args = {
    'summary_dir' : './results',
    'buffer_size' : 1000000,
    'random_seed' : 1754,
    'max_episodes': 100,
    'max_episode_len' : 600,
    'mini_batch_size': 200,
    'actor_lr':0.0001,
    'critic_lr':0.001,
    'tau':0.001,
    'state_dim':env.observation_space.shape[0],
    'action_dim':env.action_space.shape[0],
    'action_bound':env.action_space.high,
    'gamma':0.99
}
state = env.reset()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high

actor = ActorNetwork(
    state_dim = args['state_dim'],
    action_dim = args['action_dim'], 
    action_bound=args['action_bound'], 
    learning_rate = args['actor_lr'], 
    tau = args['tau'], 
    batch_size = args['mini_batch_size']
    )
critic = CriticNetwork(
    state_dim = args['state_dim'], 
    action_dim = args['action_dim'], 
    action_bound = args['action_bound'], 
    learning_rate = args['critic_lr'], 
    tau = args['tau'], 
    gamma = args['gamma']
    )
replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))
actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))
reward_result = np.zeros(2500)
paths, reward_result = train(env, test_env, args, actor, critic, actor_noise, reward_result, scaler, replay_buffer)

savemat('data_' + datetime.datetime.now().strftime("%y-%m-%d-%H-%M") + '.mat',dict(data=paths, reward=reward_result))
