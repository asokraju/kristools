# PyPI packages
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

# local modules
from rl.utils import Scaler, OrnsteinUhlenbeckActionNoise, ReplayBuffer
from rl.gym_env.buck import Buck_Converter_n
from rl.agents.ddpg import ActorNetwork_rnn, CriticNetwork_rnn, train_rnn
#---------------------------------------------------------------------
#loading the environment to get it default params
env = Buck_Converter_n()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high
#--------------------------------------------------------------------
args = {
    'summary_dir' : './results',
    'use_gpu': True,
    'buffer_size' : 1000000,
    'random_seed' : 1754,
    'max_episodes': 100,
    'max_episode_len' : 600,
    'mini_batch_size': 200,
    'actor_lr':0.0001,
    'critic_lr':0.001,
    'tau':0.001,
    'state_dim':state_dim,
    'action_dim':action_dim,
    'action_bound':action_bound,
    'gamma':0.999,
    'actor_rnn':100,
    'actor_l1':200,
    'actor_l2':100,
    'critic_rnn':100,
    'critic_l1':200,
    'critic_l2':100,
    'discretization_time': 1e-3,
    'noise_var':0.0925,
    'scaling': True,
    'save_model':True,
    'load_model':True,
    'time_steps':5
}
#---------------------------------------------------------------------------
#use GPU
if args['use_gpu']:
    physical_devices = tf.config.list_physical_devices('GPU') 
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


test_env = Buck_Converter_n(Vs = 400, L = 1, C = 1, R = 1, G = 0.1, Vdes = 230, dt = args['discretization_time'])
env = Buck_Converter_n(Vs = 400, L = 1, C = 1, R = 1, G = 0.1, Vdes = 230, dt = args['discretization_time'])

test_s = test_env.reset()
test_obs=[]
test_steps = int(1/args['discretization_time'])
test_episodes = 2000
for _ in range(test_episodes):
    u = np.random.uniform(-1,1)
    for _ in range(test_steps):
        s, _,_,_ = test_env.step(u)
        test_obs.append(s)
scaler = Scaler(2)
scaler.update(np.concatenate(test_obs).reshape((test_steps*test_episodes,env.observation_space.shape[0])))
var, mean = scaler.get()
print(var, mean)
# obs = []
# for _ in range(200):
#     s, _, _, _ = test_env.step(0.4)
#     s_scaled = np.float32((s - mean) * var)
#     obs.append(s_scaled)
#plt.plot(np.concatenate(obs).reshape((200,env.observation_space.shape[0])))
#plt.show()

##-----------------------------------------------------------
state = env.reset()

actor = ActorNetwork_rnn(
    state_dim = args['state_dim'],
    action_dim = args['action_dim'], 
    action_bound=args['action_bound'], 
    learning_rate = args['actor_lr'], 
    tau = args['tau'], 
    batch_size = args['mini_batch_size'],
    params_rnn = args['actor_rnn'],
    params_l1 = args['actor_l1'],
    params_l2 = args['actor_l2'],
    time_steps = args['time_steps']
    )
critic = CriticNetwork_rnn(
    state_dim = args['state_dim'], 
    action_dim = args['action_dim'], 
    action_bound = args['action_bound'], 
    learning_rate = args['critic_lr'], 
    tau = args['tau'], 
    gamma = args['gamma'],
    params_rnn = args['critic_rnn'],
    params_l1 = args['critic_l1'],
    params_l2 = args['critic_l2'],
    time_steps = args['time_steps']
    )

#loading the weights
if args['load_model']:
    if os.path.isfile(args['summary_dir'] + "/actor_weights.h5"):
        print('loading actor weights')
        actor.actor_model.load_weights(args['summary_dir'] + "/actor_weights.h5")
    if os.path.isfile(args['summary_dir'] + "/target_actor_weights.h5"):
        print('loading actor target weights')
        actor.actor_model.load_weights(args['summary_dir'] + "/target_actor_weights.h5")
    if os.path.isfile(args['summary_dir'] + "/critic_weights.h5"):
        print('loading critic weights')
        critic.critic_model.load_weights(args['summary_dir'] + "/critic_weights.h5")
    if os.path.isfile(args['summary_dir'] + "/target_critic_weights.h5"):
        print('loading critic target weights')
        critic.critic_model.load_weights(args['summary_dir'] + "/target_critic_weights.h5")

replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))
actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))
reward_result = np.zeros(2500)
paths, reward_result = train_rnn(env, test_env, args, actor, critic, actor_noise, reward_result, scaler, replay_buffer)

# Saving the weights
if args['save_model']:
    actor.actor_model.save_weights(
        filepath = args['summary_dir'] + "/actor_weights.h5",
        overwrite=True,
        save_format='h5')
    actor.target_actor_model.save_weights(
        filepath = args['summary_dir'] + "/target_actor_weights.h5",
        overwrite=True,
        save_format='h5')
    critic.critic_model.save_weights(
        filepath = args['summary_dir'] + "/critic_weights.h5",
        overwrite=True,
        save_format='h5')
    critic.target_critic_model.save_weights(
        filepath = args['summary_dir'] + "/target_critic_weights.h5",
        overwrite=True,
        save_format='h5')



#Plotting an new testing environment
if args['scaling']:
    var, mean = scaler.get()
else:
    var, mean = 1.0, 0.0

obs_scaled, obs, actions = [], [], []
test_s = test_env.reset()
for _ in range(args['time_steps']-1):
    s_scaled = np.float32((test_s - mean) * var)
    obs_scaled.append(s_scaled)
    obs.append(test_s)
    test_s, r, terminal, info = test_env.step(np.array([test_env.action_des], dtype="float32"))
    actions.append([env.action_des])

s_scaled = np.float32((test_s - mean) * var)
obs_scaled.append(s_scaled)
obs.append(test_s)   
actions.append([env.action_des])
for _ in range(1000):
    S_0 = obs_scaled[-args['time_steps']:]
    test_a = actor.predict(np.reshape(S_0, (1, args['time_steps'], args['state_dim'])))
    test_s, r, terminal, info = test_env.step(test_a[0])
    s2_scaled = np.float32((test_s - mean) * var)
    obs_scaled.append(s2_scaled)
test_env.plot()

savemat('data_' + datetime.datetime.now().strftime("%y-%m-%d-%H-%M") + '.mat',dict(data=paths, reward=reward_result))

