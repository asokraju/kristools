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

# local modules
from rl.utils import Scaler, OrnsteinUhlenbeckActionNoise, ReplayBuffer
from rl.gym_env.buck_microgrid import Buck_microgrid
from rl.agents.ddpg import ActorNetwork_rnn, CriticNetwork_rnn, train_multi_agent
from rl.agents.ddpg_cbf import cbf_microgrid, train_multi_agent_dcbf

def save_weights(actors, critics):

    #saves the weights of actors and the critics so that we can reuse them
    paths = args['summary_dir']
    for node in range(4):
        actors[node].actor_model.save_weights(
            filepath = paths + "/actor_"+ str(node) + "_weights.h5",
            overwrite=True,
            save_format='h5')
        actors[node].target_actor_model.save_weights(
            filepath = paths + "/target_actor_"+ str(node) + "_weights.h5",
            overwrite=True,
            save_format='h5')
        critics[node].critic_model.save_weights(
            filepath = paths + "/critic_"+ str(node) + "_weights.h5",
            overwrite=True,
            save_format='h5')
        critics[node].target_critic_model.save_weights(
            filepath = paths + "/target_critic_"+ str(node) + "_weights.h5",
            overwrite=True,
            save_format='h5')

def main(args, reward_result):
    #use GPU
    if args['use_gpu']:
        physical_devices = tf.config.list_physical_devices('GPU') 
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    #initalizing environment
    test_env = Buck_microgrid(dt = args['discretization_time'])
    env = Buck_microgrid(dt = args['discretization_time'])

    # #training the scales functions for each node
    scaler={
        0: (np.array([0.02971586, 0.02961047], dtype=np.float32), np.array([ 19.491829, 199.30959 ], dtype=np.float32)),
        1 : (np.array([0.02972841, 0.02960925], dtype=np.float32), np.array([ 19.562923, 199.31097 ], dtype=np.float32)),
        2 : (np.array([0.02977384, 0.02961133], dtype=np.float32), np.array([ 19.747936, 199.3163  ], dtype=np.float32)),
        3 : (np.array([0.02978811, 0.02961049], dtype=np.float32), np.array([ 19.833357, 199.3185  ], dtype=np.float32))
        }
    
    ## I am choose the same architecture for all the actors and critics
    args_actor = {'state_dim' : args['state_dim'],
    'action_dim' : args['action_dim'], 
    'action_bound':args['action_bound'], 
    'learning_rate' : args['actor_lr'], 
    'tau' : args['tau'], 
    'batch_size' : args['mini_batch_size'],
    'params_rnn' : args['actor_rnn'],
    'params_l1' : args['actor_l1'],
    'params_l2' : args['actor_l2'],
    'time_steps' : args['time_steps']}

    args_critic = {'state_dim' : args['state_dim'], 
        'action_dim' : args['action_dim'], 
        'action_bound' : args['action_bound'], 
        'learning_rate' : args['critic_lr'], 
        'tau' : args['tau'], 
        'gamma' : args['gamma'],
        'params_rnn' : args['critic_rnn'],
        'params_l1' : args['critic_l1'],
        'params_l2' : args['critic_l2'],
        'time_steps' : args['time_steps']}
    actors = [ActorNetwork_rnn(**args_actor) for _ in range(4)]
    critics = [CriticNetwork_rnn(**args_critic) for _ in range(4)]

    #loading the weights
    if args['load_model']:
        for node in range(4):
            if os.path.isfile(args['summary_dir'] + "/actor_"+ str(node) + "_weights.h5"):
                print('loading actor {} weights'.format(node+1))
                actors[node].actor_model.load_weights(args['summary_dir'] + "/actor_"+ str(node) + "_weights.h5")

            if os.path.isfile(args['summary_dir'] + "/target_actor_"+ str(node) + "_weights.h5"):
                print('loading actor {} target  weights'.format(node+1))
                actors[node].actor_model.load_weights(args['summary_dir'] + "/target_actor_weights.h5")

            if os.path.isfile(args['summary_dir'] + "/critic_"+ str(node) + "_weights.h5"):
                print('loading critic {} weights'.format(node+1))
                critics[node].critic_model.load_weights(args['summary_dir'] + "/critic_"+ str(node) + "_weights.h5")

            if os.path.isfile(args['summary_dir'] + "/target_critic_"+ str(node) + "_weights.h5"):
                print('loading critic {} target weights'.format(node+1))
                critics[node].critic_model.load_weights(args['summary_dir'] + "/target_critic_"+ str(node) + "_weights.h5")

    #replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))
    #actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(args['action_dim']))
    # Initialize replay memory
    replay_buffer1 = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))
    replay_buffer2 = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))
    replay_buffer3 = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))
    replay_buffer4 = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))
    replay_buffers = [replay_buffer1, replay_buffer2, replay_buffer3, replay_buffer4]
    #reward_result = np.zeros(2500)
    #paths, reward_result = train_multi_agent_dcbf(env, test_env, args, actors, critics, reward_result, scaler, replay_buffers, save_weights, cbf_microgrid)

    # Saving the weights
    #save_weights(actors, critics)

    #---------------------------------------------------------------------------
    #Plotting an new testing environment
    print('-'*20)
    print("starting the simulation")
    print('-'*20)
    obs, obs_scaled, = [[] for _ in range(4)], [[] for _ in range(4)]
    complete_states = []
    actions = []
    complete_state = test_env.reset()
    complete_states.append(complete_state)
    a = test_env.action_des
    for steps in range(args['time_steps']+1):
        for node in range(4):
            node_state, node_reward = test_env.get_node(node)
            if args['scaling']:
                var, mean = scaler[node]
            else:
                var, mean = 1.0, 0.0
            obs[node].append(np.float32(node_state))
            obs_scaled[node].append(np.float32((node_state - mean) * var))

        if steps < args['time_steps']:
            test_env.step(a)
            actions.append(a.tolist())
    for _t in range(int(15e4)):
        a_nodes = []
        if _t % int(1e4) == 0:
            print("time_step: ", _t)
        if _t==int(5e4):
            test_env.G = test_env.G*(1.05)
            print("increased the load by 5%")
        for node in range(4):
            #first we collect the last #time_steps states of a node
            S_0 = obs_scaled[node][-args['time_steps']: ]
            #using this state we predict the new action and add the exploration noise
            a_node = actors[node].predict(np.reshape(S_0, (1, args['time_steps'], args['state_dim'])))
            # action should be inbetween -1 and 1, so we clip it
            a_node = np.clip(a_node[0], -args['action_bound'], args['action_bound'])
            a_nodes.append(a_node[0])
        a_cbf = cbf_microgrid(env, a_nodes, h_cbf = 0, eta_1 = 0)
        #print(a_cbf)
        actions.append(a_cbf)
        #we take step using this action
        s2, r, terminal, info = test_env.step(np.array(a_cbf))
        complete_states.append(s2)
        for node in range(4):
            node_state, node_reward = test_env.get_node(node)
            if args['scaling']:
                var, mean = scaler[node]
            else:
                var, mean = 1.0, 0.0
            obs[node].append(np.float32(node_state))
            obs_scaled[node].append(np.float32((node_state - mean) * var))
    print("Saving the data")    
    savefig_filename = os.path.join(args['summary_dir'], 'test_microgrid_plot_4.png')
    test_env.plot(savefig_filename=savefig_filename)

    savemat(os.path.join(args['summary_dir'], 'save_data4.mat'), dict(data=obs, actions=actions, complete_states=complete_states))

    return [actors, critics]


#---------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')
    #loading the environment to get it default params
    env = Buck_microgrid()
    state_dim = 2#env.observation_space.shape[0]
    action_dim = 1#env.action_space.shape[0]
    action_bound = 1#env.action_space.high
    #--------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    #general params
    parser.add_argument('--summary_dir', help='directory for saving and loading model and other data', default='./Power-Converters/kristools/results')
    parser.add_argument('--use_gpu', help='weather to use gpu or not', type = bool, default=True)
    parser.add_argument('--save_model', help='Saving model from summary_dir', type = bool, default=True)
    parser.add_argument('--load_model', help='Loading model from summary_dir', type = bool, default=False)
    parser.add_argument('--random_seed', help='seeding the random number generator', default=1754)
    
    #agent params
    parser.add_argument('--buffer_size', help='replay buffer size', type = int, default=1000000)
    parser.add_argument('--max_episodes', help='max number of episodes', type = int, default=50)
    parser.add_argument('--max_episode_len', help='Number of steps per epsiode', type = int, default=1200)
    parser.add_argument('--mini_batch_size', help='sampling batch size',type =int, default=1000)
    parser.add_argument('--actor_lr', help='actor network learning rate',type =float, default=0.0001)
    parser.add_argument('--critic_lr', help='critic network learning rate',type =float, default=0.001)
    parser.add_argument('--gamma', help='models the long term returns', type =float, default=0.999)
    parser.add_argument('--noise_var', help='Variance of the exploration noise', default=0.0925)
    parser.add_argument('--scaling', help='weather to scale the states before using for training', type = bool, default=True)
    
    #model/env paramerters
    parser.add_argument('--state_dim', help='state dimension of environment', type = int, default=state_dim)
    parser.add_argument('--action_dim', help='action space dimension', type = int, default=action_dim)
    parser.add_argument('--action_bound', help='upper and lower bound of the actions', type = float, default=action_bound)
    parser.add_argument('--discretization_time', help='discretization time used for the environment ', type = float, default=1e-3)

    #Network parameters
    parser.add_argument('--time_steps', help='Number of time-steps for rnn (LSTM)', type = int, default=2)
    parser.add_argument('--actor_rnn', help='actor network rnn paramerters', type = int, default=20)
    parser.add_argument('--actor_l1', help='actor network layer 1 parameters', type = int, default=400)
    parser.add_argument('--actor_l2', help='actor network layer 2 parameters', type = int, default=300)
    parser.add_argument('--critic_rnn', help='critic network rnn parameters', type = int, default=20)
    parser.add_argument('--critic_l1', help='actor network layer 1 parameters', type = int, default=400)
    parser.add_argument('--critic_l2', help='actor network layer 2 parameters', type = int, default=300)
    parser.add_argument('--tau', help='target network learning rate', type = float, default=0.001)
    
    #args = vars(parser.parse_args())
    args = {'action_bound': 1,
    'action_dim': 1, 
    'actor_l1': 400,
    'actor_l2': 300,
    'actor_lr': 0.0001,
    'actor_rnn': 20,
    'buffer_size': 1000000,
    'critic_l1': 400,
    'critic_l2': 300,
    'critic_lr': 0.001,
    'critic_rnn': 20,
    'discretization_time': 0.001,
    'gamma': 0.99,
    'load_model': True,
    'max_episode_len': 1200,
    'max_episodes': 1,
    'mini_batch_size': 1000,
    'noise_var': 0.0925,
    'random_seed': 1754,
    'save_model': True,
    'scaling': True,
    'state_dim': 2,
    'summary_dir': './Power-Converters/kristools/results',
    'tau': 0.001,
    'time_steps': 2,
    'use_gpu': False}
    pp.pprint(args)

    reward_result = np.zeros(2500)
    [actors, critics] = main(args, reward_result)