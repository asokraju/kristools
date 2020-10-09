##----------------------------------------------------------------------------------------------------------------------------
##-------------------------------------------Incorporaating Dissipativity---------------------------------------------------
##----------------------------------------------------------------------------------------------------------------------------
##---------------------------------------------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
import argparse
from tensorflow import keras
from tensorflow.keras import Input, Model, Sequential, layers
import datetime
from scipy.io import savemat

from cvxopt import matrix
from cvxopt import solvers
def cbf_buck(env, u_rl, h_cbf = 0, eta_1 = 0):
    
    I_d = env.Ides
    V_d = env.Vdes
    u_d = env.udes
    #eta_1 = 0.5 

    P = matrix(np.diag([1.0, 1e24]), tc='d')
    q = matrix(np.zeros(2))
    delta = env.T

    I = env.state[0]
    V = env.state[1]

    #c = 0 #env.R*((I-I_d)**2) + env.G*((V-V_d)**2) +eta_1*h
    c = - (u_rl -u_d)*(I-V_d)+eta_1*h_cbf
    #print('c', c)
    G = np.array([[I-V_d, -1], [-1, 0], [1, 0]])
    G = matrix(G,tc='d')

    h = np.array([c, u_rl, 1-u_rl])
    #print(h)
    h = np.squeeze(h).astype(np.double)
    h = matrix(h,tc='d')

    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h)

    u_bar = sol['x']
    # if np.abs(u_bar[1]) > 0.001:
    #     print("Violation of Safety: ")
    #     print(u_bar[1])
    return u_bar[0]


def cbf_microgrid(env, a_nodes, h_cbf = 0, eta_1 = 0):
    

    I_d = env.Ides
    V_d = env.Vdes
    u_d = env.udes
    delta = env.T
    a_cbf_nodes = []
    for node in range(4):
        node_state, node_reward = env.get_node(node)

        I = node_state[0]
        V = node_state[1]

        P = matrix(np.diag([1.0, 1e24]), tc='d')
        q = matrix(np.zeros(2))
        delta = env.T


        a_rl = np.clip(a_nodes[node], -1, 1)
        u_rl = (a_rl +1)/2
        #c = 0 #env.R*((I-I_d)**2) + env.G*((V-V_d)**2) +eta_1*h
        c = - (u_rl -u_d[node])*(I-V_d[node])
        #print('c', c)
        G = np.array([[I-V_d[node], -1], [-1, 0], [1, 0]])
        G = matrix(G,tc='d')

        h = np.array([c, u_rl, 1-u_rl])
        #print(h)
        h = np.squeeze(h).astype(np.double)
        h = matrix(h,tc='d')

        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h)

        u_bar = sol['x']
        du = u_bar[0]
        u_cbf = u_rl + du
        a_cbf = 2*u_cbf - 1
        #print('a_cbf', a_cbf, env.action_des[node])
        a_cbf = np.clip(a_cbf, -1, 1)
        a_cbf_nodes.append(a_cbf)

    return a_cbf_nodes

def train_buck_dcbf(env, test_env, args, actor, critic, actor_noise, reward_result, scaler, replay_buffer, cbf_buck):
    writer = tf.summary.create_file_writer(logdir = args['summary_dir'])
    actor.update_target_network()
    critic.update_target_network()
    time_steps = args['time_steps']


    paths = list()

    for i in range(args['max_episodes']):
        test_env.reset()
        s = env.reset()
        ep_reward = 0
        ep_ave_max_q = 0
        obs, obs_scaled, actions, rewards = [], [], [], []
        if args['scaling']:
            var, mean = scaler.get()
        else:
            var, mean = 1.0, 0.0
        # if i<10:
        #     temp_a = np.random.normal(0, 0.01)
        for _ in range(args['time_steps']-1):
            s_scaled = np.float32((s - mean) * var)
            obs_scaled.append(s_scaled)
            obs.append(s)
            s, r, terminal, info = env.step(np.array([env.action_des], dtype="float32"))
            actions.append(np.array([env.action_des]))

        s_scaled = np.float32((s - mean) * var)
        obs_scaled.append(s_scaled)
        obs.append(s)   
        actions.append(np.array([env.action_des]))
        for j in range(args['max_episode_len']):

            
            S_0 = obs_scaled[-args['time_steps']: ]
            
            #noise annealing
            noise = np.random.normal(0, args['noise_var']/((j/100)+1))
            a = actor.predict(np.reshape(S_0, (1, args['time_steps'], args['state_dim']))) + noise

            # action should be inbetween -1 and 1
            #print(a)
            a_rl = np.clip(a, -args['action_bound'], args['action_bound'])
            print('a_rl', a_rl)
            u_rl = (a_rl[0] +1)/2
            #print('u_rl', u_rl)
            du = cbf_buck(env, u_rl, h_cbf = 0, eta_1 = 0)
            u_cbf = u_rl + du
            a_cbf = 2*u_cbf - 1
            print('a_cbf', a_cbf, env.action_des)
            a_cbf = np.clip(a_cbf, -args['action_bound'], args['action_bound'])
            s2, r, terminal, info = env.step(a_cbf)
            s2_scaled = np.float32((s2 - mean) * var)

            #noise annealing
            obs_scaled.append(s2_scaled)
            S_2 = obs_scaled[-args['time_steps']: ]

            replay_buffer.add(S_0, np.reshape(a_cbf, (actor.action_dim,)), r, terminal, S_2)
            if replay_buffer.size() > args['mini_batch_size']:
                s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(args['mini_batch_size'])

                target_q = np.array(critic.predict_target(s2_batch, actor.predict_target(s2_batch)))
                y = []
                for k in range(args['mini_batch_size']):
                    if t_batch[k]:
                        y.append(r_batch[k])
                    else:
                        y.append(r_batch[k] + critic.gamma * np.array(target_q[k]))

                temp = np.reshape(y, (args['mini_batch_size'], 1))

                predicted_q_value = critic.train(s_batch, a_batch, temp.astype('float32'))

                ep_ave_max_q += np.amax(predicted_q_value)

                a_outs = actor.predict(s_batch)
                grads = critic.action_gradient(s_batch, a_outs)
                
                actor.train(s_batch, grads)
                actor.update_target_network()
                critic.update_target_network()

            s = s2
            ep_reward += r
            obs.append(s)
            actions.append(a[0])
            rewards.append(r)

            if j+1 == args['max_episode_len']:
                with writer.as_default():
                    tf.summary.scalar("Reward", ep_reward, step = i)
                    tf.summary.scalar("Qmax Value", ep_ave_max_q / float(j), step = i)
                    writer.flush()
                print('| Reward: {:.4f} | Episode: {:d} | Qmax: {:.4f}'.format((ep_reward), i, (ep_ave_max_q / float(j))))
                reward_result[i] = ep_reward
                path = {
                    "Observation":np.concatenate(obs).reshape((args['max_episode_len']+args['time_steps'],2)), 
                    "Action":np.concatenate(actions), 
                    "Reward":np.asarray(rewards)
                    }
                paths.append(path)
                #env.plot()
                #test_s = test_env.reset()
                # if i+1 == args['max_episodes']:
                #     env.plot()
                #     test_s = test_env.reset()
                #     for _ in range(1000):
                #         test_s_scaled = np.float32((test_s - mean) * var) 
                #         test_a = actor.predict(np.reshape(test_s_scaled,(1,actor.state_dim)))
                #         test_s, r, terminal, info = test_env.step(test_a[0])
                #     test_env.plot()
                break
    return [paths, reward_result] 

##----------------------------------------------------------------------------------------------------------------------------
##-------------------------------------------MULTI AGENT---------------------------------------------------
##----------------------------------------------------------------------------------------------------------------------------

def train_multi_agent_dcbf(env, test_env, args, actors, critics, reward_result, scaler, replay_buffers, save_weights, cbf_microgrid):
    
    # Needs 'ReplayBuffer' class, and 'save_weights' function

    writer = tf.summary.create_file_writer(logdir = args['summary_dir'])

    nodes = 4
    
    
    paths = list()

    # Initialize target network weights
    for node in range(nodes):
        actors[node].update_target_network()
        critics[node].update_target_network()


    for i in range(args['max_episodes']):
        #resetting the environments
        test_env.reset()
        env.reset()

        
        ep_reward = 0
        ep_ave_max_q = 0

        #initializing the lists
        obs, obs_scaled, actions, rewards = [[] for _ in range(nodes)], [[] for _ in range(nodes)], [[] for _ in range(nodes)], [[] for _ in range(nodes)]

        #generating T initial states, to use them in RNN's, and appending them to the lists        
        a = env.action_des
        for steps in range(args['time_steps']+1):
            for node in range(nodes):
                node_state, node_reward = env.get_node(node)
                if args['scaling']:
                    var, mean = scaler[node].get()
                else:
                    var, mean = 1.0, 0.0
                obs[node].append(np.float32(node_state))
                obs_scaled[node].append(np.float32((node_state - mean) * var))
                actions[node].append([a[node]])
                rewards[node].append(node_reward)
            if steps < args['time_steps']:
                env.step(a)
        if i%10==0:
            save_weights(actors, critics)
        #running the episode
        for j in range(args['max_episode_len']):

            #Compute the exploration noise annealing
            noise = np.random.normal(0, args['noise_var']/((j/100)+1))

            #compute the actions for each node and collect below
            a_nodes = []
            for node in range(nodes):
                #first we collect the last #time_steps states of a node
                S_0 = obs_scaled[node][-args['time_steps']: ]
                #using this state we predict the new action and add the exploration noise
                a_node = actors[node].predict(np.reshape(S_0, (1, args['time_steps'], args['state_dim']))) + noise
                # action should be inbetween -1 and 1, so we clip it
                a_node = np.clip(a_node[0], -args['action_bound'], args['action_bound'])
                a_nodes.append(a_node[0])
            
            a_cbf = cbf_microgrid(env, a_nodes, h_cbf = 0, eta_1 = 0)

            #we take step using this action
            #print('a_cbf', a_cbf, env.action_des)
            s2, r, terminal, info = env.step(np.array(a_cbf))

            #collect reward of each and append the new state (scaled) into the buffer
            node_rewards = []
            for node in range(nodes):
                node_state, node_reward = env.get_node(node)
                if args['scaling']:
                    var, mean = scaler[node].get()
                else:
                    var, mean = 1.0, 0.0
                obs[node].append(np.float32(node_state))
                obs_scaled[node].append(np.float32((node_state - mean) * var))
                actions[node].append(a_cbf[node])
                rewards[node].append(node_reward)
                node_rewards.append(node_reward)

            #for each node -- add the current state, action, reward, terminal and the previous state to the replay buffer
            for node in range(nodes):
                S_0 = obs_scaled[node][-args['time_steps']-1: -1]
                S_1 = obs_scaled[node][-args['time_steps']: ]
                replay_buffers[node].add(S_0, np.reshape(a_cbf[node], (actors[node].action_dim,)), node_rewards[node], terminal, S_1)
            
            for node, replay_buffer, critic, actor in zip(range(nodes), replay_buffers, critics, actors):
                if replay_buffer.size() >= int(args['mini_batch_size']):
                    s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(int(args['mini_batch_size']))

                    target_q = np.array(critic.predict_target(s2_batch, actor.predict_target(s2_batch)))
                    y = []
                    for k in range(args['mini_batch_size']):
                        if t_batch[k]:
                            y.append(r_batch[k])
                        else:
                            y.append(r_batch[k] + critic.gamma * np.array(target_q[k]))

                    temp = np.reshape(y, (args['mini_batch_size'], 1))

                    predicted_q_value = critic.train(s_batch, a_batch, temp.astype('float32'))

                    ep_ave_max_q += np.amax(predicted_q_value)

                    a_outs = actor.predict(s_batch)
                    grads = critic.action_gradient(s_batch, a_outs)
                    
                    actor.train(s_batch, grads)
                    actor.update_target_network()
                    critic.update_target_network()

            ep_reward += r

            #print(i, j)
            #print(np.shape(obs), obs)

            if j+1 == args['max_episode_len']:
                with writer.as_default():
                    tf.summary.scalar("Reward", ep_reward, step = i)
                    tf.summary.scalar("Qmax Value", ep_ave_max_q / float(j), step = i)
                    writer.flush()
                print('| Reward: {:.4f} | Episode: {:d} | Qmax: {:.4f}'.format((ep_reward), i, (ep_ave_max_q / float(j))))
                reward_result[i] = ep_reward
                obs_concat = []
                for node in range(nodes):
                    obs_concat.append(np.concatenate(obs[node]).reshape((args['max_episode_len']+args['time_steps']+1,2)))
                    #print(np.shape(obs_concat), obs_concat)

                path = {
                    "Observation":obs_concat, 
                    "Action":np.concatenate(actions), 
                    "Reward":np.asarray(rewards)
                    }
                paths.append(path)
                #env.plot()
                #test_s = test_env.reset()
                # if i+1 == args['max_episodes']:
                #     env.plot()
                #     test_s = test_env.reset()
                #     for _ in range(1000):
                #         test_s_scaled = np.float32((test_s - mean) * var) 
                #         test_a = actor.predict(np.reshape(test_s_scaled,(1,actor.state_dim)))
                #         test_s, r, terminal, info = test_env.step(test_a[0])
                #     test_env.plot()
                break
    return [paths, reward_result] 