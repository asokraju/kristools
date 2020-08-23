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
# print(tf.__version__)
# tf.test.gpu_device_name()




#===================================
#     Actor and Critic DNNs
#===================================


#===================================
#     Actor and Critic DNNs
#===================================


class ActorNetwork(object):
    def __init__(self, state_dim, action_dim, action_bound, learning_rate, tau, batch_size, params_l1, params_l2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau  = tau
        self.batch_size = batch_size
        self.params_l1 = params_l1
        self.params_l2 = params_l2
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        #actor network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()
        self.actor_model = keras.Model(inputs=self.inputs, outputs=self.scaled_out, name='actor_network')
        self.network_params = self.actor_model.trainable_variables

        #target actor network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()
        self.target_actor_model = keras.Model(inputs=self.target_inputs, outputs=self.target_scaled_out, name='target_actor_network')
        self.target_network_params = self.target_actor_model.trainable_variables

        #initalizing the target params with then network params
        for i in range(len(self.target_network_params)):
            self.target_network_params[i].assign(self.network_params[i])


    def create_actor_network(self):
        inputs = Input(shape = (self.state_dim,), batch_size = None, name = "actor_input_state")
        w_init = tf.random_uniform_initializer(minval=-0.03, maxval=0.03, seed=None)

        net = layers.Dense(self.params_l1, name = 'actor_dense_1', kernel_initializer = w_init)(inputs)
        net = layers.BatchNormalization()(net)
        net = layers.Activation(activation=tf.nn.relu)(net)

        net = layers.Dense(self.params_l2, name = 'actor_dense_2', kernel_initializer = w_init)(net)
        net = layers.BatchNormalization()(net)
        net = layers.Activation(activation='tanh')(net)
        
        
        out = layers.Dense(self.action_dim, activation='tanh', name = 'actor_dense_3', kernel_initializer = w_init)(net)
        scaled_out = tf.multiply(out, self.action_bound, name = "actions_scaling")
        return inputs, out, scaled_out
  
    def update_target_network(self):
        self.update_target_network_params = [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1-self.tau)) for i in range(len(self.target_network_params))]
  
    def train(self, inputs, a_gradient):
        with tf.GradientTape(watch_accessed_variables=True) as tape:
            prediction = self.actor_model(inputs)
        unnormalized_actor_gradients = tape.gradient(prediction, self.network_params, output_gradients = -a_gradient)
        actor_gradients = list(map(lambda x: tf.math.divide(x, self.batch_size), unnormalized_actor_gradients))
        self.optimizer.apply_gradients(zip(actor_gradients, self.network_params))
    
    def predict(self, inputs):
        return self.actor_model(inputs)

    def predict_target(self, inputs):
        return self.target_actor_model(inputs)



##---------------------------------------------------------------------------------------------------------------
class CriticNetwork(object):
    def __init__(self, state_dim, action_dim, action_bound, learning_rate, tau, gamma, params_l1, params_l2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau  = tau
        self.gamma = gamma
        self.params_l1 = params_l1
        self.params_l2 = params_l2
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        #Critic Network and parameters
        self.inputs_state, self.inputs_action, self.out = self.create_critic_network()
        self.critic_model = keras.Model(inputs=[self.inputs_state, self.inputs_action], outputs=self.out, name='critic_network')
        self.network_params = self.critic_model.trainable_variables

        #Target Critic Network and parameters
        self.target_inputs_state, self.target_inputs_action, self.target_out = self.create_critic_network()
        self.target_critic_model = keras.Model(inputs=[self.target_inputs_state, self.target_inputs_action], outputs=self.target_out, name='target_critic_network')
        self.target_network_params = self.target_critic_model.trainable_variables
        
        #initalizing the target params with then network params
        for i in range(len(self.target_network_params)):
            self.target_network_params[i].assign(self.network_params[i])

        #gradients of Q function with respect to actions
    
    def create_critic_network(self):
        inputs_state = Input(shape = (self.state_dim,), batch_size = None, name = "critic_input_state")
        inputs_action = Input(shape = (self.action_dim,), batch_size = None, name = "critic_input_action")
        w_init = tf.random_uniform_initializer(minval=-0.03, maxval=0.03, seed=None)
        
        #first hidden layer
        net_state = layers.Dense(self.params_l1, name = 'critic_dense_1', kernel_initializer = w_init)(inputs_state)
        net_state = layers.BatchNormalization()(net_state)
        net_state = layers.Activation(activation=tf.nn.relu)(net_state)

        # second hidden layer
        net_state = layers.Dense(self.params_l2, name = 'critic_dense_2_state', kernel_initializer = w_init)(net_state)
        net_action = layers.Dense(self.params_l2, name = 'critic_dense_2_action', kernel_initializer = w_init)(inputs_action)
        net = layers.Add()([net_state, net_action])
        net = layers.BatchNormalization()(net)
        net = layers.Activation(activation=tf.nn.relu)(net)

        #w_init = tf.random_uniform_initializer(minval=-0.03, maxval=0.03, seed=None)
        out = layers.Dense(1, name = 'Q_val', kernel_initializer = w_init)(net)
        return inputs_state, inputs_action, out

    def update_target_network(self):
        self.update_target_network_params = [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1-self.tau)) for i in range(len(self.target_network_params))]

    def train(self, input_state, input_actions, predicted_q_val):
        with tf.GradientTape(watch_accessed_variables=True) as tape:
            prediction = self.critic_model([input_state, input_actions])
            loss = tf.keras.losses.MSE(prediction, predicted_q_val)
        gradients = tape.gradient(loss, self.network_params)
        self.optimizer.apply_gradients(zip(gradients, self.network_params))
        return self.critic_model([input_state, input_actions])
  
    def action_gradient(self, input_state, input_actions):
        var = tf.constant(input_actions)
        with tf.GradientTape(watch_accessed_variables=False) as tape_a:
            tape_a.watch(var)
            prediction_a = self.critic_model([input_state, var])
        return tape_a.gradient(prediction_a, var)
  
    def predict(self, inputs_state, inputs_actions):
        return self.critic_model([inputs_state, inputs_actions])
  
    def predict_target(self, inputs_state, inputs_actions):
        return self.target_critic_model([inputs_state, inputs_actions])


##---------------------------------------------------------------------------------------------------------------
def train(env, test_env, args, actor, critic, actor_noise, reward_result, scaler, replay_buffer):
    writer = tf.summary.create_file_writer(logdir = args['summary_dir'])
    actor.update_target_network()
    critic.update_target_network()


    paths = list()

    for i in range(args['max_episodes']):
        test_env.reset()
        s = env.reset()
        ep_reward = 0
        ep_ave_max_q = 0
        obs, actions, rewards = [], [], []
        if args['scaling']:
            var, mean = scaler.get()
        else:
            var, mean = 1.0, 0.0
        if i<10:
            temp_a = np.random.normal(0, 0.1)
        for j in range(args['max_episode_len']):
            s_scaled = np.float32((s - mean) * var)
            
            #noise annealing
            noise = np.random.normal(0, args['noise_var']/((j/100)+1))
            a = actor.predict(np.reshape(s_scaled,(1,actor.state_dim))) + noise
            #print(a)
            #actor_noise()
            #print(j,a)
            if i<10:
                a = tf.constant([[temp_a + noise]])
                #print(a)
            # action should be inbetween -1 and 1
            a = np.clip(a, -args['action_bound'], args['action_bound'])

            s2, r, terminal, info = env.step(a[0])
            s2_scaled = np.float32((s2 - mean) * var)
            replay_buffer.add(
                np.reshape(s_scaled, (actor.state_dim,)), np.reshape(a, (actor.action_dim,)), r, terminal, np.reshape(s2_scaled, (actor.state_dim,)))
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
                    "Observation":np.concatenate(obs).reshape((args['max_episode_len'],2)), 
                    "Action":np.concatenate(actions), 
                    "Reward":np.asarray(rewards)
                    }
                paths.append(path)
                #env.plot()
                test_s = test_env.reset()
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

##---------------------------------------------------------------------------------------------------------------
#------------------------------using RNN---------------------------------------------------------------
##---------------------------------------------------------------------------------------------------------------
class ActorNetwork_rnn(object):
    def __init__(self, state_dim, action_dim, action_bound, learning_rate, tau, batch_size, params_rnn, params_l1, params_l2, time_steps):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau  = tau
        self.batch_size = batch_size
        self.params_rnn = params_rnn
        self.params_l1 = params_l1
        self.params_l2 = params_l2
        self.time_steps = time_steps
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        #actor network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()
        self.actor_model = keras.Model(inputs=self.inputs, outputs=self.scaled_out, name='actor_network')
        self.network_params = self.actor_model.trainable_variables

        #target actor network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()
        self.target_actor_model = keras.Model(inputs=self.target_inputs, outputs=self.target_scaled_out, name='target_actor_network')
        self.target_network_params = self.target_actor_model.trainable_variables

        #initalizing the target params with then network params
        for i in range(len(self.target_network_params)):
            self.target_network_params[i].assign(self.network_params[i])


    def create_actor_network(self):
        inputs = Input(shape = (self.time_steps, self.state_dim), batch_size = None, name = "actor_input_state")
        w_init = tf.random_uniform_initializer(minval=-0.03, maxval=0.03, seed=None)

        lstm_net = layers.GRU(
            units= self.params_rnn, 
            return_sequences=False, 
            return_state=False, 
            name = 'actor_rnn', 
            #kernel_initializer = w_init
            )(inputs)

        net = layers.Dense(self.params_l1, name = 'actor_dense_1', kernel_initializer = w_init)(lstm_net)
        net = layers.BatchNormalization()(net)
        net = layers.Activation(activation=tf.nn.relu)(net)

        net = layers.Dense(self.params_l2, name = 'actor_dense_2', kernel_initializer = w_init)(net)
        net = layers.BatchNormalization()(net)
        net = layers.Activation(activation='tanh')(net)
        
        
        out = layers.Dense(self.action_dim, activation='tanh', name = 'actor_dense_3', kernel_initializer = w_init)(net)
        scaled_out = tf.multiply(out, self.action_bound, name = "actions_scaling")
        return inputs, out, scaled_out
  
    def update_target_network(self):
        self.update_target_network_params = [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1-self.tau)) for i in range(len(self.target_network_params))]
  
    def train(self, inputs, a_gradient):
        with tf.GradientTape(watch_accessed_variables=True) as tape:
            prediction = self.actor_model(inputs)
        unnormalized_actor_gradients = tape.gradient(prediction, self.network_params, output_gradients = -a_gradient)
        actor_gradients = list(map(lambda x: tf.math.divide(x, self.batch_size), unnormalized_actor_gradients))
        self.optimizer.apply_gradients(zip(actor_gradients, self.network_params))
    
    def predict(self, inputs):
        return self.actor_model(inputs)

    def predict_target(self, inputs):
        return self.target_actor_model(inputs)



##---------------------------------------------------------------------------------------------------------------
class CriticNetwork_rnn(object):
    def __init__(self, state_dim, action_dim, action_bound, learning_rate, tau, gamma, params_rnn, params_l1, params_l2, time_steps):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau  = tau
        self.gamma = gamma
        self.params_rnn = params_rnn
        self.params_l1 = params_l1
        self.params_l2 = params_l2
        self.time_steps = time_steps
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        #Critic Network and parameters
        self.inputs_state, self.inputs_action, self.out = self.create_critic_network()
        self.critic_model = keras.Model(inputs=[self.inputs_state, self.inputs_action], outputs=self.out, name='critic_network')
        self.network_params = self.critic_model.trainable_variables

        #Target Critic Network and parameters
        self.target_inputs_state, self.target_inputs_action, self.target_out = self.create_critic_network()
        self.target_critic_model = keras.Model(inputs=[self.target_inputs_state, self.target_inputs_action], outputs=self.target_out, name='target_critic_network')
        self.target_network_params = self.target_critic_model.trainable_variables
        
        #initalizing the target params with then network params
        for i in range(len(self.target_network_params)):
            self.target_network_params[i].assign(self.network_params[i])

        #gradients of Q function with respect to actions
    
    def create_critic_network(self):
        inputs_state = Input(shape = (self.time_steps, self.state_dim), batch_size = None, name = "critic_input_state")
        inputs_action = Input(shape = ( self.action_dim,), batch_size = None, name = "critic_input_action")
        w_init = tf.random_uniform_initializer(minval=-0.03, maxval=0.03, seed=None)

        #LSTM layer
        lstm_net = layers.GRU(units = self.params_rnn, return_sequences=False, return_state=False)(inputs_state)
        
        #first hidden layer
        net_state = layers.Dense(self.params_l1, name = 'critic_dense_1', kernel_initializer = w_init)(lstm_net)
        net_state = layers.BatchNormalization()(net_state)
        net_state = layers.Activation(activation=tf.nn.relu)(net_state)

        # second hidden layer
        net_state = layers.Dense(self.params_l2, name = 'critic_dense_2_state', kernel_initializer = w_init)(net_state)
        net_action = layers.Dense(self.params_l2, name = 'critic_dense_2_action', kernel_initializer = w_init)(inputs_action)

        # net = layers.Add()([net_state, net_action])
        net = layers.concatenate([net_state, net_action])
        net = layers.Dense(self.params_l2, name = 'critic_dense_3_state', kernel_initializer = w_init)(net)
        net = layers.BatchNormalization()(net)
        net = layers.Activation(activation=tf.nn.relu)(net)

        #w_init = tf.random_uniform_initializer(minval=-0.03, maxval=0.03, seed=None)
        out = layers.Dense(1, name = 'Q_val', kernel_initializer = w_init)(net)
        return inputs_state, inputs_action, out

    def update_target_network(self):
        self.update_target_network_params = [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1-self.tau)) for i in range(len(self.target_network_params))]

    def train(self, input_state, input_actions, predicted_q_val):
        with tf.GradientTape(watch_accessed_variables=True) as tape:
            prediction = self.critic_model([input_state, input_actions])
            #print('predicted_q_val', np.shape(predicted_q_val))
            #print('prediction', np.shape(prediction))
            loss = tf.keras.losses.MSE(prediction, predicted_q_val)
        gradients = tape.gradient(loss, self.network_params)
        self.optimizer.apply_gradients(zip(gradients, self.network_params))
        return self.critic_model([input_state, input_actions])
  
    def action_gradient(self, input_state, input_actions):
        var = tf.constant(input_actions)
        with tf.GradientTape(watch_accessed_variables=False) as tape_a:
            tape_a.watch(var)
            prediction_a = self.critic_model([input_state, var])
        return tape_a.gradient(prediction_a, var)
  
    def predict(self, inputs_state, inputs_actions):
        return self.critic_model([inputs_state, inputs_actions])
  
    def predict_target(self, inputs_state, inputs_actions):
        return self.target_critic_model([inputs_state, inputs_actions])






##---------------------------------------------------------------------------------------------------------------
def train_rnn(env, test_env, args, actor, critic, actor_noise, reward_result, scaler, replay_buffer):
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
            a = np.clip(a, -args['action_bound'], args['action_bound'])

            s2, r, terminal, info = env.step(a[0])
            s2_scaled = np.float32((s2 - mean) * var)

            #noise annealing
            obs_scaled.append(s2_scaled)
            S_2 = obs_scaled[-args['time_steps']: ]

            replay_buffer.add(S_0, np.reshape(a, (actor.action_dim,)), r, terminal, S_2)
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
def train_multi_agent(env, test_env, args, actors, critics, reward_result, scaler, replay_buffers, save_weights):

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

            #we take step using this action
            s2, r, terminal, info = env.step(a_nodes)

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
                actions[node].append(a_nodes[node])
                rewards[node].append(node_reward)
                node_rewards.append(node_reward)

            #for each node -- add the current state, action, reward, terminal and the previous state to the replay buffer
            for node in range(nodes):
                S_0 = obs_scaled[node][-args['time_steps']-1: -1]
                S_1 = obs_scaled[node][-args['time_steps']: ]
                replay_buffers[node].add(S_0, np.reshape(a_nodes[node], (actors[node].action_dim,)), node_rewards[node], terminal, S_1)
            
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