import tensorflow as tf
import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
import argparse
from tensorflow import keras
from tensorflow.keras import Input, Model, Sequential, layers

# print(tf.__version__)
# tf.test.gpu_device_name()




#===================================
#     Actor and Critic DNNs
#===================================


#===================================
#     Actor and Critic DNNs
#===================================


class ActorNetwork(object):
    def __init__(self, state_dim, action_dim, action_bound, learning_rate, tau, batch_size):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau  = tau
        self.batch_size = batch_size
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        #actor network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()
        self.actor_model = keras.Model(inputs=self.inputs, outputs=self.scaled_out, name='actor_network')
        self.network_params = self.actor_model.trainable_variables

        #target actor network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()
        self.target_actor_model = keras.Model(inputs=self.target_inputs, outputs=self.target_scaled_out, name='target_actor_network')
        self.target_network_params = self.target_actor_model.trainable_variables


    def create_actor_network(self):
        inputs = Input(shape = (self.state_dim,), batch_size = None, name = "actor_input_state")
        w_init = tf.random_uniform_initializer(minval=-0.03, maxval=0.03, seed=None)

        net = layers.Dense(400, name = 'actor_dense_1', kernel_initializer = w_init)(inputs)
        net = layers.BatchNormalization()(net)
        net = layers.Activation(activation=tf.nn.relu)(net)

        net = layers.Dense(300, name = 'actor_dense_2', kernel_initializer = w_init)(net)
        net = layers.BatchNormalization()(net)
        net = layers.Activation(activation='tanh')(net)
        
        
        out = layers.Dense(self.action_dim, activation='tanh', name = 'actor_dense_3', kernel_initializer = w_init)(net)
        scaled_out = tf.multiply(out, self.action_bound, name = "actions_scaling")
        return inputs, out, scaled_out
  
    def update_target_network(self):
        self.update_target_network_params = [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1-self.tau)) for i in range(len(self.target_network_params))]
  
    def train(self, inputs, a_gradient):
        with tf.GradientTape() as self.tape:
            self.prediction = self.actor_model(inputs)
        self.unnormalized_actor_gradients = self.tape.gradient(self.prediction, self.network_params, output_gradients = -a_gradient)
        self.actor_gradients = list(map(lambda x: tf.math.divide(x, self.batch_size), self.unnormalized_actor_gradients))
        self.optimizer.apply_gradients(zip(self.actor_gradients, self.network_params))
    
    def predict(self, inputs):
        return self.actor_model(inputs)

    def predict_target(self, inputs):
        return self.target_actor_model(inputs)




class CriticNetwork(object):
    def __init__(self, state_dim, action_dim, action_bound, learning_rate, tau, gamma):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau  = tau
        self.gamma = gamma
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        #Critic Network and parameters
        self.inputs_state, self.inputs_action, self.out = self.create_critic_network()
        self.critic_model = keras.Model(inputs=[self.inputs_state, self.inputs_action], outputs=self.out, name='critic_network')
        self.network_params = self.critic_model.trainable_variables

        #Target Critic Network and parameters
        self.target_inputs_state, self.target_inputs_action, self.target_out = self.create_critic_network()
        self.target_critic_model = keras.Model(inputs=[self.target_inputs_state, self.target_inputs_action], outputs=self.target_out, name='target_critic_network')
        self.target_network_params = self.target_critic_model.trainable_variables

        #gradients of Q function with respect to actions
    
    def create_critic_network(self):
        inputs_state = Input(shape = (self.state_dim,), batch_size = None, name = "critic_input_state")
        inputs_action = Input(shape = (self.action_dim,), batch_size = None, name = "critic_input_action")
        w_init = tf.random_uniform_initializer(minval=-0.03, maxval=0.03, seed=None)
        
        #first hidden layer
        net_state = layers.Dense(400, name = 'critic_dense_1', kernel_initializer = w_init)(inputs_state)
        net_state = layers.BatchNormalization()(net_state)
        net_state = layers.Activation(activation=tf.nn.relu)(net_state)

        # second hidden layer
        net_state = layers.Dense(300, name = 'critic_dense_2_state', kernel_initializer = w_init)(net_state)
        net_action = layers.Dense(300, name = 'critic_dense_2_action', kernel_initializer = w_init)(inputs_action)
        net = layers.Add()([net_state, net_action])
        net = layers.BatchNormalization()(net)
        net = layers.Activation(activation=tf.nn.relu)(net)

        #w_init = tf.random_uniform_initializer(minval=-0.03, maxval=0.03, seed=None)
        out = layers.Dense(1, name = 'Q_val', kernel_initializer = w_init)(net)
        return inputs_state, inputs_action, out

    def update_target_network(self):
        self.update_target_network_params = [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1-self.tau)) for i in range(len(self.target_network_params))]

    def train(self, input_state, input_actions, predicted_q_val):
        with tf.GradientTape() as self.tape:
            self.prediction = self.critic_model([input_state, input_actions])
            self.loss = tf.keras.losses.MSE(self.prediction, predicted_q_val)
        self.gradients = self.tape.gradient(self.loss, self.network_params)
        self.optimizer.apply_gradients(zip(self.gradients, self.network_params))
        return self.critic_model([input_state, input_actions])
  
    def action_gradient(self, input_state, input_actions):
        var = tf.constant(input_actions)
        with tf.GradientTape(watch_accessed_variables=False) as self.tape:
            self.tape.watch(var)
            self.prediction = self.critic_model([input_state, var])
        return self.tape.gradient(self.prediction, var)
  
    def predict(self, inputs_state, inputs_actions):
        return self.critic_model([inputs_state, inputs_actions])
  
    def predict_target(self, inputs_state, inputs_actions):
        return self.target_critic_model([inputs_state, inputs_actions])



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
        var, mean = scaler.get()
        if i<10:
            temp_a = np.random.normal(0, 0.9)
        for j in range(args['max_episode_len']):
            s_scaled = np.float32((s - mean) * var)
            noise = np.random.normal(0, 0.05)
            a = actor.predict(np.reshape(s_scaled,(1,actor.state_dim))) + noise
            #print(a)
            #actor_noise()
            #print(j,a)
            if i<10:
                a = tf.constant([[temp_a+ noise]])
                #print(a)

            s2, r, terminal, info = env.step(a[0])
            s2_scaled = np.float32((s2 - mean) * var)
            replay_buffer.add(np.reshape(s_scaled, (actor.state_dim,)), np.reshape(a, (actor.action_dim,)), r, terminal, np.reshape(s2_scaled, (actor.state_dim,)))
            if replay_buffer.size() > args['mini_batch_size']:
                s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(args['mini_batch_size'])
                target_q = np.array(critic.predict_target(s2_batch, actor.predict_target(s2_batch)))
                y = []
                for k in range(args['mini_batch_size']):
                    if t_batch[k]:
                        y.append(r_batch[k])
                    else:
                        y.append(r_batch[k] + critic.gamma * np.array(target_q[k]))

                temp = np.reshape(y, (args['mini_batch_size'],1))

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
                if i+1 == args['max_episodes']:
                    for _ in range(1000):
                        test_s_scaled = np.float32((test_s - mean) * var) 
                        test_a = actor.predict(np.reshape(test_s_scaled,(1,actor.state_dim)))
                        test_s, r, terminal, info = test_env.step(test_a[0])
                    test_env.plot()
                break




