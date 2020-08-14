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
import wesutils
import os


# local modules
from rl.utils import Scaler, OrnsteinUhlenbeckActionNoise, ReplayBuffer
from rl.gym_env.buck import Buck_Converter_n
from rl.agents.ddpg import ActorNetwork, CriticNetwork, train


config_path = "config.yml"

if __name__ == "__main__":

    # load config file and create directory for logging results
    config = wesutils.load_config(config_path)
    experiment_dir = wesutils.create_logdir(config['log_dir'],
                                            config['algorithm'],
                                            config['env_name'],
                                            config_path)

    # use GPU if config says to
    if config['use_gpu']:
        physical_devices = tf.config.list_physical_devices('GPU') 
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # create env and get state and action space info
    env = Buck_Converter_n()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high

    # add these entries to config, since train needs them
    config['state_dim'] = state_dim
    config['action_dim'] = action_dim
    config['action_bound'] = action_bound
    config['experiment_dir'] = experiment_dir

    test_env = Buck_Converter_n(Vs = 400, L = 1, C = 1, R = 1, G = 0.1, Vdes = 230, dt = config['discretization_time'])
    env = Buck_Converter_n(Vs = 400, L = 1, C = 1, R = 1, G = 0.1, Vdes = 230, dt = config['discretization_time'])

    test_s = test_env.reset()
    test_obs=[]
    test_steps = int(1/config['discretization_time'])
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

    actor = ActorNetwork(
        state_dim = config['state_dim'],
        action_dim = config['action_dim'], 
        action_bound=config['action_bound'], 
        learning_rate = config['actor_lr'], 
        tau = config['tau'], 
        batch_size = config['mini_batch_size'],
        params_l1 = config['actor_l1'],
        params_l2 = config['actor_l2']
        )
    critic = CriticNetwork(
        state_dim = config['state_dim'], 
        action_dim = config['action_dim'], 
        action_bound = config['action_bound'], 
        learning_rate = config['critic_lr'], 
        tau = config['tau'], 
        gamma = config['gamma'],
        params_l1 = config['critic_l1'],
        params_l2 = config['critic_l2']
        )

    #loading the weights
    if args['load_model']:
        if os.path.isfile(args['log_dir'] + "/actor_weights.h5"):
            print('loading actor weights')
            actor.actor_model.load_weights(args['log_dir'] + "/actor_weights.h5")
        if os.path.isfile(args['log_dir'] + "/target_actor_weights.h5"):
            print('loading actor target weights')
            actor.actor_model.load_weights(args['log_dir'] + "/target_actor_weights.h5")
        if os.path.isfile(args['log_dir'] + "/critic_weights.h5"):
            print('loading critic weights')
            critic.critic_model.load_weights(args['log_dir'] + "/critic_weights.h5")
        if os.path.isfile(args['log_dir'] + "/target_critic_weights.h5"):
            print('loading critic target weights')
            critic.critic_model.load_weights(args['log_dir'] + "/target_critic_weights.h5")

    replay_buffer = ReplayBuffer(int(config['buffer_size']), int(config['random_seed']))
    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))
    reward_result = np.zeros(2500)
    paths, reward_result = train(env, test_env, config, actor, critic, actor_noise, reward_result, scaler, replay_buffer)

    # Saving the weights
    if args['save_model']:
        actor.actor_model.save_weights(
            filepath = args['log_dir'] + "/actor_weights.h5",
            overwrite=True,
            save_format='h5')
        actor.target_actor_model.save_weights(
            filepath = args['log_dir'] + "/target_actor_weights.h5",
            overwrite=True,
            save_format='h5')
        critic.critic_model.save_weights(
            filepath = args['log_dir'] + "/critic_weights.h5",
            overwrite=True,
            save_format='h5')
        critic.target_critic_model.save_weights(
            filepath = args['log_dir'] + "/target_critic_weights.h5",
            overwrite=True,
            save_format='h5')


    #plotting
    test_s = test_env.reset()
    for _ in range(10000):
        if config['scaling']:
            test_s_scaled = np.float32((test_s - mean) * var)
        else:
            test_s_scaled = test_s
        test_a = actor.predict(np.reshape(test_s_scaled,(1,actor.state_dim)))
        test_s, r, terminal, info = test_env.step(test_a[0])

    savefig_filename = os.path.join(experiment_dir, 'results_plot.png')
    test_env.plot(savefig_filename=savefig_filename)

    savemat(os.path.join(experiment_dir, 'data.mat'),
            dict(data=paths, reward=reward_result))

