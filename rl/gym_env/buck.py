import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt

class Buck_Converter_n(gym.Env):
    """
    Buck converter model following gym interface.
    We are assuming that the switching frequency is very High.
    Action space is continious and symmetric.
    Parameters need to satisfy: (1+RG)Vdes < Vs.
    It saves the input and state trajectories inside self.state_trajectory, self.action_trajectory.
    Has a plot method.

    env = Buck_Converter_n(Vs = 400, L = 1, C = 1, R =  1, G = .1, Vdes = 150, dt = 1e-2)
    env.reset()

    N_steps = 1000
    for i in range(N_steps):
        s2, r, terminal, info = env.step(env.action_des)
    env.plot()
    """
    metadata = {'render.modes': ['console']}

    def __init__(self, Vs = 400, L = 0.001, C = 0.001, R = 0.001, G = 0.04, Vdes = 220, dt = 1e-5):
        super(Buck_Converter_n, self).__init__()

        #parameters
        self.Vs = Vs
        self.L = L
        self.C = C
        self.R = R
        self.G = G

        #step size; since L and C are very low, the ode becomes stiff
        #For the default parameters the step size should in the order of 1e-5
        self.T = dt

        #the steady-state equilibrium of the system is
        self.Vdes = Vdes
        self.Ides = self.G * self.Vdes
        self.udes = (1/self.Vs) * (1 + self.G * self.R) * self.Vdes
        self.action_des = 2*self.udes - 1
        
        #The control action is duty-ratio which lies between 0 and 1 (We are assuming that the switching frequency is very High)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([-np.inf, -np.inf]), high=np.array([+np.inf, +np.inf]), shape=None, dtype=np.float32)
        
        self._get_state()

        #lists to save the states and actions 
        self.state_trajectory = []
        self.action_trajectory = []
        self.count_steps = 0 # counts the number of steps taken
    
    def _get_state(self):
        #initializing the state vector near to the desired values
        if self.udes >= 1:
            raise ValueError("for buck converter desired dutyratio should be less than 1 and greater than zero. Try: decreasing Vdes or increasing Vs.")
        I = np.random.uniform(low = self.Ides-1, high = self.Ides+1)
        V = np.random.uniform(low = self.Vdes-1, high = self.Vdes+1)
        self.state = np.array([I, V])

    def _set_state(self, I, V):
        #using this function we can change the state variable
        self.state = np.array([I, V])

    def desired(self):
        #Provides the steady-state variables
        return np.array([self.Ides, self.Vdes, self.udes])

    def reset(self):
        """
        Important: re-initializing the state vector near to the desired values
        :return: (np.array) 
        and restting the state_trajectory and action_trajectory buffers
        """
        #self.state = np.array(np.random.normal([self.Ides , self.Vdes], 5)).astype(np.float32)
        #reset
        self._get_state()

        self.state_trajectory = []
        self.action_trajectory = []
        self.count_steps = 0
        return self.state
    
    def step(self, action):
        #u = (action + 1)/2.0

        u = np.clip((action + 1)/2.0, 0, 1)

        i = self.state[0]
        v = self.state[1]

        didt = (-1.0/self.L)*(self.R*i + v - self.Vs*u)
        dvdt = (1.0/self.C)*(i - self.G*v)

        new_i = i + self.T * didt
        new_v = v + self.T * dvdt

        self.state = np.array([new_i, new_v]).astype(np.float32)

        # normalize the rewards
        reward = -((new_v-self.Vdes)/self.Vs)**2
        done = False

        self.state_trajectory.append(self.state)
        self.action_trajectory.append([action])
        
        self.count_steps += 1

        return self.state, reward, done, {}

    def render(self, mode='console'):
        # Prints the error between the desired values and their corresponding states
        ei = self.state[0] - self.Ides
        ev = self.state[1] - self.Vdes
        if mode != 'console':
            raise NotImplementedError()
        # agent is represented as a cross, rest as a dot
        print("I-Ides = {:.2f}, V-Vdes = {:.2f}".format(ei, ev))

    def close(self):
        pass
    
    def plot(self, savefig_filename=None):
        title_nodes = ['Current', 'Voltage', 'action']
        test_steps = self.count_steps
        time = np.array(range(test_steps), dtype=np.float32)*self.T
        test_obs_reshape = np.concatenate(self.state_trajectory).reshape((test_steps ,self.observation_space.shape[0]))
        test_act_reshape = np.concatenate(self.action_trajectory).reshape((test_steps ,self.action_space.shape[0]))
        state_dim = self.observation_space.shape[0]
        act_dim = self.action_space.shape[0]
        total_dim = self.observation_space.shape[0] + self.action_space.shape[0]

        fig, ax = plt.subplots(nrows=1, ncols=total_dim, figsize = (24,4))
        for i in range(state_dim):
            ax[i].plot(time, test_obs_reshape[:, i], label=title_nodes[i])
            ax[i].plot(time, np.full(test_obs_reshape[:,i].shape[0], self.desired()[i]), marker = '.', label='desired')
            #ax[i].set_ylim(des[i]-50, des[i]+50)
            ax[i].set_title(title_nodes[i], fontsize=15)
            ax[i].set_xlabel('Time', fontsize=10)
            ax[i].set_label('Label via method')
            ax[i].legend()
        for i in range(state_dim, total_dim):
            ax[i].plot(time, test_act_reshape, label=title_nodes[i])
            ax[i].plot(time, np.full(test_act_reshape.shape[0], self.action_des), marker = '.', label='desired')
            ax[i].set_xlabel('Time', fontsize=10)
            ax[i].set_title(title_nodes[i], fontsize=15)
            ax[i].set_label('Label via method')
            ax[i].legend()

        if savefig_filename is not None:
            assert isinstance(savefig_filename, str), \
                    "filename for saving the figure must be a string"
            plt.savefig(savefig_filename)
        else:
            plt.show()