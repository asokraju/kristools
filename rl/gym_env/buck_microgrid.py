# Imports
import networkx
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt

class Buck_microgrid(gym.Env):
    """
    Buck converter model following gym interface
    We are assuming that the switching frequency is very High
    Action space is continious
    """ 
    metadata = {'render.modes': ['console']}

    def __init__(self, dt = 1e-5):
        super(Buck_microgrid, self).__init__()

        #parameters 1+
        self.Vs = np.array([400, 400, 400, 400])
        #self.L = np.diag(np.array([1.0, 1.0, 1.0, 1.0]))#
        self.L = np.diag(np.array([1.8, 2.0, 3.0, 2.2])*1e-3)
        #self.C = np.diag(np.array([1.0, 1.0, 1.0, 1.0]))#
        self.C = np.diag(np.array([2.2, 1.9, 2.5, 1.7])*1e-3)
        #self.R = np.diag(np.array([1.0, 1.0, 1.0, 1.0]))#
        self.R = np.diag(np.array([1.5, 2.3, 1.7, 2.1])*0)
        #self.G = np.diag(np.array([0.1, 0.1, 0.1, 0.1]))# 
        self.G = np.diag(1/np.array([16.7, 50, 16.7, 20]))

        self.Lt = np.diag(np.array([2.1, 2, 3, 2.2])*1e-5)
        self.Rt = np.diag(np.array([7, 5, 8, 6])*1e-4)

        """
        W = inv(diag([0.4 0.2 0.15 0.25]));
        D = 100*[1 -1 0 0; -1 2 -1 0; 0 -1 2 -1; 0 0 -1 1];
        B = [-1 0 0 -1;
              1 -1 0 0; 
              0 1 -1 0; 
              0 0 1 1  ];
        """
        #Graph structure
        self.inc_mat = np.array([[-1, 0, 0, -1],
                          [1, -1, 0, 0], 
                          [0, 1, -1, 0], 
                          [0, 0, 1, 1  ]])
        self.adj_mat = (np.dot(self.inc_mat, self.inc_mat.T)-2*np.eye(4)).astype(int)
        self.Graph = nx.from_numpy_matrix(self.adj_mat)
        self.pos = nx.spring_layout(self.Graph) #networkx.random_layout(G)
        self.options = {
            'node_color': 'red',
            'node_size': 1300,
            'width': 1,
            'arrowstyle': '-|>',
            'arrowsize': 12,
            'pos' : self.pos}
        

        #step size; since L and C are very low, the ode becomes stiff
        #For the default parameters the step size should in the order of 1e-6
        self.T = dt

        #the steady-state equilibrium of the system is
        self.Vdes = np.array([230, 230, 230, 230])
        self.Itdes = -np.dot(np.linalg.inv(self.Rt), np.dot(self.inc_mat.T, self.Vdes))
        self.Ides = np.dot(self.G, self.Vdes) - np.dot(self.inc_mat, self.Itdes)

        self.udes = (1/self.Vs) * (np.dot(self.R, self.Ides) + self.Vdes)
        self.action_des = 2 * self.udes - 1
        if any(self.Vs <= self.Vdes):
            raise ValueError("for buck converter desired voltage should be less the source Voltage: Vdes < Vs ")
        
        #The control action is duty-ratio which lies between 0 and 1 (We are assuming that the switching frequency is very High)
        #However, RL algos work with symmetric control actions 
        # hence we transform the action space between -1 and 1
        # action = 2*duty-ratio -1
        #duty-ratio = 0.5*(action + 1)
        #lists to save the states and actions 
        self.state_trajectory = []
        self.action_trajectory = []
        self.count_steps = 0 # counts the number of steps taken

        self.action_space = spaces.Box(low=np.array([-1, -1, -1, -1]), high=np.array([+1, +1, +1, +1], dtype=np.float64))
        
        low_obs = np.full(shape = (12,), fill_value = -np.inf, dtype=np.float64)
        high_obs = np.full(shape = (12,), fill_value = np.inf, dtype=np.float64)

        self.observation_space = spaces.Box(low = low_obs, high = high_obs, dtype=np.float64)
        
        self._get_state()
    
    def _get_state(self):
        #initializing the state vector near to the desired values
        I = np.random.uniform(low = self.Ides-1, high = self.Ides+1)
        It = np.random.uniform(low = self.Itdes-1, high = self.Itdes+1)
        V = np.random.uniform(low = self.Vdes-1, high = self.Vdes+1)
        self.state = np.concatenate([I, It, V])

    def _set_state(self, I, It, V):
        #using this function we can change the state variable
        self.state = np.concatenate([I, It, V])

    def desired(self):
        #Provides the steady-state variables
        return np.array([self.Ides, self.Itdes, self.Vdes, self.udes, self.action_des])

    def reset(self):
        """
        Important: re-initializing the state vector near to the desired values
        :return: (np.array) 
        """
        self.state_trajectory = []
        self.action_trajectory = []
        self.count_steps = 0
        #self.state = np.array(np.random.normal([self.Ides , self.Vdes], 5)).astype(np.float32)
        self._get_state()
        return self.state
    
    def step(self, action):

        temp_u = (1 + action[0])/2

        u = np.clip(temp_u, 0, 1)

        i = self.state[0:4]
        it = self.state[4:8]
        v = self.state[8:12]

        didt = np.dot(self.R, i) + v - u*self.Vs
        didt = -np.dot(np.linalg.inv(self.L), didt)

        ditdt = np.dot(self.Rt, it) + np.dot(self.inc_mat.T, v)
        ditdt = -np.dot(np.linalg.inv(self.Lt), ditdt)

        dvdt = i + np.dot(self.inc_mat, it)- np.dot(self.G, v)
        dvdt = np.dot(np.linalg.inv(self.C), dvdt)



        new_i = i + self.T * didt
        new_it = it + self.T * ditdt
        new_v = v + self.T * dvdt

        self.state = np.concatenate([new_i, new_it, new_v]).astype(np.float32)

        # normalize the rewards
        self.reward = -np.mean((new_v-self.Vdes)**2)
        self.done = False

        self.state_trajectory.append(self.state)
        self.action_trajectory.append(action)
        
        self.count_steps += 1
        return self.state, self.reward, self.done, {}

    def render(self, mode='console'):
        # Prints the error between the desired values and their corresponding states
        ei = self.state[0:4] - self.Ides
        eit = self.state[4:8] - self.Itdes
        ev = self.state[8:12] - self.Vdes
        if mode != 'console':
            raise NotImplementedError()
        # agent is represented as a cross, rest as a dot
        print("I-Ides = {:.2f}, It-Itdes = {:.2f}, V-Vdes = {:.2f}".format(ei, eit, ev))

    def network_graph(self):
        networkx.draw_networkx(self.Graph, arrows=True, **self.options)

    def get_node(self, node_index):
        # use this after using step function
        # given index of the node, the function will return its current states i.e [current, voltage, the net line curent] and its reward 
        #node_index : 0 - 3
        i = self.state[0:4]
        v = self.state[8:12]
        o_it = np.dot(self.inc_mat, self.state[4:8])
        node_reward = -(v[node_index]- self.Vdes[node_index])**2
        node_state = np.array([i[node_index], v[node_index]])
        return node_state, node_reward

    def close(self):
        pass
        
    def plot(self, savefig_filename=None):
        #number_of_colors = data.shape[1]
        #color = ['r', 'b']
        state_dim = self.observation_space.shape[0]
        act_dim = self.action_space.shape[0]
        des1 = np.array([self.Ides, self.Itdes, self.Vdes, self.action_des])
        title_nodes = ['Node - 1', 'Node - 2', 'Node - 3', 'Node - 4']
        title_state = ['DGU current', 'Line current', 'Load Voltage', 'action']
        test_steps = self.count_steps
        time = np.array(range(test_steps), dtype=np.float32)*self.T
        test_obs_reshape = np.concatenate(self.state_trajectory).reshape((test_steps ,self.observation_space.shape[0]))
        #print(np.shape(self.action_trajectory),self.action_trajectory)
        test_act_reshape = np.concatenate(self.action_trajectory).reshape((test_steps ,self.action_space.shape[0]))
        test_reshaped = np.concatenate([test_obs_reshape, test_act_reshape], axis = 1)

        temp = 0
        fig, ax = plt.subplots(nrows=4, ncols=4, figsize = (16,16))
        for j in range(0, 4):
            des = des1[j]
            #fig, ax = plt.subplots(nrows=1, ncols=4, figsize = (24,4))
            time = np.array(range(test_reshaped.shape[0]), dtype=np.float32)*self.T
            for i in range(4):
                ax[j, i].plot(time, test_reshaped[:, i+temp])
                ax[j, i].plot(time, np.full(test_reshaped[:,i+temp].shape[0], des[i]), marker = '.')
                #ax[j, i].set_ylim(des[i]-50, des[i]+50)
                ax[j, i].set_title(title_nodes[i] + ' :  ' + title_state[j], fontsize=10)
                ax[j, i].set_xlabel('Time', fontsize=8)
            temp = temp + 4
            fig.suptitle(title_state[j], fontsize=10)
        #plt.show()
        if savefig_filename is not None:
            assert isinstance(savefig_filename, str), \
                    "filename for saving the figure must be a string"
            plt.savefig(savefig_filename)
        else:
            plt.show()
            