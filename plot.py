import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import seaborn as sns

import numpy as np
import pandas as pd
from scipy.io import savemat, loadmat
import os
sns.set_context("paper")
sns.set_style("whitegrid")
sns.set()

def net_reward(paths, episodes):
    r_agent = [[] for _ in  range(4)]
    episodes_ = []
    for mat_path in iter(paths):
        r = [[] for _ in range(4)]
        DATA = loadmat(mat_path)
        #episodes_.append(np.shape(DATA['data'][0])[0])
        #episodes = min(episodes_)
        for epi in range(3, episodes):
            _, _, rew = DATA['data'][0][epi][0][0]
            for i in range(4):
                r[i].append(rew[i].tolist())
        r_sum = np.array(r).sum(axis = 2)
        [r_agent[i].append(rew) for i, rew in enumerate(r_sum)]
    return r_agent

def plot_agent_rewards(paths, episodes, savefig_filename, set_format = 'pdf'):
    data = net_reward(paths = paths, episodes = episodes)
    DataFrame = [pd.DataFrame(r) for r in iter(data)]
    fig, axes = plt.subplots(1,4,figsize = (20,4))

    for i, df in enumerate(DataFrame):
        mean = df.iloc[2:].mean(axis=0)
        std  = df.iloc[2:].std(axis=0)
        steps = np.arange(mean.size)
        #print(steps,mean)
        # for run in range(df.shape[0]):
        #     axes[i].plot(steps, df.iloc[run,:], color='000000', alpha=0.1)
        axes[i].plot(steps, mean, color='b', alpha=1, label='ddpg')
        axes[i].fill_between(steps, mean - std, mean + std,color='b', alpha=0.25)
        axes[i].set_ylim(-6e6,0.1e6)
        axes[i].set_title('node-'+str(i), fontsize=15)
        axes[i].set_label('Label via method')
        axes[i].legend()
    if savefig_filename is not None:
        assert isinstance(savefig_filename, str), "filename for saving the figure must be a string"
        plt.savefig(savefig_filename, format = set_format)
    else:
        plt.show()

# path = './Power-Converters/kristools/results/matfiles/'
# files_cbf = os.listdir(path + 'cbf/')
# files_ddpg = os.listdir(path + 'ddpg/')

# #print(files)
# paths_cbf = [path + file for file in files_cbf]
# paths_ddpg = [path + file for file in files_ddpg]
# print(paths_cbf, paths_ddpg)
#print(paths_cbf)
#plot_agent_rewards(paths = paths_cbf, episodes = 500, savefig_filename=path+'plot.pdf', set_format = 'pdf')

def plot_agent_rewards_together(paths_ddpg, paths_cbf, episodes, savefig_filename=None, format = 'pdf'):
    config = {'colors':['r','b'],
              'label':['ddpg','dcbf'],
              }
    fig, axes = plt.subplots(1,4,figsize = (20,4))
    for en, paths in enumerate([paths_ddpg, paths_cbf]):
        data = net_reward(paths = paths, episodes = episodes)
        DataFrame = [pd.DataFrame(r) for r in iter(data)]
        #pd.DataFrame.to_csv
        for i, df in enumerate(DataFrame):
            mean = df.iloc[2:].mean(axis=0)
            std  = df.iloc[2:].std(axis=0)
            steps = np.arange(mean.size)
            #print(steps,mean)
            # for run in range(df.shape[0]):
            #     axes[i].plot(steps, df.iloc[run,:], color='000000', alpha=0.1)
            axes[i].plot(steps, mean, '--', color=config['colors'][en], alpha=1, label=config['label'][en])
            axes[i].fill_between(steps, mean - std, mean + std,color=config['colors'][en], alpha=0.25)
            axes[i].set_ylim(-6e6,0.1e6)
            axes[i].set_title('node - '+str(i+1), fontsize=15)
            axes[i].set_label('Label via method')
            axes[i].legend()
            axes[i].yaxis.set_major_formatter(FormatStrFormatter('%.1E'))
            # if i==0:
            #     axes[i].yaxis.set_major_formatter(FormatStrFormatter('%.1E'))
            # else:
            #     axes[i].tick_params(
            #         axis='y',          # changes apply to the x-axis
            #         which='both',      # both major and minor ticks are affected
            #         bottom=False,      # ticks along the bottom edge are off
            #         top=False,         # ticks along the top edge are off
            #         left=False,
            #         right=False,
            #         labelbottom=False)
    if savefig_filename is not None:
        assert isinstance(savefig_filename, str), "filename for saving the figure must be a string"
        plt.savefig(savefig_filename, format = format)
    else:
        plt.show()


path = './Power-Converters/kristools/results/matfiles/'
files_cbf = os.listdir(path + 'cbf/')
files_ddpg = os.listdir(path + 'ddpg/')

#print(files)
paths_cbf = [path + 'cbf/'+ file for file in files_cbf]
paths_ddpg = [path + 'ddpg/'+ file for file in files_ddpg]
plot_agent_rewards_together(paths_ddpg, paths_cbf, episodes = 200, savefig_filename=path+'/DDPG_DCBF_200_new.pdf', format = 'pdf')