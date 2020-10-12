
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import seaborn as sns

import numpy as np
import pandas as pd
from scipy.io import savemat, loadmat
import os
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pylab as pylab
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib.pylab as pylab

from rl.gym_env.buck_microgrid import Buck_microgrid



sns.set_context("paper")
sns.set_style("whitegrid")
sns.set()
params = {#'legend.fontsize': 'x-large',
          #'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large',
         'pdf.fonttype' : 42,
         'ps.fonttype' : 42
         }
pylab.rcParams.update(params)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


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

def plot_agent_rewards_together(paths_ddpg, paths_cbf, config):
    fig, axes = plt.subplots(1,4,figsize = config['figsize'], sharex=config['sharex'])
    for en, paths in enumerate([paths_ddpg, paths_cbf]):
        data = net_reward(paths = paths, episodes = config['episodes'])
        DataFrame = [pd.DataFrame(r) for r in iter(data)]
        #pd.DataFrame.to_csv
        for i, df in enumerate(DataFrame):
            mean = df.iloc[2:].mean(axis=0)
            std  = df.iloc[2:].std(axis=0)
            steps = np.arange(int(mean.size))
            #print(steps,mean)
            # for run in range(df.shape[0]):
            #     axes[i].plot(steps, df.iloc[run,:], color='000000', alpha=0.1)
            axes[i].plot(steps, mean, '--', color=config['colors'][en], alpha=1, label=config['label'][en])
            axes[i].fill_between(steps, mean - 0.5*std, mean + 0.5*std,color=config['colors'][en], alpha=0.25)
            axes[i].set_ylim(-6e6,0.1e6)
            axes[i].set_title('node - '+str(i+1), fontsize=config['font_size'])
            axes[i].set_label('Label via method')
            axes[i].legend(loc=config['legend_loc'], fontsize=config['font_size'])
            #axes[i].yaxis.set_major_formatter(FormatStrFormatter('%.1E'))
            axes[i].tick_params(axis='both', which='major', labelsize=config['font_size'])
            axes[i].tick_params(axis='both', which='minor', labelsize=config['font_size'])
            axes[i].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            axes[i].set_xlabel('episodes', fontsize=config['font_size'])
            axes[i].set_ylabel('returns', fontsize=config['font_size'])
            #axes[i].grid(linestyle='--', linewidth='0.5', color='000000', alpha = 0.5)
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
    if config['save_fig']:
        assert isinstance(config['savefig_filename'], str), "filename for saving the figure must be a string"
        plt.savefig(config['savefig_filename'], format = config['format'])
    else:
        plt.show()



path = './Power-Converters/kristools/results/'
files_cbf = os.listdir(path + 'cbf/')
files_ddpg = os.listdir(path + 'ddpg/')

#print(files)
paths_cbf = [path + 'cbf/'+ file for file in files_cbf]
paths_ddpg = [path + 'ddpg/'+ file for file in files_ddpg]
print(paths_ddpg)
config = {
    'colors':['r','b'],
    'label':['ddpg','dcbf'],
    'save_fig':True,
    'savefig_filename' : path + '/DDPG_DCBF_200_oct_12_1.pdf',
    'font_size' : 20,
    'figsize' : (20,7),
    'frame_on' : True,
    'sharex' : True,
    'format' :'pdf',
    'legend_loc' : "lower right",
    'episodes' : 200}
plot_agent_rewards_together(paths_ddpg, paths_cbf, config)



env = Buck_microgrid(dt = 1e-6)
mat_path = path + '/test_dcbf_data/save_data1.mat'
DATA = loadmat(mat_path)
obs, actions = DATA['data'], DATA['actions']
limit = int(150002/2)
diss =[[] for _ in range(4)]
for _ in range(4):
    i, v = obs[_][:limit, 0], obs[_][:limit, 1]
    id = env.Ides[_]
    vd = env.Vdes[_]
    a = actions.T[_]
    ud = env.udes[_]
    r = env.R[_][_]
    g = env.G[_][_]
    supply_rate = [-(I-id)*(U-ud)+r*(I-id)**2+g*(g-vd)**2 for I, V, U in zip(i, v, a)]

    diss[_].append(supply_rate[0])
    for _t, s_a in enumerate(supply_rate):
        if _t>0:
            diss[_].append(diss[_][_t-1] + supply_rate[_t])


def cbf_plot(config):
    limit = config['x_max']#int(150002/2)
    fig, axes = plt.subplots(nrows = 2, figsize = config['figsize'], sharex=config['sharex'], subplot_kw=dict(frameon=config['frame_on']))
    plt.subplots_adjust(hspace=.1)
    steps = np.arange(obs[0][:limit,1].shape[0])*1e-6
    for i in range(4):
        I, V = obs[i][:limit, 0], obs[i][:limit, 1]
        d = diss[i]
        axes[0].plot(np.arange(np.shape(d)[0])*1e-6, d, label=config['label_D'][i], linewidth=2)
        axes[1].plot(steps, V, label=config['label_V'][i], linewidth=2)
        axes[1].plot(steps,  np.full_like(V, config['Vdes'][i]), linestyle='dashed', linewidth=2, color='000000')
    
    axes[0].set_label('Label via method')
    axes[0].legend(loc=config['legend_loc'], ncol = 4, fontsize=config['font_size'])
    axes[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axes[0].tick_params(axis='both', which='major', labelsize=config['font_size'])
    axes[0].tick_params(axis='both', which='minor', labelsize=config['font_size'])
    axes[0].grid(linestyle='--', linewidth='0.5', color='000000', alpha = 0.1)
    axes[0].set_ylabel('$cbf -- b^i(t,x_t^i)$', fontsize=config['font_size'])

    axes[1].axvline(x=0.05, color='red', linestyle='--', alpha = 0.3 )
    axes[1].set_ylim(config['ylim_V'][0], config['ylim_V'][1])
    axes[1].set_label('Label via method')
    axes[1].legend(loc=config['legend_loc'], ncol = 4, fontsize=config['font_size'])
    #axes[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axes[1].tick_params(axis='both', which='major', labelsize=config['font_size'])
    axes[1].tick_params(axis='both', which='minor', labelsize=config['font_size'])
    axes[1].grid(linestyle='--', linewidth='0.5', color='000000', alpha = 0.1)
    axes[1].set_xlabel('time', fontsize=config['font_size'])
    axes[1].set_ylabel('Voltage (V)', fontsize=config['font_size'])
    if config['save_fig']:
        assert isinstance(config['savefig_filename'], str), \
                "filename for saving the figure must be a string"
        plt.savefig(config['savefig_filename'], format = 'pdf')
    else:
        plt.show()
config = {
    'ylim_D' : (0, 4e8),
    'ylim_V' : (200,250),
    #'label_D' : ['$b^1$', '$\mathregular{b^2}$','$ \mathregular{b^3}$', '$\mathregular{b^4}$'],
    #'label_D' : ['$b^1(t,x_t^1)$', '$b^2(t,x_t^2)$', '$b^3(t,x_t^3)$', '$b^4(t,x_t^4)$'],
    'label_D' : ['$i=1$','$i=2$','$i=3$','$i=4$'],
    'label_V' : ['$V_1$', '$V_2$','$ V_3$', '$V_4$'],
    'save_fig':True,
    'savefig_filename' : path + '/test_dcbf_data/validation_1.pdf',
    'Vdes':np.array([232, 230, 228, 228]),
    'font_size' : 15,
    'legend_loc' : "upper left",
    'figsize' : (8,6),
    'frame_on' : True,
    'sharex' : True,
    'x_max' : int(150002/2)
}
cbf_plot(config)