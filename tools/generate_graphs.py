import csv
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

def convert_to_mean_df(panda_df):
    tp = panda_df['Timesteps'][:10000].values
    cr = panda_df['Policy Value Norm'].values
    cr = cr.reshape(10, 10000)
    cr = np.mean(cr, axis=0)
    df = pd.DataFrame({'Timesteps':tp, 'Policy Value Norm':cr})
    return df

qlearn = convert_to_mean_df(pd.read_csv('/home/rishabh/work/btp/TRAiL/tools/FourLarge/FourLargeRooms_qlearn_10000.csv'))
countbased_qlearn = convert_to_mean_df(pd.read_csv('/home/rishabh/work/btp/TRAiL/tools/FourLarge/FourLargeRooms_countbased_qlearn_10000.csv'))
trail_qlearn = convert_to_mean_df(pd.read_csv('/home/rishabh/work/btp/TRAiL/tools/FourLarge/FourLargeRooms_trail_qlearn_can_10000.csv'))
trex_qlearn = convert_to_mean_df(pd.read_csv('/home/rishabh/work/btp/TRAiL/tools/FourLarge/FourLargeRooms_trex_qlearn_can_10000.csv'))
plt.title('Comparison Plot')
plt.grid()


ax = sns.lineplot(x="Timesteps", y="Policy Value Norm", data=qlearn)
ax = sns.lineplot(x="Timesteps", y="Policy Value Norm", data=countbased_qlearn)
ax = sns.lineplot(x="Timesteps", y="Policy Value Norm", data=trail_qlearn)
ax = sns.lineplot(x="Timesteps", y="Policy Value Norm", data=trex_qlearn)
ax.set(ylabel='Policy Value Norm (w.r.t Ground Truth)')

# plt.legend(["Q-Learning", "Count Based Exploration", "Trex", "Trex (Count-Based)"])
plt.legend(["Q-Learning", "Count Based", "Trex", "Trex++"])
plt.savefig('results/PVN Comparison Plot Four Large Rooms (Mean).png')

