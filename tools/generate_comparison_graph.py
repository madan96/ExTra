import csv
import pandas as pd
import numpy as np
import argparse

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt


plots = ['Cumulative Reward', 'Mean Average Reward', 'Exp Quality']
plot_freq = 200
exp_check_threshold = 1000

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

def convert_to_mean_df(panda_df, seeds, iters, plot_id):
    if plot_id == 2:
        tp = np.arange(start=0, stop=exp_check_threshold, step=plot_freq)
    elif plot_id == 1:
        tp = np.arange(start=0, stop=iters, step=plot_freq)
    
    cr = panda_df[plots[plot_id]].values[:seeds * int(iters / plot_freq)]
    # print (cr.shape)
    cr = cr.reshape(seeds, int(iters / plot_freq))
    cr = np.mean(cr, axis=0)
    if plot_id == 2:
        cr = cr[:tp.shape[0]]
    # print (cr[0])
    poly_y = smooth(cr, 0.6)
    df = pd.DataFrame({'Timesteps':tp, plots[plot_id]:poly_y})
    return df

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument(
        '--env-name',
        default='FourLargeRooms',
        type=str
    )
    argparser.add_argument(
        '--alg',
        default='pursuit',
        type=str
    )
    argparser.add_argument(
        '--plot-id',
        default=1,
        type=int
    )
    argparser.add_argument(
        '--seeds',
        default=10,
        type=int
    )
    argparser.add_argument(
        '--iters',
        default=10000,
        type=int
    )
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')

    args = argparser.parse_args()

    # Baselines
    base_path = "/home/rishabh/work/btp/TRAiL/tools/{}/{}_{}_qlearn_{}.csv".format(args.env_name, args.env_name, args.alg, args.iters)
    baseline = convert_to_mean_df(pd.read_csv(base_path), args.seeds, args.iters, args.plot_id)
    
    plugin_path = "/home/rishabh/work/btp/TRAiL/tools/{}/{}_trex{}_qlearn_{}.csv".format(args.env_name, args.env_name, args.alg, args.iters)
    plugin = convert_to_mean_df(pd.read_csv(plugin_path), args.seeds, args.iters, args.plot_id)
    
    # Optimal
    optimal_path = "/home/rishabh/work/btp/TRAiL/tools/{}/{}_optimal_mar_data.csv".format(args.env_name, args.env_name)
    optimal = pd.read_csv(optimal_path)

    # Curve data for optimal
    tp = np.arange(start=0, stop=args.iters, step=plot_freq)
    ar = optimal['Mean Average Reward'].values
    ar = np.mean(ar)
    ar = ar.repeat(int(args.iters / plot_freq))
    # ar = smooth(ar, 0.995)
    optimal = pd.DataFrame({'Timesteps':tp, 'Mean Average Reward':ar})

    # warm_qlearn = convert_to_mean_df(pd.read_csv('/home/rishabh/work/btp/TRAiL/tools/NineLarge/NineLargeRooms_warm_qlearn_20000.csv'))
    # plt.title('Comparison Plot')
    plt.grid()

    # Plot Data
    # if args.plot_id == 1:
    #     ax = sns.lineplot(x="Timesteps", y=plots[args.plot_id], data=optimal)
    ax = sns.lineplot(x="Timesteps", y=plots[args.plot_id], data=baseline)
    ax = sns.lineplot(x="Timesteps", y=plots[args.plot_id], data=plugin)
    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=4)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # ax = sns.lineplot(x="Timesteps", y="Mean Average Reward", data=warm_qlearn)
    ax.set_ylabel("", fontsize=18)
    ax.set_xlabel("", fontsize=18)

    # plt.legend(["Q-Learning", "Count Based Exploration", "Trex", "Trex (Count-Based)"])

        # if args.env_name == 'NineLargeRooms':
    plt.legend(["Vanilla", "With T-ReX"], fontsize=22)
    # else:
    #     # if args.env_name == 'NineLargeRooms':
    #     plt.legend(["TReX", "Softmax", "Pursuit", "MBIE-EB", r'$\epsilon$' + '-Greedy'], fontsize=15)
    
    if args.plot_id == 2:
        metric = 'EQ'
    else:
        metric = 'MAR'

    plt.savefig('results/{}_{}_{}.png'.format(metric, args.alg, args.env_name))

    if args.plot_id == 1:
        area_baseline = np.trapz(baseline[plots[args.plot_id]].values, dx=1)
        area_plugin = np.trapz(plugin[plots[args.plot_id]].values, dx=1)
        area_opt = np.trapz(optimal[plots[args.plot_id]].values, dx=1)

        print ("AUC {} TReX: {} % Optimal AUC: {}".format(metric, area_plugin, area_plugin * 100 / area_opt))
        print ("AUC {} {}: {} % Optimal AUC: {}".format(metric, args.alg, area_baseline, area_baseline * 100 / area_opt))
    
    if args.plot_id == 2:
        area_baseline = np.trapz(baseline[plots[args.plot_id]].values, dx=1)
        area_plugin = np.trapz(plugin[plots[args.plot_id]].values, dx=1)

        print ("AUC {} TReX: {}".format(metric, area_plugin))
        print ("AUC {} {}: {}".format(metric, args.alg, area_baseline))
