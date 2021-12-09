'''
Plots sensitivity curves over step-sizes for multiple runs
'''

import sys
from os import path
import gin
import os
from os.path import join
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.append(path.dirname(path.abspath(__file__)))


def plot_sensitivity_curves_mse(
        algs,
        epsilon,
        gamma,
        configs,
        results_subfolder,
        filename,
        env,
        num_features,
        lambda_w_main,
        num_neighbors,
        arch_color,
        arch_style
):
    '''
    Generates a plot with the following specifications:
        x-axis = step-sizes
        y-axis = average binned squared error across trials.
    All architectures shown.
    '''

    mpl.rcParams.update({'font.size': 10})
    fig, ax = plt.subplots(figsize=(10, 5))  # a single plot for a single epsilon value
    ax.set_ylabel('Final Average Binned Squared Return Error')

    data_adap = [];
    data_rand = [];
    data_lin = [];
    alphas_adap = configs['adaptive'];
    alphas_random = configs['random'];
    alphas_lin = configs['linear']

    for alg in algs:
        alphas_alg = configs[alg]
        for alpha_w in alphas_alg:
            data_alpha = []
            # retrieve average binned square error of each architecture:
            if 'adaptive' in alg or 'random' in alg:
                for k_idx, k in enumerate(num_neighbors):
                    subfolder = join(results_subfolder,
                                     '{}_{}_eps{}_n{}_alpha{}_gamma{}_lambda{}_k{}'.format(alg,
                                                                                           env,
                                                                                           epsilon,
                                                                                           num_features,
                                                                                           alpha_w,
                                                                                           gamma,
                                                                                           lambda_w_main,
                                                                                           k))
                    if not os.path.exists(subfolder):
                        print("folder ", subfolder, " does not exist")
                        raise FileNotFoundError

                    try:
                        mse = np.load(join(subfolder, "mse.npz"), allow_pickle=True)['arr_0']
                        se = np.load(join(subfolder, "se.npz"), allow_pickle=True)['arr_0']
                        data_alpha.append((mse[-1], se[-1]))
                    except FileNotFoundError:
                        print("file {} or {} does not exist!".format(join(subfolder, "mse.npz"),
                                                                     join(subfolder, "se.npz")))
                        raise FileNotFoundError

            elif alg == 'linear':
                subfolder = join(results_subfolder, '{}_{}_eps{}_n{}_alpha{}_gamma{}_lambda{}'.format(alg,
                                                                                                      env,
                                                                                                      epsilon,
                                                                                                      num_features,
                                                                                                      alpha_w,
                                                                                                      gamma,
                                                                                                      lambda_w_main))
                if not os.path.exists(subfolder):
                    raise FileNotFoundError

                try:
                    mse = np.load(join(subfolder, "mse.npz"), allow_pickle=True)['arr_0']
                    se = np.load(join(subfolder, "se.npz"), allow_pickle=True)['arr_0']
                    data_alpha.append((mse[-1], se[-1]))
                except FileNotFoundError:
                    print("file {} or {} does not exist!".format(join(subfolder, "mse.npz"), join(subfolder, "se.npz")))
                    raise OSError

            if alg == 'adaptive':
                data_adap.append(data_alpha)
            elif alg == 'random':
                data_rand.append(data_alpha)
            elif alg == 'linear':
                data_lin.append(data_alpha)

    data_adap = np.array(data_adap)
    data_rand = np.array(data_rand)
    data_lin = np.array(data_lin)

    # Plot data:
    for i, alg in enumerate(algs):
        if alg == 'linear':
            mse = data_lin[:, k_idx, 0].flatten()
            se = data_lin[:, k_idx, 1].flatten()
            ax.errorbar(alphas_lin, mse, yerr=se, label=alg, color=arch_color[i], ls=arch_style[i])

            ymin = min(data_lin[:, k_idx, 0])
            xmin = alphas_lin[np.argmin(data_lin[:, k_idx, 0])]
            plt.scatter(xmin, ymin, marker='*', s=70, color=arch_color[i])

        elif alg == 'adaptive':
            for k_idx, k in enumerate(num_neighbors):
                mse = data_adap[:, k_idx, 0].flatten()
                se = data_adap[:, k_idx, 1].flatten()
                ax.errorbar(alphas_adap, mse, yerr=se, label=alg, color=arch_color[i], ls=arch_style[i])

                ymin = min(data_adap[:, k_idx, 0])
                xmin = alphas_adap[np.argmin(data_adap[:, k_idx, 0])]
                plt.scatter(xmin, ymin, marker='*', s=70, color=arch_color[i])

        elif alg == 'random':
            for k_idx, k in enumerate(num_neighbors):
                mse = data_rand[:, k_idx, 0].flatten()
                se = data_rand[:, k_idx, 1].flatten()
                ax.errorbar(alphas_random, mse, yerr=se, label=alg, color=arch_color[i], ls=arch_style[i])

                ymin = min(data_rand[:, k_idx, 0])
                xmin = alphas_random[np.argmin(data_rand[:, k_idx, 0])]
                plt.scatter(xmin, ymin, marker='*', s=70, color=arch_color[i])

    ax.set_xlabel('Step size')
    ax.set_xscale('log')
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.legend(loc='center right')
    fig.tight_layout()
    # plt.show()
    plt.savefig(filename)
    plt.close()
    print("saved figure =", filename)


@gin.configurable
def create_plots(num_lifetimes=(30,),
                 num_steps=5000000,
                 results_dir="results",
                 output_dir="output",
                 epsilon=(0.5,),
                 env=('FrogsEye',),
                 discount=(0.99,),
                 architecture=('linear', 'random', 'adaptive',),
                 arch_color=('forestgreen', 'orange', 'royalblue',),
                 arch_style=('-', '-', '-',),
                 step_size_exp_range=((6, 5), (6, 5), (6, 5),),
                 step_size_base_range=((3, 4), (3, 4), (1, 2),),
                 num_features=(100,),
                 lambda_w=(0.8,),
                 bin_size=100000,
                 num_neighbors=(10,),
                 aux_step_size=(3e-6,),
                 recreate_paper_results=True,
                 ):
    # generate folder where plots will be saved:
    folder_plots = output_dir
    if not os.path.exists(folder_plots):
        os.makedirs(folder_plots, exist_ok=True)

    # make sure that data has been saved in the correct folder
    folder_data_saved = results_dir
    if not os.path.exists(folder_data_saved):
        print("results folder DOES NOT EXITS -- {}".format(folder_data_saved))
        raise OSError

    configs = {}
    # Unpack configuration.
    configs_zip = zip(architecture, step_size_base_range, step_size_exp_range)
    for arch, (bmin, bmax), (emin, emax) in configs_zip:
        step_sizes = np.unique([np.round(b * 10 ** -x, decimals=x + 1)
                                for x in range(emin, emax, -1)
                                for b in range(bmin, bmax, 1)])
        print("arch =", arch, " step_sizes =", step_sizes)
        configs[arch] = step_sizes
    arr_lambda = np.load(join(folder_data_saved, "best_step_sizes.npy"), allow_pickle=True)

    for l in range(arr_lambda.shape[0]):
        lambda_w_main = lambda_w[l]
        for i, eps in enumerate(list(epsilon)):
            print("epsilon", eps)
            # generate plots filename.
            filename = join(folder_plots,
                            "sensitivity_curve_epsilon={}_gamma={}_lambda={}.png".format(eps,
                                                                                         discount[0],
                                                                                         lambda_w_main)
                            )
            # plot sensitivity curves for all algorithms.
            plot_sensitivity_curves_mse(architecture,
                                        eps,
                                        discount[0],
                                        configs,
                                        results_subfolder=folder_data_saved,
                                        filename=filename,
                                        env=env[0],
                                        num_features=num_features[0],
                                        lambda_w_main=lambda_w_main,
                                        num_neighbors=num_neighbors,
                                        arch_color=arch_color,
                                        arch_style=arch_style)


if __name__ == '__main__':
    gin.parse_config_file('plotting_config.gin')
    create_plots()
