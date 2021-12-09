'''
Plots average squared return error for all architectures, using their best step-size
'''

import sys
import os
from os import path
from os.path import join
import gin
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
sys.path.append(path.dirname(path.abspath(__file__)))


def plot_mse(folder_data_saved, eps_data_arrays, algs, epsilons,
             folder_save_plots, gamma, env, num_features, bin_size, lambda_w_main, num_neighbors,
             arch_color=('forestgreen', 'orange', 'royalblue',),
             arch_style=('-', '-', '-',)):
    '''
    Plot timesteps on x-axis, MSE on y-axis for all architectures under their best step-sizes;
    The plot is generated for a given epsilon and gamma parameters.
    '''
    mpl.rcParams.update({'font.size': 14})
    for j, epsilon in enumerate(epsilons):
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, alg in enumerate(algs):
            best_alpha_alg_eps = eps_data_arrays[j, i][-1]
            print("eps {}, alg {}, best_alpha {}".format(epsilon, alg, best_alpha_alg_eps))
            if 'adaptive' in alg or 'random' in alg:
                for k_idx, k in enumerate(num_neighbors):
                    alpha_k = best_alpha_alg_eps[k_idx]
                    subfolder = join(folder_data_saved, '{}_{}_eps{}_n{}_alpha{}_gamma{}_lambda{}_k{}'.format(alg,
                                                                                                              env,
                                                                                                              epsilon,
                                                                                                              num_features,
                                                                                                              alpha_k,
                                                                                                              gamma,
                                                                                                              lambda_w_main,
                                                                                                              k))

                    mse = np.load(join(subfolder, 'mse.npz'), allow_pickle=True)['arr_0']
                    se = np.load(join(subfolder, "se.npz"), allow_pickle=True)['arr_0']
                    x_axis = np.arange(1, len(mse) + 1) * bin_size
                    ax.errorbar(x_axis, mse, yerr=se, label=alg, color=arch_color[i], ls=arch_style[i])

            elif alg == 'linear':
                alpha_k = best_alpha_alg_eps[0]
                subfolder = join(folder_data_saved, '{}_{}_eps{}_n{}_alpha{}_gamma{}_lambda{}'.format(alg,
                                                                                                      env,
                                                                                                      epsilon,
                                                                                                      num_features,
                                                                                                      alpha_k,
                                                                                                      gamma,
                                                                                                      lambda_w_main))
                mse = np.load(join(subfolder, 'mse.npz'), allow_pickle=True)['arr_0']
                se = np.load(join(subfolder, "se.npz"), allow_pickle=True)['arr_0']
                x_axis = np.arange(1, len(mse) + 1) * bin_size
                ax.errorbar(x_axis, mse, yerr=se, label=alg, color=arch_color[i], ls=arch_style[i])

        ax.set_xlabel('Time steps')
        ax.set_ylabel('Mean Squared Return Error')
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.legend(loc='upper right')
        ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
        fig.tight_layout()
        plt.savefig(join(folder_save_plots, "mean_squared_return_error.png"))
        print("saved file", join(folder_save_plots, "mean_squared_return_error.png"))
        plt.close()


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
                 step_size_base_range=((3, 4), (3, 4), (1, 2),),
                 step_size_exp_range=((6, 5), (6, 5), (6, 5),),
                 num_features=(100,),
                 lambda_w=(0.8,),
                 bin_size=100000,
                 num_neighbors=(10,),
                 aux_step_size=(3e-6,),
                 recreate_paper_results=False,
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
    # Unpack configuration
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
        arr = arr_lambda[l, :, :, :, :] # contains the final mean return error and its corresponding step-size
        plot_mse(folder_data_saved, arr, architecture, epsilon, folder_plots, discount[0], env[0], num_features[0],
                 bin_size, lambda_w_main=lambda_w_main, num_neighbors=num_neighbors, arch_color=arch_color,
                 arch_style=arch_style)


if __name__ == '__main__':
    gin.parse_config_file('plotting_config.gin')
    create_plots()
