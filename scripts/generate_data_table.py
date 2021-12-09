'''
Saves the final average return error for each architecture, under its best step-size
'''

import sys
from os import path
import gin
import os
from os.path import join
import numpy as np
sys.path.append(path.dirname(path.abspath(__file__)))


@gin.configurable
def create_plots(num_lifetimes=(30,),
                 num_steps=int(5e6),
                 bin_size=int(1e5),
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
                 num_neighbors=(10,),
                 aux_step_size=(3e-6,),
                 recreate_paper_results=False,
                 ):
    '''
    Generates an array that has dimensions:
        num lambdas x num epsilons x num architectures x num alphas + 2 x num neighbors
    The 4th dimension in the table contains 2 extra entries: the final average return error and best step-size.
    '''
    # generate folder where plots will be saved:
    folder_plots = output_dir
    if not os.path.exists(folder_plots):
        os.makedirs(folder_plots, exist_ok=True)

    # make sure that data has been saved in the correct folder:
    folder_data_saved = results_dir
    if not os.path.exists(folder_data_saved):
        print("results folder DOES NOT EXITS -- {}".format(folder_data_saved))
        raise OSError

    configs = {}
    alphas_list = []
    # Unpack configuration
    configs_zip = zip(architecture, step_size_base_range, step_size_exp_range)
    for arch, (bmin, bmax), (emin, emax) in configs_zip:
        step_sizes = np.unique([np.round(b * 10 ** -x, decimals=x + 1)
                                for x in range(emin, emax, -1)
                                for b in range(bmin, bmax, 1)])
        print("arch =", arch, " step_sizes =", step_sizes)
        configs[arch] = step_sizes
        alphas_list.append(step_sizes)

    lambda_data = []
    for lambda_w_main in lambda_w:
        eps_data_arrays = []
        for eps in epsilon:
            eps_data = []
            for representation in architecture:
                rep_MSE_data = []
                alphas_rep = configs[representation]
                for alpha in alphas_rep:
                    k_data = []
                    # Retrieve average binned square error data for each architecture:
                    if 'adaptive' in representation or 'random' in representation:
                        for k in num_neighbors:
                            exp_setting_name = "{}_{}_eps{}_n{}_alpha{}_gamma{}_lambda{}_k{}".format(representation,
                                                                                                     env[0],
                                                                                                     eps,
                                                                                                     num_features[0],
                                                                                                     alpha,
                                                                                                     discount[0],
                                                                                                     lambda_w_main,
                                                                                                     k)
                            folder = join(folder_data_saved, exp_setting_name)
                            if not os.path.isdir(folder):
                                print("folder {} DOES NOT EXIST".format(folder))
                                raise FileNotFoundError
                            try:
                                final_mse = np.load(join(folder, "mse.npz"))['arr_0'][-1]
                                k_data += [final_mse]
                            except FileNotFoundError:
                                print("mse file not found in folder", folder)
                                raise OSError

                    else:
                        exp_setting_name = "{}_{}_eps{}_n{}_alpha{}_gamma{}_lambda{}".format(representation,
                                                                                             env[0],
                                                                                             eps,
                                                                                             num_features[0],
                                                                                             alpha,
                                                                                             discount[0],
                                                                                             lambda_w_main)
                        folder = join(folder_data_saved, exp_setting_name)
                        if not os.path.isdir(folder):
                            print("folder {} DOES NOT EXIST".format(folder))
                            raise FileNotFoundError
                        try:
                            final_mse = np.load(join(folder, "mse.npz"))['arr_0'][-1]
                            k_data = list(final_mse * np.ones_like(np.arange(len(num_neighbors))))
                        except FileNotFoundError:
                            print("mse file not found in folder", folder)
                            raise OSError

                    rep_MSE_data += [k_data]
                eps_data += [rep_MSE_data]

            # find smallest average binned squared error across all step-sizes and all number of neighbors
            max_num_alphas = max(alphas_rep.shape[0] for alphas_rep in alphas_list)
            for i, rep_MSE_data in enumerate(eps_data):
                # Make all lists have the same number of elements:
                num_alphas = len(rep_MSE_data)
                need_to_add = max_num_alphas - num_alphas
                for j in range(need_to_add):
                    infs = [np.inf for _ in num_neighbors]
                    rep_MSE_data.append(infs)
                rep_MSE_arr = np.array(rep_MSE_data)

                # find minimum average binned squared error across step-sizes, for each k value under epsilon
                min_MSE_rep = np.min(rep_MSE_arr, axis=0)
                eps_data[i].append(list(min_MSE_rep))

                # find best step-size for each architecture:
                best_alpha_idx = np.argmin(rep_MSE_arr, axis=0)
                # add best step-size index to end of list:
                repr_str = architecture[i]
                alphas_rep = configs[repr_str]
                best_alpha = alphas_rep[best_alpha_idx]
                eps_data[i].append(list(best_alpha))
                print(repr_str, "best step size =", best_alpha)

            eps_data = np.array(eps_data)
            eps_data_arrays += [eps_data]
        lambda_data += [eps_data_arrays]
    arr_lambda = np.array(lambda_data)
    arr_edited = arr_lambda[:, :, :, -2:, :]
    print("table =", arr_edited)
    np.save(join(folder_data_saved, "best_step_sizes.npy"), arr_edited)


if __name__ == '__main__':
    gin.parse_config_file('plotting_config.gin')
    create_plots()
