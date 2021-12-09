'''
Computes and saves the average binned square return error for each configuration, across all trials.
'''

import gin
import os
import itertools
import plot_utils
import numpy as np


def compute_plotting_data(configs):
    '''
    Inputs: experiment configurations
    What this function does:
        - computes the average binned squared return error across all trials
        - generates a list of missing results and prints it
    '''
    list_missing_files = []
    for i, config in enumerate(configs):
        results_dir, env, num_seeds, epsilon, arch, num_features, step_size, discount, lambda_w, num_bins, \
        num_neighbors, bin_size, recreate_paper_results, num_steps = config

        # make sure that sub_folder where results are saved exists. If not, skip it and move to the next one.
        if 'adaptive' in arch or 'random' in arch:
            results_subdir = "{}_{}_eps{}_n{}_alpha{}_gamma{}_lambda{}_k{}".format(arch,
                                                                                   env,
                                                                                   epsilon,
                                                                                   num_features,
                                                                                   step_size,
                                                                                   discount,
                                                                                   lambda_w,
                                                                                   num_neighbors)
        else:
            results_subdir = "{}_{}_eps{}_n{}_alpha{}_gamma{}_lambda{}".format(arch,
                                                                               env,
                                                                               epsilon,
                                                                               num_features,
                                                                               step_size,
                                                                               discount,
                                                                               lambda_w)
        if not os.path.isdir(os.path.join(results_dir, results_subdir)):
            print(os.path.join(results_dir, results_subdir) + " does not exist!")
            continue

        print("results_subdir", results_subdir, " exists")
        data = []
        for seed in range(num_seeds):
            fname = 'seed={}_sq_errors.npz'.format(seed)
            results_file = os.path.join(results_dir, results_subdir, fname)
            # Load squared error data.
            try:
                sq_errors = np.load(results_file, allow_pickle=True)['arr_0']
            except FileNotFoundError:
                print('Results file {} not found!'.format(results_file))
                list_missing_files += [os.path.join(results_subdir, fname)]
                continue
            if sq_errors.shape[0] > num_steps:
                sq_errors = sq_errors[:num_steps]
            binned_sq_errors = plot_utils.bin_vector_FrogsEye_experiments(sq_errors, num_bins=num_bins)

            # save binned squared errors
            np.savez_compressed(os.path.join(results_dir, results_subdir, 'seed={}_binned_SE.npz'.format(seed)),
                                binned_sq_errors)
            data.append(binned_sq_errors)

        # Compute and save mean and standard errors.
        mean, se, _ = plot_utils.mean_confidence_interval(np.array(data))
        mse_results_file = os.path.join(results_dir, results_subdir, 'mse')
        se_results_file = os.path.join(results_dir, results_subdir, 'se')
        np.savez_compressed(mse_results_file, mean)
        np.savez_compressed(se_results_file, se)

    print("number of missing files for architecture {} =".format(arch), len(list_missing_files))
    print(list_missing_files)
    print("")


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
                 recreate_paper_results=True,
                 ):
    print("num_steps =", num_steps, "  bin_size=", bin_size)
    num_bins = int(num_steps / bin_size)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Unpack configuration
    configs = zip(architecture, step_size_base_range, step_size_exp_range)
    for arch, (bmin, bmax), (emin, emax) in configs:
        step_sizes = np.unique([np.round(b * 10 ** -x, decimals=x + 1)
                                for x in range(emin, emax, -1)
                                for b in range(bmin, bmax, 1)])
        print("arch =", arch, " step_sizes =", step_sizes)

        config = itertools.product((results_dir,),
                                   env,
                                   num_lifetimes,
                                   epsilon,
                                   (arch,),
                                   num_features,
                                   step_sizes,
                                   discount,
                                   lambda_w,
                                   (num_bins,),
                                   num_neighbors,
                                   (bin_size,),
                                   (recreate_paper_results,),
                                   (num_steps,),
                                   )
        # generate and save plots:
        compute_plotting_data(config)


if __name__ == '__main__':
    gin.parse_config_file('plotting_config.gin')
    print("parsed gin file")
    create_plots()
