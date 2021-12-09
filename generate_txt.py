'''
Reads from configuration file and generates txt file to be used in each job
'''
import gin
import os
import itertools
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def generate_job_arrays(experimental_configs):
    '''
    Reads experiment configuration and appends it as a line in txt file, with respective flags.
    '''
    permutation_file = open("FrogEye_parameter_sweep.txt", "w")

    for config in experimental_configs:
        results_path = config[0]
        trial = config[1]
        num_steps = config[2]
        bin_size = config[3]
        discount = config[4]
        environ = config[5]
        num_obs = config[6]
        epsilon = config[7]
        architecture = config[8]
        lambda_w = config[9]
        num_aux_preds = config[10]
        num_neighbors = config[11]
        num_nonlinear_features = config[12]
        main_alpha_w = config[13]
        non_linearity = config[14]
        aux_alpha_w = config[15]
        preactiv_bias = config[16]

        job_str = "--results_path=" + results_path + " --trial=" + str(trial) + \
                  " --num_steps=" + str(num_steps) + " --bin_size=" + str(bin_size) + " --discount=" + str(discount) + \
                  " --environ=" + str(environ) + " --num_obs=" + str(num_obs) + " --epsilon=" + str(epsilon) + \
                  " --architecture=" + str(architecture) + " --lambda_w=" + str(lambda_w) + " --num_aux_preds=" + \
                  str(num_aux_preds) + " --num_neighbors=" + str(num_neighbors) + " --num_nonlinear_features=" + \
                  str(num_nonlinear_features) + " --main_alpha_w=" + str(main_alpha_w) + " --non_linearity=" + \
                  str(non_linearity) + " --aux_alpha_w=" + str(aux_alpha_w) + " --preactiv_bias=" + str(preactiv_bias)+\
                  "\n"

        permutation_file.write(job_str)


@gin.configurable
def run_experiments(results_dir_name='results',
                    parallel_experiments=True,
                    num_trials=30,
                    num_steps=int(5e6),
                    bin_size=int(1e5),
                    discount=0.99,
                    env='FrogsEye',
                    num_obs=4000,
                    epsilon=(0.5,),
                    architecture=('adaptive', 'random', 'linear',),
                    lambda_w=(0.8,),
                    num_aux_preds=4000,
                    num_neighbors=(10,),
                    num_nonlinear_features=100,
                    step_size_base_range=((1, 2), (3, 4), (3, 4),),
                    step_size_exp_range=((6, 5), (6, 5), (6, 5),),
                    non_linearity=('relu', 'relu', 'relu',),
                    aux_step_size=(3e-6,),
                    preactiv_bias='paper_params',
                    ):
    '''
    Iterates through experiment configuration and calls generate_job_arrays function
    '''

    print("inside run_experiments function, results_dir_name =", results_dir_name)
    print("d =", num_obs, " m =", num_aux_preds, " k =", num_neighbors, " n =", num_nonlinear_features)
    print("bin_size =", bin_size, " T =", num_steps)

    # generate folder where data will be saved
    results_path = results_dir_name
    if not os.path.exists(results_path):
        os.makedirs(results_path, exist_ok=True)

    # iterate through experiment configurations and call generate_job_arrays
    configs = []
    configs_zip = zip(architecture, step_size_base_range, step_size_exp_range, non_linearity)
    for arch, (bmin, bmax), (emin, emax), non_lin in configs_zip:
        step_size = np.unique([np.round(b * 10 ** -x, decimals=x + 1)
                               for x in range(emin, emax, -1)
                               for b in range(bmin, bmax, 1)])
        print("arch = {}, step_size={}".format(arch, step_size))

        if ('adaptive' not in arch) and ('random' not in arch):
            num_neighbors = (0,)
        config = itertools.product((results_path,),
                                   range(num_trials),
                                   (num_steps,),
                                   (bin_size,),
                                   (discount,),
                                   (env,),
                                   (num_obs,),
                                   epsilon,
                                   (arch,),
                                   lambda_w,
                                   (num_aux_preds,),
                                   num_neighbors,
                                   (num_nonlinear_features,),
                                   step_size,
                                   (non_lin,),
                                   aux_step_size,
                                   (preactiv_bias,),
                                   )
        configs.append(config)

    experimental_configs = itertools.chain.from_iterable(configs)
    generate_job_arrays(experimental_configs)


if __name__ == '__main__':
    gin.parse_config_file('config_relu_FrogsEye.gin')
    run_experiments()
