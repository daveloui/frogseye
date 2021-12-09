import gin
import os
import itertools
import numpy as np
import multiprocessing as mp
from agent_env_interaction import run_experiment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def run_sequential_experiments(experimental_configs):
    for config in experimental_configs:
        run_experiment(config)


def run_parallel_experiments(experimental_configs):
    print("inside run_parallel_experiments")
    num_procs = mp.cpu_count()
    print("inside else statement; num_procs =", num_procs)
    print("number of CPUs =", num_procs)
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(1. / num_procs)
    with mp.Pool(processes=num_procs) as p:
        p.map(run_experiment, experimental_configs)


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
        print(" num_aux_preds =", num_aux_preds)
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
    if parallel_experiments:
        run_parallel_experiments(experimental_configs)
    else:
        run_sequential_experiments(experimental_configs)


if __name__ == '__main__':
    print("in main_relu.py")
    gin.parse_config_file('config_relu_FrogsEye.gin')
    print("parsed gin file")
    run_experiments()