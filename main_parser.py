'''
Parses user inputs and launches experiment
'''

import sys
import os
import argparse
from agent_env_interaction import run_experiment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def parameter_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--results_path', action='store', dest='results_path',
                        default='/Users/fatima_davelouis/projects/prediction-adapted-networks/results/FrogsEye_relu_paper_params',
                        help='Input results_path')

    parser.add_argument('--trial', action='store', dest='trial', default='0', help='Trial index', type=int)

    parser.add_argument('--num_steps', dest='num_steps', default='5000000', help='Number of time steps', type=int)

    parser.add_argument("--bin_size", action='store', dest='bin_size', default='100000', help='bin_size', type=int)

    parser.add_argument("--discount", action='store', dest='discount', default='0.99', help='discount', type=float)

    parser.add_argument("--environ", action='store', dest='environ', default='FrogsEye', help='environment')

    parser.add_argument("--num_obs", action='store', dest='num_obs', default='4000', help='num_obs', type=int)

    parser.add_argument("--epsilon", action='store', dest='epsilon', default='0.5', help='epsilon', type=float)

    parser.add_argument("--architecture", action='store', dest='architecture', default='adaptive', help='architecture')

    parser.add_argument("--lambda_w", action='store', dest='lambda_w', default='0.8',
                        help='eligibility trace for the main value', type=float)

    parser.add_argument("--num_aux_preds", action='store', dest='num_aux_preds', default='4000',
                        help='num_aux_preds', type=int)

    parser.add_argument("--num_neighbors", action='store', dest='num_neighbors', default='10',
                        help='num_neighbors', type=int)

    parser.add_argument("--num_nonlinear_features", action='store', dest='num_nonlinear_features', default='100',
                        help='num_nonlinear_features', type=int)

    parser.add_argument("--main_alpha_w", action='store', dest='main_alpha_w', default='1e-06',
                        help='main_alpha_w', type=float)

    parser.add_argument("--non_linearity", action='store', dest='non_linearity', default='relu', help='non_linearity')

    parser.add_argument("--aux_alpha_w", action='store', dest='aux_alpha_w', default='3e-06',
                        help='aux_alpha_w', type=float)

    parser.add_argument("--preactiv_bias", action='store', dest='preactiv_bias', default='4',
                        help='preactiv_bias') #, type=int)

    parameters = parser.parse_args()

    return parameters


def main():
    parameters = parameter_parser()
    print("parsed params successfully")

    if not os.path.exists(parameters.results_path):
        os.makedirs(parameters.results_path, exist_ok=True)

    try:
        parameters.preactiv_bias = int(parameters.preactiv_bias)
    except ValueError:
        pass

    all_params = (parameters.results_path,
                  parameters.trial,
                  parameters.num_steps,
                  parameters.bin_size,
                  parameters.discount,
                  parameters.environ,
                  parameters.num_obs,
                  parameters.epsilon,
                  parameters.architecture,
                  parameters.lambda_w,
                  parameters.num_aux_preds,
                  parameters.num_neighbors,
                  parameters.num_nonlinear_features,
                  parameters.main_alpha_w,
                  parameters.non_linearity,
                  parameters.aux_alpha_w,
                  parameters.preactiv_bias,)

    run_experiment(all_params)


if __name__ == '__main__':
    print(sys.argv[0])  # prints main_parser.py
    print(sys.argv[1])  # prints var1
    print(sys.argv[2])  # prints var2
    main()
