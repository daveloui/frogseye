'''
Implementation of all functions that save data to disk.
'''

import numpy as np
import scripts.plot_utils as plot_utils


def save_mse_data(rewards,
                  preds,
                  filename_prefix,
                  bin_size=100000,
                  discount=0.99,
                  num_steps=5000000,
                  environ='FrogsEye'):
    if environ == 'FrogsEye':
        returns = plot_utils.compute_returns_FrogsEye_experiments(rewards, discount, bin_size=bin_size,
                                                                  num_steps=num_steps)
    sq_errors = (returns - preds[:-bin_size])**2
    np.savez_compressed(filename_prefix + "_values", preds)
    np.savez_compressed(filename_prefix + "_returns", returns)
    np.savez_compressed(filename_prefix + "_rewards", rewards)
    np.savez_compressed(filename_prefix + "_sq_errors", sq_errors)
