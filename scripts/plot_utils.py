'''
Code used for processing data before saving it, and for plotting data
'''

import numpy as np
import scipy
import scipy.stats

  
def bin_vector_FrogsEye_experiments(x, num_bins):
  '''
  Separates the data into bins and averages over each bin.
  '''
  j = int(len(x) / num_bins)
  return np.array([np.mean(x[i * j:(i + 1) * j]) for i in range(num_bins)])


def mean_confidence_interval(data, confidence=0.95):
    """
    Code obtained from the link below:
    https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a, axis=0), scipy.stats.sem(a, axis=0)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2., n - 1)
    return m, se, h


def compute_returns_FrogsEye_experiments(rewards, discount, bin_size=100000, num_steps=5000000):
    '''
    Computes the returns using dynamic programming to save computational complexity
    '''
    assert len(rewards) == num_steps+bin_size
    assert len(rewards) >= bin_size
    list_returns = []
    N = len(rewards)
    truncated_return = 0

    # compute the return backwards
    for t in range(N-1, -1, -1):  # R_{num_steps + bin_size}, ..., R_1, R_0
        truncated_return = rewards[t] + discount * truncated_return
        list_returns += [truncated_return]
    list_returns.reverse()
    return np.array(list_returns)[:-bin_size]
