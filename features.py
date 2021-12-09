""" Functions that compute and return non-linear features using JAX """

import functools
import jax
import jax.numpy as jnp


# functions for Adaptive architecture:
@functools.partial(jax.jit, static_argnums=(1,))                    # before vmap, w_bar has size m x d
@functools.partial(jax.vmap, in_axes=(0, None, None, None, None))   # w_bar has size  d (parallelize over neighborhoods)
def _get_adaptive_nonlinear_features_relu(w_bar, k, x_w, A, b):
    _, ell = jax.lax.top_k(jnp.abs(w_bar), k)                       # ell has size k
    ell_sorted = jnp.sort(ell)
    x_w_ell = x_w[ell_sorted]                                       # x_w_ell has size k
    return jax.nn.relu(jnp.matmul(A, x_w_ell) + b), ell_sorted      # returns a vector of size n

  
@functools.partial(jax.jit, static_argnums=(1,))
def _get_adaptive_features_relu(w, k, obs, A, b):
    h, ell_sorted = _get_adaptive_nonlinear_features_relu(w, k, obs, A, b)   # h has size m x n
    return jnp.concatenate((obs, h.flatten(), jnp.ones(1))), ell_sorted      # a "1" is concatenated, because the main prediction has a bias unit


# functions for Random architecture:
@jax.jit
@functools.partial(jax.vmap, in_axes=(0, None, None, None))
def _get_random_nonlinear_features_relu(idxs, obs, A, b):
    neighborhood = obs[idxs]
    return jax.nn.relu(jnp.matmul(A, neighborhood) + b), idxs


@jax.jit
def _get_random_features_relu(obs, idxs, A, b):
    h, idxs = _get_random_nonlinear_features_relu(idxs, obs, A, b)
    return jnp.concatenate((obs, h.flatten(), jnp.ones(1))), idxs
