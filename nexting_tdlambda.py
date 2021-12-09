'''
TD-lambda implementation for auxiliary predictions (General Value Functions)
'''
import jax
import jax.numpy as jnp
import functools


class NextingTDLambda(object):
  def __init__(self,
               first_x_w,
               discount=0.0,
               alpha_w=1.e-4,
               lambda_w=0.95,
               num_aux_preds=2):
    self.last_x_w = first_x_w
    self.alpha_w = alpha_w
    self.lambda_w = lambda_w
    self.num_aux_preds = num_aux_preds
    num_obs = len(self.last_x_w)
    self.w = jnp.zeros((self.num_aux_preds, num_obs))
    self.z = jnp.zeros((self.num_aux_preds, num_obs))
    self.discount = discount

  def predict(self, x_w):
    '''
    Inputs:
      x_w: observation from the environment
    Outputs:
      General value function evaluated at x_w
    '''
    return jnp.matmul(self.w, x_w)
    
  def update(self, x_w, cumulants):
    '''
    Inputs:
      x_w: observation from the environment
      cumulants: array of cumulants; each cumulant corresponds to a specific GVF
    What the function does:
      Updates the GVF weights and eligibility trace vectors
      Updates the current observation.
    Outputs:
      N/A
    '''
    self.w, self.z = _update(self.last_x_w,
                             cumulants,
                             x_w,
                             self.w,
                             self.z,
                             self.discount,
                             self.alpha_w,
                             self.lambda_w)
    self.last_x_w = x_w

  
@functools.partial(jax.jit, static_argnums=(5, 6, 7))
@functools.partial(jax.vmap, in_axes=(None, 0, None, 0, 0, None, None, None))
def _update(last_x_w, cumulant, x_w, w, z, discount, alpha_w, lambda_w): 
  td_error = cumulant + discount * jnp.dot(w, x_w) - jnp.dot(w, last_x_w)
  z *= discount * lambda_w
  z += last_x_w
  w += alpha_w * td_error * z
  return w, z
