'''
Implementation of standard TD-lambda algorithm for the main value function.
'''

import jax
import jax.numpy as jnp


class TDLambda(object):
    def __init__(self,
                 first_x_w,
                 discount=0.0,
                 alpha_w=1.e-4,
                 lambda_w=0.95):
        self.last_x_w = first_x_w
        self.alpha_w = alpha_w
        self.lambda_w = lambda_w
        self.w = jnp.zeros_like(self.last_x_w)
        self.z = jnp.zeros_like(self.last_x_w)
        self.discount = discount
        self.td_error = None

    def predict(self, x_w):
        '''
        Inputs:
            x_w: state representation
        Outputs:
            Main value function evaluated at x_w
        '''
        return jnp.dot(self.w, x_w)

    def update(self, x_w, reward):
        '''
        Inputs:
            x_w: state representation
        What the function does:
            Updates the main value weights, eligibility trace vectors and td error.
            Updates the current features.
        Outputs:
            N/A
        '''
        self.w, self.z = _update(last_x_w=self.last_x_w,
                                 reward=reward,
                                 x_w=x_w,
                                 w=self.w,
                                 z=self.z,
                                 discount=self.discount,
                                 alpha_w=self.alpha_w,
                                 lambda_w=self.lambda_w)
        self.last_x_w = x_w


@jax.jit
def _update(last_x_w, reward, x_w, w, z, discount, alpha_w, lambda_w):
    td_error = reward + discount * jnp.dot(w, x_w) - jnp.dot(w, last_x_w)
    z *= discount * lambda_w
    z += last_x_w
    w += alpha_w * td_error * z
    return w, z
