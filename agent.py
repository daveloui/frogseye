import jax.numpy as jnp
import numpy as np
import tdlambda
import nexting_tdlambda
import features


class Agent(object):
    '''
    Implements all architectures specified in https://arxiv.org/pdf/2106.09776.pdf
        - network with prediction-adaptive neighborhoods,
        - network with random neighborhoods
        - linear architecture
    '''
    def __init__(self,
                 first_obs,
                 num_aux_preds,
                 num_neighbors,
                 num_nonlinear_features,
                 aux_discount=0.99,
                 aux_alpha_w=3.e-6,
                 aux_lambda_w=0.8,
                 main_discount=0.99,
                 main_alpha_w=1.e-6,
                 main_lambda_w=0.8,
                 rep_type='adaptive',
                 seed=0,
                 non_linearity='relu',
                 preactiv_bias='paper_params'):
        self.seed = seed
        self._rand = np.random.RandomState(self.seed)

        if rep_type == 'adaptive':
            self.agent_state = Adaptive_Res_Layer(first_obs=first_obs,
                                                  num_aux_preds=num_aux_preds,
                                                  num_neighbors=num_neighbors,
                                                  num_nonlinear_features=num_nonlinear_features,
                                                  discount=aux_discount,
                                                  alpha_w=aux_alpha_w,
                                                  lambda_w=aux_lambda_w,
                                                  seed=seed,
                                                  non_linearity=non_linearity,
                                                  preactiv_bias=preactiv_bias)

        elif rep_type == 'linear':
            self.agent_state = Linear_Layer(first_obs)

        elif rep_type == 'random':
            self.agent_state = Random_Res_Layer(first_obs=first_obs,
                                                num_aux_preds=num_aux_preds,
                                                num_neighbors=num_neighbors,
                                                num_nonlinear_features=num_nonlinear_features,
                                                seed=seed,
                                                non_linearity=non_linearity,
                                                preactiv_bias=preactiv_bias)

        # pass the first obs (zero vector) to agent_state. Do not update the GVF weights, because the agent has not
        # interacted with the environment at this point yet.
        first_x_w, idxs = self.agent_state.get_features(obs=first_obs, update_GVFs=False)

        # Initialize the main prediction.
        self.main_prediction = tdlambda.TDLambda(first_x_w=first_x_w,
                                                 discount=main_discount,
                                                 alpha_w=main_alpha_w,
                                                 lambda_w=main_lambda_w)


class Linear_Layer(Agent):
    ''' Linear architecture
        The state-representation is simply the observation vector concatenated with a "1"
    '''
    def __init__(self, first_obs):
        self.last_obs = first_obs

    def get_features(self, obs, update_GVFs=None):
        '''
        Computes and outputs the features
        '''
        return jnp.concatenate((obs, jnp.ones(1))), None


class Adaptive_Res_Layer(Agent):
    ''' Prediction-Adapted Neighborhoods architecture '''
    def __init__(self,
                 first_obs,
                 num_aux_preds=5,
                 num_neighbors=3,
                 num_nonlinear_features=3,
                 discount=0.0,
                 alpha_w=1.e-4,
                 lambda_w=0.95,
                 seed=0,
                 non_linearity='relu',
                 preactiv_bias='paper_params'):
        self.num_obs = len(first_obs)
        self.num_aux_preds = num_aux_preds
        self.num_neighbors = num_neighbors
        self.num_nonlinear_features = num_nonlinear_features
        self.seed = seed
        self.non_linearity = non_linearity
        self.preactiv_bias = preactiv_bias

        self.aux_preds = nexting_tdlambda.NextingTDLambda(first_x_w=first_obs,
                                                          discount=discount,
                                                          alpha_w=alpha_w,
                                                          lambda_w=lambda_w,
                                                          num_aux_preds=num_aux_preds)

        self._rand = np.random.RandomState(self.seed)
        self.cumulant_indices = np.sort(self._rand.choice(self.num_obs,
                                                          self.num_aux_preds,
                                                          replace=False))

        self.initialize_matrices_and_weights()

    def initialize_matrices_and_weights(self):
        '''
        Defines the parameters (fully-connected layer) of the network
        '''
        filter_shape = (self.num_nonlinear_features, self.num_neighbors)
        self._rand = np.random.RandomState(self.seed)
        self.A = self._rand.normal(size=filter_shape)

        self._rand = np.random.RandomState(self.seed)
        if self.preactiv_bias == 'paper_params':
            self.bias = float(-4) * jnp.ones(self.num_nonlinear_features)

    def get_features(self, obs, update_GVFs=True):
        '''
        Computes and outputs the non-linear features
        '''
        if update_GVFs:
            self.update(obs)

        if self.non_linearity == 'relu':
            return features._get_adaptive_features_relu(self.aux_preds.w,
                                                        self.num_neighbors,
                                                        obs,
                                                        self.A,
                                                        self.bias, )

    def update(self, obs):
        '''
        Updates the auxiliary prediction (GVF) weights.
        '''
        cumulants = obs[self.cumulant_indices]
        self.aux_preds.update(x_w=obs, cumulants=cumulants)  # stores x_w as last_x_w


class Random_Res_Layer(Agent):
    ''' Random Neighborhoods architecture '''
    def __init__(self,
                 first_obs,
                 num_aux_preds=5,
                 num_neighbors=3,
                 num_nonlinear_features=3,
                 seed=0,
                 non_linearity='relu',
                 preactiv_bias='paper_params'):
        self.num_obs = len(first_obs)
        self.num_aux_preds = num_aux_preds
        self.num_neighbors = num_neighbors
        self.num_nonlinear_features = num_nonlinear_features
        self.seed = seed
        self.non_linearity = non_linearity
        self.preactiv_bias = preactiv_bias

        self.initialize_matrices_and_weights()

        self._rand = np.random.RandomState(self.seed)
        self.random_idxs = np.stack([self._rand.choice(self.num_obs, size=self.num_neighbors, replace=False) for _ in
                                     range(self.num_aux_preds)])

    def initialize_matrices_and_weights(self):
        '''
        Defines the parameters (fully-connected layer) of the network
        '''
        filter_shape = (self.num_nonlinear_features, self.num_neighbors)
        self._rand = np.random.RandomState(self.seed)
        self.A = self._rand.normal(size=filter_shape)

        self._rand = np.random.RandomState(self.seed)
        if self.preactiv_bias == 'paper_params':
            self.bias = float(-4) * jnp.ones(self.num_nonlinear_features)

    def get_features(self, obs, update_GVFs=None):
        '''
        Computes and outputs the non-linear features.
        '''
        return features._get_random_features_relu(obs=obs,
                                                  idxs=self.random_idxs,
                                                  A=self.A,
                                                  b=self.bias)



