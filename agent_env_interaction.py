'''
Implementation of the agent-environment interaction, and storing of results.
'''

import os
from os.path import join
import jax.numpy as jnp
import numpy as np
import frogs_eye_env
import agent
import save_data
import traceback


def run_experiment(args):
    try:
        '''
        Runs a single trial, specified by the "seed" parameter.
        '''
        # read all user inputs:
        results_path = args[0]
        trial = args[1]
        num_steps = args[2]
        bin_size = args[3]
        discount = args[4]
        environ = args[5]
        num_obs = args[6]
        epsilon = args[7]
        architecture = args[8]
        lambda_w = args[9]
        num_aux_preds = args[10]
        num_neighbors = args[11]
        num_nonlinear_features = args[12]
        main_alpha_w = args[13]
        non_linearity = args[14]
        aux_alpha_w = args[15]
        preactiv_bias = args[16]

        # Initialize the environment.
        env = frogs_eye_env.FrogsEyeEnv(num_obs=num_obs,
                                        dyn_seed=trial)

        # Create folder name and file name where data will be saved:
        if architecture == 'linear':
            exp_setting_name = architecture + "_" + env.string + \
                               "_eps{}_n{}_alpha{}_gamma{}_lambda{}".format(epsilon,
                                                                            num_nonlinear_features,
                                                                            main_alpha_w,
                                                                            discount,
                                                                            lambda_w)
        else:
            exp_setting_name = architecture + "_" + env.string + \
                               "_eps{}_n{}_alpha{}_gamma{}_lambda{}_k{}".format(epsilon,
                                                                                num_nonlinear_features,
                                                                                main_alpha_w,
                                                                                discount,
                                                                                lambda_w,
                                                                                num_neighbors)

        if not os.path.exists(results_path):
            os.makedirs(results_path, exist_ok=True)

        folder = os.path.join(results_path + "/" + exp_setting_name.split("_seed=")[0])
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        filename_prefix = join(folder, "seed=" + str(trial))

        # reset environment and generate first observation
        obs = jnp.zeros_like(env.reset())  # This is the true state of the environment

        # Initialize the agent. We don't update the GVF weights:
        sys = agent.Agent(first_obs=obs,
                          num_aux_preds=num_aux_preds,
                          num_neighbors=num_neighbors,
                          num_nonlinear_features=num_nonlinear_features,
                          aux_discount=discount,
                          aux_alpha_w=aux_alpha_w,
                          aux_lambda_w=lambda_w,
                          main_discount=discount,
                          main_alpha_w=main_alpha_w,
                          main_lambda_w=lambda_w,
                          rep_type=architecture,
                          seed=0,
                          non_linearity=non_linearity,
                          preactiv_bias=preactiv_bias)

        # get first feature vector; do not update GVF weights in the process.
        last_x_w, _ = sys.agent_state.get_features(obs=obs, update_GVFs=False)
        # make the first features zero, because the agent has not interacted with the environment yet.
        last_x_w = jnp.zeros_like(last_x_w)

        # Run experiment:
        rewards = []; preds = [];
        for _ in range(num_steps + bin_size):
            obs, reward = env.step()
            # Update the auxiliary prediction weights and construct features.
            x_w, _ = sys.agent_state.get_features(obs=obs)

            # Update data logs.
            preds += [sys.main_prediction.predict(last_x_w)]
            rewards += [reward]

            # Update the main prediction.
            sys.main_prediction.update(x_w=x_w, reward=reward)
            last_x_w = x_w

        # Save data logs.
        preds = np.array(preds)
        rewards = np.array(rewards)
        save_data.save_mse_data(rewards=rewards,
                                preds=preds,
                                filename_prefix=filename_prefix,
                                bin_size=bin_size,
                                discount=discount,
                                num_steps=num_steps,
                                environ=environ)

    except:  # if an error happened, make sure that we print the error specifications
        traceback.print_exc()
