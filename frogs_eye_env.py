import numpy as np
import matplotlib.pyplot as plt
'''
Implementation of the Frog's Eye environment, as specified in the paper: https://arxiv.org/pdf/2106.09776.pdf
'''


class FrogsEyeEnv(object):
    def __init__(self,
                 num_obs=4000,
                 epsilon=0.5,
                 box_len=16,
                 sensor_radius=0.6,
                 particle_radius=.5,
                 particle_scale=0.05,
                 reward_radius=.5,
                 decay_rate=0.01,
                 dyn_seed=0,
                 obs_seed=0):
        self.string = 'FrogsEye'
        self.dyn_seed = dyn_seed
        self.obs_seed = obs_seed
        self._dyn_rand = np.random.RandomState(self.dyn_seed)
        self._obs_rand = np.random.RandomState(self.obs_seed)

        self.num_obs = num_obs
        self.epsilon = epsilon
        self.box_len = box_len
        self.sensor_radius = sensor_radius
        self.particle_radius = particle_radius
        self.particle_scale = particle_scale
        self.reward_radius = reward_radius
        self.decay_rate = decay_rate

        # Initialize sensor locations.
        deltas = self._obs_rand.rand(self.num_obs, 2)
        self.sensor_locs = np.array([[dx * self.box_len - self.box_len * 0.5,
                                      dy * self.box_len - self.box_len * 0.5]
                                     for (dx, dy) in deltas])

        # Pre-compute initialization locations for the particle.
        self.init_locs = []
        num_init_locs = 0
        while num_init_locs < 100:
            loc = self.box_len * (self._obs_rand.rand(2) - 0.5)
            d = np.sqrt(np.sum(loc ** 2))
            if d <= (self.particle_radius + self.reward_radius):  # d < self.reward_radius:
                continue
            else:
                num_init_locs += 1
                self.init_locs.append(np.copy(loc))

        self.reset_count = 0

    def _get_obs(self):
        obs = np.copy(self.sensor_state)
        noise = self._obs_rand.rand(self.num_obs)
        obs[noise < self.epsilon / 2] = 0
        obs[noise > 1 - self.epsilon / 2] = 1
        return obs

    def _get_reward(self):
        dist = np.sqrt(np.sum(self.particle_loc ** 2))
        if dist < self.particle_radius + self.reward_radius:
            return 1.
        return 0.

    def reset(self):
        self.reset_count += 1
        # Select random initial position for particle.
        idx = self._dyn_rand.choice(len(self.init_locs))
        self.particle_loc = self.init_locs[idx]

        # Initialize and activate sensors based on the particle location.
        dists = np.sqrt(np.sum((self.sensor_locs - self.particle_loc) ** 2, axis=-1))
        activate = dists < self.sensor_radius + self.particle_radius
        self.sensor_state = np.zeros(self.num_obs)
        self.sensor_state[activate] = 1.

        # Return the noisy sensor outputs.
        self.obs = self._get_obs()
        return self.obs

    def step(self):
        # Reset dynamics.
        dist = np.sqrt(np.sum(self.particle_loc ** 2))
        if dist < self.particle_radius + self.reward_radius:
            self.obs = self.reset()
        elif np.any(np.abs(self.particle_loc) > self.box_len * 0.5):  # You moved past the edge of the box
            self.obs = self.reset()
        else:
            self.particle_loc = (1. - self.decay_rate) * self._dyn_rand.normal(loc=self.particle_loc,
                                                                               scale=self.particle_scale, size=(2,))
            dists = np.sqrt(np.sum((self.sensor_locs - self.particle_loc) ** 2, axis=-1))
            turn_on = dists < self.sensor_radius + self.particle_radius
            self.sensor_state[:] = 0.
            self.sensor_state[turn_on] = 1.
            self.obs = self._get_obs()

        self.reward = self._get_reward()
        return self.obs, self.reward

    def plot_env(self, fname):
        fig, ax = plt.subplots(figsize=(10, 10))
        fig.tight_layout()
        lim = self.box_len * 0.5
        ax.set_xlim((-lim, lim))
        ax.set_ylim((-lim, lim))
        ax.axis('off')

        circle = plt.Circle(self.particle_loc,
                            self.particle_radius,
                            color='grey',
                            alpha=0.5)
        ax.add_patch(circle)
        ax.scatter(self.particle_loc[0], self.particle_loc[1], marker='x')

        circle = plt.Circle(np.zeros(2),
                            self.reward_radius,
                            color='red',
                            alpha=0.5)
        ax.add_patch(circle)

        idxs = self.obs == 1
        ax.scatter(self.sensor_locs[:, 0],
                   self.sensor_locs[:, 1],
                   marker='+',
                   color='gold',
                   s=50)
        ax.scatter(self.sensor_locs[idxs, 0],
                   self.sensor_locs[idxs, 1],
                   marker='+',
                   color='blue',
                   s=50)

        plt.savefig(fname)
        plt.close()
