import numpy as np
from gym import utils,spaces
from gym.envs.mujoco import mujoco_env

class HopperEnv_discrete(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'hopper.xml', 4)
        utils.EzPickle.__init__(self)

        self.K = 5
        a_dim = self.action_space.shape[0]
        highs = np.copy(self.action_space.high)
        lows = np.copy(self.action_space.low)

        self.action_space = spaces.MultiDiscrete([self.K]*a_dim)
        self.action_space.nvec = self.action_space.nvec.astype('int32')
        self.actions = np.zeros((a_dim, self.K))
        for i in range(a_dim):
            delta = (highs[i] - lows[i])/(self.K-1)
            self.actions[i,:] = np.array([lows[i]+delta*k for k in range(self.K)])

    def step(self, a_index):

        if a_index.ndim == 1 and (a_index.dtype==int or a_index.dtype=='int32'):
            a = np.zeros(a_index.shape[0])
            for i in range(self.actions.shape[0]):
                a[i] = self.actions[i, a_index[i]]
        else:
            print('input action form is wrong, double check..')
            a = a_index

        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 0.
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 0 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            np.clip(self.sim.data.qvel.flat, -10, 10)
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20
