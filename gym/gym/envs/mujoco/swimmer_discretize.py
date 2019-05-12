import numpy as np
from gym import utils,spaces
from gym.envs.mujoco import mujoco_env

class SwimmerEnv_discretize(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'swimmer.xml', 4)
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

        ctrl_cost_coeff = 0.000
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        reward_fwd = (xposafter - xposbefore) / self.dt
        reward_ctrl = - ctrl_cost_coeff * np.square(a).sum()
        reward = reward_fwd + reward_ctrl
        ob = self._get_obs()
        return ob, reward, False, dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos.flat[2:], qvel.flat])

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        )
        return self._get_obs()
