import numpy as np
from gym import utils,spaces
from gym.envs.mujoco import mujoco_env

class ReacherEnv_discretize(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)

        self.K = 5
        a_dim = self.action_space.shape[0]
        highs = np.copy(self.action_space.high)
        lows = np.copy(self.action_space.low)

        self.action_space = spaces.MultiDiscrete([self.K]*a_dim)
        self.action_space.nvec = self.action_space.nvec.astype('int32')
        self.actions = np.zeros((a_dim, self.K))

        self.goal_threshold = 0.05

        for i in range(a_dim):
            delta = (highs[i] - lows[i])/(self.K-1)
            self.actions[i,:] = np.array([lows[i]+delta*k for k in range(self.K)])


    def is_success(self):
        if np.linalg.norm(self.get_body_com('fingertip') - self.get_body_com('target')) <= 0.05:
            return True
        else:
            return False

    def step(self, a_index):
        #assert a_index.ndim == 1 and a_index.dtype == int
        if a_index.ndim == 1 and (a_index.dtype==int or a_index.dtype=='int32'):
            a = np.zeros(a_index.shape[0])
            for i in range(self.actions.shape[0]):
                a[i] = self.actions[i, a_index[i]]
        else:
            print('input action form is wrong, double check..')
            a = a_index

        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = 0 #- np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        info = {'is_success': self.is_success()}
        done = False
        return ob, reward, done, info #dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < 2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])
