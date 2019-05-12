import os
import numpy as np

from gym import utils,spaces
from gym.envs.robotics import fetch_env
import numpy as np
from gym.envs.robotics import utils as r_utils

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'slide.xml')


class FetchSlideEnv_discrete(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.05,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.7, 1.1, 0.4, 1., 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=-0.02, target_in_the_air=False, target_offset=np.array([0.4, 0.0, 0.0]),
            obj_range=0.1, target_range=0.3, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
    
        highs = self.action_space.high
        lows = self.action_space.low
        highs[0:3] = highs[0:3]*0.05
        lows[0:3] = lows[0:3]*0.05
        self.K = 5

        if self.block_gripper is True:
            a_dim = 3
        else:
            a_dim = 4

        obs = self._get_obs()
        raw = self.get_raw_obs(obs)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=raw.shape, dtype='float32')

        self.action_space = spaces.MultiDiscrete([self.K]*a_dim)
        self.action_space.nvec = self.action_space.nvec.astype('int32')
        self.real_actions = np.zeros((a_dim, self.K))
        for i in range(a_dim):
            delta = (highs[i] - lows[i])/(self.K - 1)
            self.real_actions[i,:] = np.array([lows[i] + delta*k for k in range(self.K)])



    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()
        pos_ctrl, gripper_ctrl = action[:3], action[3]
        rot_ctrl = [1., 0., 1., 0.]
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])
        r_utils.ctrl_set_action(self.sim, action)
        r_utils.mocap_set_action(self.sim, action)

    def get_raw_obs(self, obs):
        raw_obs = np.copy(obs['observation'])
        if self.has_object is True:
            raw_obs[3:6] -= obs['desired_goal'] #3-6 stores the achieved goal
        else:
            raw_obs[0:3] -= obs['desired_goal'] #0-3 stores the achieved goal
        return raw_obs

    def step(self, action_index):
        assert action_index.shape == (3,)
        if action_index.ndim == 1 and (action_index.dtype==int or action_index.dtype=='int32'):
            action = np.zeros(action_index.shape[0]+1) #add the gripper open
            for i in range(self.real_actions.shape[0]):
                action[i] = self.real_actions[i, action_index[i]]
        else:
            print("input action form is wrong, double check ..")
            action = np.zeros(self.real_actions.shape[0]+1)

        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()
        done = False
        info = {'is_success': self._is_success(obs['achieved_goal'], self.goal),}
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)

        raw_obs = self.get_raw_obs(obs)
        return raw_obs, reward, done, info

    def reset(self):
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal().copy()
        obs = self._get_obs()
        raw_obs = self.get_raw_obs(obs)
        return raw_obs
