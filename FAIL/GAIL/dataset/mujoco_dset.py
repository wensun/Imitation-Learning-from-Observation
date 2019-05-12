'''
Data structure of the input .npz:
the data is save in python dictionary format with keys: 'acs', 'ep_rets', 'rews', 'obs'
the values of each item is a list storing the expert trajectory sequentially
a transition can be: (data['obs'][t], data['acs'][t], data['obs'][t+1]) and get reward data['rews'][t]
'''

from baselines import logger
import numpy as np
import random
import pickle
import ipdb


class Dset(object):
    def __init__(self, inputs, labels, randomize):
        self.inputs = inputs
        self.labels = labels
        assert len(self.inputs) == len(self.labels)
        self.randomize = randomize
        self.num_pairs = len(inputs)
        self.init_pointer()

    def init_pointer(self):
        self.pointer = 0
        if self.randomize:
            idx = np.arange(self.num_pairs)
            np.random.shuffle(idx)
            self.inputs = self.inputs[idx, :]
            self.labels = self.labels[idx, :]

    def get_next_batch(self, batch_size):
        # if batch_size is negative -> return all
        if batch_size < 0:
            return self.inputs, self.labels
        if self.pointer + batch_size >= self.num_pairs:
            self.init_pointer()
        end = self.pointer + batch_size
        inputs = self.inputs[self.pointer:end, :]
        labels = self.labels[self.pointer:end, :]
        self.pointer = end
        return inputs, labels


class Mujoco_Dset(object):
    def __init__(self, expert_path, train_fraction=0.7, traj_limitation=-1, randomize=True):
        traj_data = np.load(expert_path)
        if traj_limitation < 0:
            traj_limitation = len(traj_data['obs'])
        obs = traj_data['obs'][:traj_limitation]
        acs = traj_data['acs'][:traj_limitation]

        # obs, acs: shape (N, L, ) + S where N = # episodes, L = episode length
        # and S is the environment observation/action space.
        # Flatten to (N * L, prod(S))
        self.obs = np.reshape(obs, [-1, np.prod(obs.shape[2:])])
        self.acs = np.reshape(acs, [-1, np.prod(acs.shape[2:])])

        self.rets = traj_data['ep_rets'][:traj_limitation]
        self.avg_ret = sum(self.rets)/len(self.rets)
        self.std_ret = np.std(np.array(self.rets))
        if len(self.acs) > 2:
            self.acs = np.squeeze(self.acs)
        assert len(self.obs) == len(self.acs)
        self.num_traj = min(traj_limitation, len(traj_data['obs']))
        self.num_transition = len(self.obs)
        self.randomize = randomize
        self.dset = Dset(self.obs, self.acs, self.randomize)
        # for behavior cloning
        self.train_set = Dset(self.obs[:int(self.num_transition*train_fraction), :],
                              self.acs[:int(self.num_transition*train_fraction), :],
                              self.randomize)
        self.val_set = Dset(self.obs[int(self.num_transition*train_fraction):, :],
                            self.acs[int(self.num_transition*train_fraction):, :],
                            self.randomize)
        self.log_info()

    def log_info(self):
        logger.log("Total trajectorues: %d" % self.num_traj)
        logger.log("Total transitions: %d" % self.num_transition)
        logger.log("Average returns: %f" % self.avg_ret)
        logger.log("Std for returns: %f" % self.std_ret)

    def get_next_batch(self, batch_size, split=None):
        if split is None:
            return self.dset.get_next_batch(batch_size)
        elif split == 'train':
            return self.train_set.get_next_batch(batch_size)
        elif split == 'val':
            return self.val_set.get_next_batch(batch_size)
        else:
            raise NotImplementedError

    def plot(self):
        import matplotlib.pyplot as plt
        plt.hist(self.rets)
        plt.savefig("histogram_rets.png")
        plt.close()


class FAIL_Dset(object):
    def __init__(self, expert_path, num_trajs, randomize=True):        
        self.expert_path = expert_path
        self.num_trajs = num_trajs
        self.randomize = randomize

        self.data = pickle.load(open(expert_path, "rb"))
        self.total_num_trajs = len(self.data)
        self.horizon_length= len(self.data[0])
        self.observation_dim = self.data[0][0][0][0].shape[0]
        self.total_num_observations = self.total_num_trajs * self.horizon_length

        self.extract_num = np.min((self.total_num_trajs, self.num_trajs))

        random.shuffle(self.data)
        self.load_expert_data()
        self.init_pointer()

    def init_pointer(self):
        self.pointer = 0
        if self.randomize:
            random.shuffle(self.data)
            num_points = self.expert_data.shape[0]
            idx = np.arange(num_points)
            np.random.shuffle(idx)
            self.expert_data = self.expert_data[idx, :]

    def load_expert_data(self):
        self.expert_data = np.zeros((self.extract_num, self.horizon_length, self.observation_dim))
        total_rewards = []

        for i in range(self.extract_num):
            traj_i = self.data[i]
            max_T = np.min((self.horizon_length, len(traj_i)))
            total_reward = 0.
            assert max_T == self.horizon_length
            for t in range(max_T):
                self.expert_data[i, t, :] = traj_i[t][0][0]
                total_reward += traj_i[t][2][0]

                if t > 1:
                    assert np.linalg.norm(traj_i[t][0][0] - traj_i[t-1][-1][0]) <= 1e-2

            total_rewards.append(total_reward)

        #  FIX: GAIL doesn't differentiate between state distributions at different time-steps.
        # It tries to match state distributions at all timesteps
        self.expert_data = np.reshape(self.expert_data, [-1, self.observation_dim])

        #randomize:
        print("randomized dataset")
        num_points = self.expert_data.shape[0]
        idx = np.arange(num_points)
        np.random.shuffle(idx)
        self.expert_data = self.expert_data[idx,:]
        assert self.expert_data.shape[0] == self.extract_num*self.horizon_length

        self.mean_total_reward, self.std_total_reward = np.mean(total_rewards), np.std(total_rewards)
        print('Mean reward obtained by expert is', self.mean_total_reward, 'and std is', self.std_total_reward, 'Number of unique observations in expert data is', np.unique(self.expert_data, axis=0).shape[0])

    def get_next_batch(self, batch_size):
        if batch_size < 0:
            return self.expert_data
        if batch_size > self.extract_num * self.horizon_length:
            #  FIX: for the nan bug
            # Batch size is bigger than expert dataset
            self.init_pointer()
            expert_dataset_size = self.extract_num * self.horizon_length
            num_repeat = batch_size // expert_dataset_size
            expert_data = np.copy(self.expert_data)
            for _ in range(num_repeat-1):
                self.init_pointer()
                expert_data = np.concatenate([expert_data, self.expert_data], axis=0)
            num_remaining = batch_size - num_repeat * expert_dataset_size
            expert_data = np.concatenate([expert_data, self.expert_data[:num_remaining, :]], axis=0)
            self.pointer = num_remaining
            return expert_data
        if self.pointer + batch_size >= self.extract_num * self.horizon_length:
            self.init_pointer()
        end = self.pointer + batch_size
        expert_data = np.copy(self.expert_data[self.pointer:end, :])
        self.pointer = end
        return expert_data

def test(expert_path, traj_limitation, plot):
    dset = Mujoco_Dset(expert_path, traj_limitation=traj_limitation)
    if plot:
        dset.plot()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_path", type=str, default="../data/deterministic.trpo.Hopper.0.00.npz")
    parser.add_argument("--traj_limitation", type=int, default=None)
    parser.add_argument("--plot", type=bool, default=False)
    args = parser.parse_args()
    test(args.expert_path, args.traj_limitation, args.plot)
