'''
Disclaimer: this code is highly based on trpo_mpi at @openai/baselines and @openai/imitation
'''

import argparse
import os.path as osp
import logging
from mpi4py import MPI
from tqdm import tqdm

import numpy as np
import gym

from mlp_policy import *
from minmax import minmax_solver
from parse_expert_data import extract_expert_traj_observations
from baselines.common import set_global_seeds, tf_util as U
from baselines.common.misc_util import boolean_flag
from baselines import bench
from baselines import logger
from generate_traj import *
from IPython import embed
import os
import pickle

def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of FAIL")
    parser.add_argument('--env_id', help='environment ID', default='CartPole-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--expert_path', type=str, default='CartPole-v1expert_traj_linear_deterministic.p')
    parser.add_argument('--policy_hidden_size', type=int, default=64)
    parser.add_argument('--num_hid_layers', help='number of hidden layer in policy: default zero, i.e., linear policy', type=int, default=2)
    parser.add_argument('--task', type=str, choices=['train', 'evaluate', 'sample'], default='train')
    parser.add_argument('--num_timesteps', help='total number of samples', type=int, default=1e6)
    parser.add_argument('--min_max_game_iteration', help='number of iterations in each min-max game', type = int, default = 500)
    parser.add_argument('--num_roll_in', help="num of roll ins in online train model", type=int,default=100)
    parser.add_argument('--adam_lr', help='learning rate in adam', type=float, default=1e-2)
    parser.add_argument('--l2_lambda',help='l2 regularization lambda', type=float, default=1e-7)
    parser.add_argument('--horizon', help='horizon of episode, num of policies should be horizon - 1', type=int, default = 100)
    parser.add_argument('--num_expert_trajs', help='number of expert trajectories', type=int, default=500)
    parser.add_argument('--warm_start', help='initlize pi with the previous trained one', type=int, default=0)
    parser.add_argument('--mixing',help='mixing the previous policy with uniform policy for pi_ref', type=int,default=0)

    return parser.parse_args()

def main(args):
    print(args)
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(args.seed)
    env = gym.make(args.env_id)
    ob_space = env.observation_space
    ac_space = env.action_space
    H = args.horizon
    lr = args.adam_lr
    T = args.min_max_game_iteration
    num_timesteps = args.num_timesteps
    #for each pi_h: we roll_in num_roll_in many times, resulting (h+1)*num_roll_in may interactions
    num_roll_in = int(np.ceil(num_timesteps/((1 + H-1)*(H-1)/2.)))
    print("total number of roll in for training each policy: {0}".format(num_roll_in))
    #initalize pis:
    print("initializing {0} many policies".format(H-1))
    pis = []
    for i in range(H-1): #in total H-1 many policies:
        pi = MlpPolicy('pi'+str(i), reuse = False, ob_space = ob_space, l2_lambda=args.l2_lambda,
                             ac_space = ac_space, hid_size = args.policy_hidden_size,
                             num_hid_layers = args.num_hid_layers)
        pis.append(pi)

    U.initialize()
    env.seed(args.seed)

    #test initial policy performance:
    return_info = traj_segment_generate(pis,env,args.env_id,20, True, None)
    initial_avg_rew = return_info['avg_rew']
    print("before training, avg_rew is {0}".format(initial_avg_rew))

    if args.task == 'train':
        expert_trajs,success_rate=extract_expert_traj_observations(args.expert_path, args.num_expert_trajs)
        assert expert_trajs.ndim == 3
        assert expert_trajs.shape[0] == args.num_expert_trajs
        assert expert_trajs.shape[1] >= H
        final_perf = train_offline(env, args.env_id, pis, expert_trajs, H, num_roll_in, T, lr, warm_start=args.warm_start,mixing = args.mixing)
        #train_online(env, args.env_id, pis, expert_trajs, H, args.num_timesteps, args.num_roll_in, lr)

    #elif args.task == 'evaluate':
    #    avg_traj_rew, std_traj_rew = runner(env,pis, H, stochastic = False, num_traj = 50)
    env.close()
    return final_perf

def main_repeat(args):
    all_perf = []
    #seeds = (np.random.rand(5)*1e6).astype(int)
    print("training on five random seeds...")
    seeds = [608249, 301207, 955449, 107439,  82627]
    all_perf.append(seeds)
    all_perf.append(args)

    for i in range(len(seeds)):
        args.seed = seeds[i]
        print("at the {0} th repeat with seed {1}".format(i, args.seed))
        final_perf = main(args)
        all_perf.append(final_perf)

    pickle.dump(all_perf, open(args.env+"final_perf.p", 'wb'))


def js_from_samples(X1, X2):
    #X1, X2: n x d matrices
    mean_0 = np.mean(X1, axis = 0)
    std_0 = np.std(X1, axis = 0) + 1e-7
    mean_1 = np.mean(X2,axis=0)
    std_1 = np.std(X2,axis=0) + 1e-7

    kl = 0.5*(np.sum(std_0/std_1) + (mean_0-mean_1).dot(np.diag(1./std_1)).dot(mean_0-mean_1)
              - X1.shape[1] + np.log(np.sum(std_1)/np.sum(std_0)))

    ikl = 0.5*(np.sum(std_1/std_0) + (mean_1 - mean_0).dot(np.diag(1./std_0)).dot(mean_1 - mean_0) 
             - X1.shape[1] + np.log(np.sum(std_0)/np.sum(std_1)))

    return (kl+ikl)/2.


def train_offline(env, env_id, pis, expert_trajs, H, num_roll_in, T, lr, warm_start = False, mixing = False):
    #H: length of trajectory, meaning we need to train H-1 many policies
    #at x_h, pi_h results x_{h+1}, with h starting from 0

    #in default, for discrete space env, we use uniform distribution as pi_ref
    #for continues, we set pi_ref as the previous trained policy
    if isinstance(env.action_space, gym.spaces.Discrete):
        uniform_pi = uniform_policy_discrete(env.action_space.n)
    elif isinstance(env.action_space, gym.spaces.Box):
        uniform_pi = uniform_policy_continuous(env.action_space.shape[0],
                                                 env.action_space.low[0], env.action_space.high[0])
    elif isinstance(env.action_space, gym.spaces.MultiDiscrete):
        uniform_pi = uniform_policy_MultiDiscrete(env.action_space.nvec)

    for h in range(H-1):
        print("############## training pi_{0} ###################".format(h))

        if mixing == True and h > 0:
            print("pi_ref resulting from mixing..")
            pi_ref = mix_policy_continuous(uniform_pi, pis[h-1], epsilon=0.5) #mixing with prob 0.5
        else:
            pi_ref = uniform_pi

        roll_in_info = traj_segment_generate(pis[0:h], env, env_id, num_roll_in, stochastic=True, pi_ref = pi_ref)
        X_h = roll_in_info['obs_after_roll_in']
        A_h = roll_in_info['act_ref']
        A_ref_logp = roll_in_info['act_ref_logp']
        X_hp1 = roll_in_info['obs_after_ref']
        avg_rew_sofar = roll_in_info['avg_rew']
        success_rate = roll_in_info['success_rate']
        assert X_h.shape[0] == X_hp1.shape[0]
        print("## at training pi_{0}, collect {1} many triples, avg_rew_so_far {2}, success_rate {3} ##".format(h,
                                                                        X_h.shape[0], avg_rew_sofar, success_rate))
        js = js_from_samples(X_h, expert_trajs[:,h,:])
        print("estimated JS-divergence at step {0} is {1}".format(h, js))
        #note: X is at time step h, and x_next is at h+1

        #extract corresponding expert data at h+1 from expert trajectories
        X_star_hp1 = expert_trajs[:,h+1,:] #(x) from expert at h+1

        #warm start policy pi_{h} by initizing it using pi_{h-1}
        if h > 0 and warm_start is True:
            print("warm start: initlizing the current policy using the previous one..")
            theta_hn1 = pis[h-1].get_traniable_variables_flat()
            pis[h].set_trainable_variable_flat(theta_hn1)

        #solve min_max game here to compute pi_h:
        game_utilities = minmax_solver(X_h, A_h, X_hp1, X_star_hp1, A_ref_logp, pis[h], T, lr)

    final_perf = traj_segment_generate(pis, env, env_id, 200, stochastic=True, pi_ref = None)
    return final_perf


def train_online(env,env_id, pis, expert_trajs, H, num_timesteps, num_roll_in, lr):
    assert len(pis) == H-1
    T = int(np.ceil(num_timesteps*1./(num_roll_in*H*1.)))
    for t in range(T):
        print("#### At iteration {0} out of total {1} iters".format(t, T))
        return_info = traj_segment_generate(pis,env,env_id,num_roll_in, True, None)
        print('collected {0} many trajs, avg rew {1}'.format(return_info['trajs_obs'].shape[0],
                                                             return_info['avg_rew']))
        js_est = train_online_gradient(env, env_id, pis, expert_trajs, num_roll_in, 1, lr, return_info)
        print("at iter {0}, js_est {1}".format(t, js_est))


def train_online_gradient(env, env_id, pis, expert_trajs, num_roll_in, K, lr, return_info = None):
    #use pis to generate a set of trajectories, update pis all together
    H = len(pis) #pi_0, pi_{H-1}
    if return_info is None:
        return_info = traj_segment_generate(pis, env, env_id, num_roll_in, True, None)

    trajs_obs = return_info['trajs_obs']
    assert len(pis) == trajs_obs.shape[1] #x_i <-> pi_i(x_i)
    assert trajs_obs.shape[0] == num_roll_in
    obs_after_roll_in = return_info['obs_after_roll_in'] #x generated from pis[-1]

    js_est = []
    for h in range(H-1): #train pi_0, ... pi_{H-2}
        pi = pis[h]
        X_h = trajs_obs[:,h]
        A_h = return_info['trajs_act'][:,h]
        A_logp_h = return_info['trajs_act_logp'][:,h]
        X_hp1 = trajs_obs[:,h+1]
        X_star_hp1 = expert_trajs[:,h+1]
        game_u = minmax_solver(X_h,A_h,X_hp1,X_star_hp1, A_logp_h, pi, K, lr)
        js_est.append(js_from_samples(X_hp1, X_star_hp1))
        #if h % 4 == 0:
        #js_est.append(game_u)

    #train the pi_{H-1}:
    X_Hn1 = trajs_obs[:,-1]
    A_Hn1 = return_info['trajs_act'][:,-1]
    A_logp_Hn1 = return_info['trajs_act_logp'][:,-1]
    X_H = obs_after_roll_in
    X_star_H = expert_trajs[:,H]
    game_u = minmax_solver(X_Hn1, A_Hn1, X_H, X_star_H, A_logp_Hn1, pis[H-1], K, lr)
    js_est.append(game_u)
    #js_est.append(js_from_samples(X_H,X_star_H))

    #print("js est: {0}".format(js_est))
    return js_est



def runner(env, pis, H, stochastic = True, num_traj = 50):
    assert len(pis) == H-1 #in total, H-1 many policies geneartes x_0, to x_{H-1} ---length H traj
    avg_traj_rew, std_traj_rew, _ = test_policies(pis, env, stochastic=stochastic, num_traj =num_traj)
    print("average total rew: {0}, std total rew: {1}".format(avg_traj_rew, std_traj_rew))
    return avg_traj_rew, std_traj_rew


if __name__ == '__main__':
    args = argsparser()
    final_perf = main(args)
    filename = "data/fail_" + args.env_id + "_" + str(args.num_expert_trajs) + "_" + str(args.num_timesteps) + "_" + str(args.seed) +".p"
    # pickle.dump([args, final_perf], open(args.env_id+"seed_{0}".format(args.seed),'wb'))
    pickle.dump([args, final_perf], open(filename,'wb'))


