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
from gym import wrappers

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


def record_video(monitored_env, pis, env_id, ep_id):
    print("recording video for {}".format(env_id))
    if env_id[0:5] == "Swimm" or env_id[0:5] == "Hoppe":
        H = 300
    else:
        H = 50
    num_policies = len(pis)
    success = False
    images = []
    #ob = monitored_env.reset()

    trials = 20
    for trial in range(trials):
        ob = monitored_env.reset()
        traj_images = []
        for i in range(H):
            monitored_env.env._render_callback()
            img = monitored_env.env.sim.render(500,500)
            #print(img.shape)
            traj_images.append(img)
            #if i < ep_id:
            #    action = pis[i].act(True,ob)
            #else:
            #    action = pis[ep_id-1].act(True,ob)
            pi_id = i%num_policies
            #print(pi_id)
            action = pis[pi_id].act(True,ob)
            ob_next, rew,done,info = monitored_env.step(action)
            ob = ob_next;
            if info['is_success'] == True:
                images += traj_images
                break;

    filename = "video_record/{0}_{1}.p".format(env_id, ep_id)

    print("at ep: {0}_{1}".format(ep_id, np.array(images).shape))
    #pickle.dump(np.array(images), open(filename, 'wb'))
    np.save(filename, np.array(images))
    print("Done recording this run")



def main(args):
    print(args)
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(args.seed)
    env = gym.make(args.env_id)

    if args.env_id[0:5] == "Swimm":
        env_monitor = gym.make("Swimmer-v2")
    elif args.env_id[0:5] == "Hoppe":
        env_monitor = gym.make("Hopper-v2")
    else:
        env_monitor = gym.make(args.env_id)
    #env_monitor = wrappers.Monitor(env_2, 'video/{0}'.format(args.env_id),force=True)
    #env_monitor = env_2

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

    if os.path.exists(args.expert_path) is not True:
        os.system("wget http://kalman.ml.cmu.edu/FAIL_datasets/{}".format(args.expert_path))

    #test initial policy performance:
    return_info = traj_segment_generate(pis,env,args.env_id,20, True, None)
    initial_avg_rew = return_info['avg_rew']
    print("before training, avg_rew is {0}".format(initial_avg_rew))

    if args.task == 'train':
        expert_trajs,success_rate=extract_expert_traj_observations(args.expert_path, args.num_expert_trajs, args.env_id)
        assert expert_trajs.ndim == 3
        assert expert_trajs.shape[0] == args.num_expert_trajs
        assert expert_trajs.shape[1] >= H
        final_perf = train_offline(env, args.env_id, pis, expert_trajs, H, num_roll_in, T, lr, warm_start=args.warm_start,mixing = args.mixing, monitor_env = env_monitor)
        #train_online(env, args.env_id, pis, expert_trajs, H, args.num_timesteps, args.num_roll_in, lr)

    env.close()
    return final_perf


def js_from_samples(X1, X2, env_id = None):
    #X1, X2: n x d matrices
    mean_0 = np.mean(X1, axis = 0)
    std_0 = np.std(X1, axis = 0) + 1e-7
    mean_1 = np.mean(X2,axis=0)
    std_1 = np.std(X2,axis=0) + 1e-7

    kl = 0.5*(np.sum(std_0/std_1) + (mean_0-mean_1).dot(np.diag(1./std_1)).dot(mean_0-mean_1)
              - X1.shape[1] + np.log(np.sum(std_1)/np.sum(std_0)))

    ikl = 0.5*(np.sum(std_1/std_0) + (mean_1 - mean_0).dot(np.diag(1./std_0)).dot(mean_1 - mean_0) 
             - X1.shape[1] + np.log(np.sum(std_0)/np.sum(std_1)))

    if env_id[0:5] == "Fetch":
        X1_sub = X1[:,0:9]
        X2_sub = X2[:,0:9]
        avg_dis_to_goal = [np.mean(np.sum(X1_sub[:,:3]*X1_sub[:,:3],axis=1)**0.5), np.mean(np.sum(X1_sub[:,3:6]*X1_sub[:,3:6],axis=1)**0.5), np.mean(np.sum(X1_sub[:,6:9]*X1_sub[:,6:9],axis=1)**0.5)]
        avg_dis_to_goal_exp = [np.mean(np.sum(X2_sub[:,:3]*X2_sub[:,:3],axis=1)**0.5), np.mean(np.sum(X2_sub[:,3:6]*X2_sub[:,3:6],axis=1)**0.5), np.mean(np.sum(X2_sub[:,6:9]*X2_sub[:,6:9],axis=1)**0.5)]
    else:
        X1_sub = X1[:,-15:]
        X2_sub = X2[:,-15:]
        avg_dis_to_goal = np.mean(np.sum(X1_sub*X1_sub,axis=1)**0.5)
        avg_dis_to_goal_exp = np.mean(np.sum(X2_sub*X2_sub,axis=1)**0.5)


    return (kl+ikl)/2.,avg_dis_to_goal,avg_dis_to_goal_exp



def train_online(env, env_id, pis, expert_trajs, H, num_roll_in, T, lr, warm_start = False, mixing = False, batch_size = 512):

    for h in range(H-1):
        print("############## training pi_{0} ###################".format(h))
        num_batches = np.ceil(num_roll_in/batch_size)
        print("total number of batches here: {0}".format(num_batches))
        for epoch in range(int(num_batches)):
            #roll in with pi_0, ..., pi_{h-1}, then use pi_h as ref
            roll_in_info = traj_segment_generate(pis[0:h], env, env_id, batch_size, stochastic=True, pi_ref = pis[h])
            #train pi_0, pi_1, .., pi_{h-1}:
            game_us = []
            for ih in range(h):
                pi = pis[ih]
                X_ih = roll_in_info['trajs_obs'][:,ih]
                A_ih = roll_in_info['trajs_act'][:,ih]
                A_log_ih = roll_in_info['trajs_act_logp'][:,ih]
                if ih+1 == h:
                    X_ihp1 = roll_in_info['obs_after_roll_in']
                else:
                    X_ihp1 = roll_in_info['trajs_obs'][:,ih+1]
                game_u = minmax_solver(X_ih,A_ih, X_ihp1, expert_trajs[:,ih+1], A_log_ih, pi, 1, lr)
                game_us.append(game_u)

            #train pi_h:
            X_h = roll_in_info['obs_after_roll_in']
            A_h = roll_in_info['act_ref']
            A_ref_logp = roll_in_info['act_ref_logp']
            X_hp1 = roll_in_info['obs_after_ref']
            avg_rew_sofar = roll_in_info['avg_rew']
            success_rate = roll_in_info['success_rate']
            #print("## at training pi_{0}, collect {1} many triples, avg_rew_so_far {2}, success_rate {3} ##".format(h,X_h.shape[0], avg_rew_sofar, success_rate))
            X_star_hp1 = expert_trajs[:,h+1,:]
            game_utilities = minmax_solver(X_h, A_h, X_hp1, X_star_hp1, A_ref_logp, pis[h], 1, lr)
            game_us.append(game_utilities)

            if epoch%5 == 0:
                print(game_us)

    final_perf = traj_segment_generate(pis, env, env_id, 200, stochastic=True, pi_ref = None)
    return final_perf


def train_one_policy(h,env, env_id, pis, expert_trajs, H, num_roll_in, T, lr, pi_ref, warm_start, train_all = False):
    roll_in_info = traj_segment_generate(pis[0:h], env,env_id, num_roll_in, stochastic=True, pi_ref = pi_ref)

    X_h = roll_in_info['obs_after_roll_in']
    A_h = roll_in_info['act_ref']

    if h > 0:
        tmp_A = roll_in_info['trajs_act'][:, h-1]
        print(np.mean(np.sum(tmp_A*tmp_A,axis=1)**0.5))

    A_ref_logp = roll_in_info['act_ref_logp']
    X_hp1 = roll_in_info['obs_after_ref']
    avg_rew_sofar = roll_in_info['avg_rew']
    success_rate = roll_in_info['success_rate']
    print("## at training pi_{0}, collect {1} many triples, avg_rew_so_far {2}, success_rate {3} ##".format(h,X_h.shape[0], avg_rew_sofar, success_rate))

    js,avg_dis_to_goal, avg_dis_to_goal_exp = js_from_samples(X_h, expert_trajs[:,h,:], env_id)
    print('estimated JS-divergence at step {0} is {1}, avg dis to goal {2}, expert avg dis to goal {3}'.format(h, js, avg_dis_to_goal, avg_dis_to_goal_exp))

    X_star_hp1 = expert_trajs[:,h+1,:] #(x) from expert at h+1
    game_utilities = minmax_solver(X_h, A_h, X_hp1, X_star_hp1, A_ref_logp, pis[h], T, lr, h=h+1)

    if train_all == True: #option: train all previous policy using the generated data here. 
        for t in range(h): #train pi_0, pi_1, ..., pi_{h-1} using the generate data as well:
            print("--------at training {0}-th policy inside training {1}-th policy----------".format(t,h))
            X_t = roll_in_info['trajs_obs'][:,t] #X_t
            A_t = roll_in_info['trajs_act'][:,t] #A_t
            X_tp1 = roll_in_info['trajs_obs'][:,t+1] if t < h-1 else X_h #X_h is the states generated after executing x_{h-1}
            A_ref_logp_t = roll_in_info['trajs_act_logp'][:,t] #logp of A_t
            X_star_tp1 = expert_trajs[:,t+1]
            game_us = minmax_solver(X_t, A_t, X_tp1, X_star_tp1, A_ref_logp_t, pis[t], int(T/2), lr = 1e-3, h=None)



def train_offline(env, env_id, pis, expert_trajs, H, num_roll_in, T, lr, warm_start = False, mixing = False, monitor_env = None):
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

        #if monitor_env is not None and h > 0:
            #record_video(monitor_env, pis[0:h], env_id, h)

        if mixing == True and h > 0:
            print("pi_ref resulting from mixing..")
            pi_ref = mix_policy_continuous(uniform_pi, pis[h-1], epsilon=0.5) #mixing with prob 0.5
        else:
            pi_ref = uniform_pi

        repeats = 5
        if h > 0 and warm_start:
            print('warm start...')
            theta_hn1 = pis[h-1].get_traniable_variables_flat()
            pis[h].set_trainable_variable_flat(theta_hn1)

        for repeat in range(repeats):
            print("at repeat {}".format(repeat))
            pi_ref = uniform_pi if repeat == 0 else pis[h] # pis[h]
            train_one_policy(h, env,env_id, pis, expert_trajs, H, int(num_roll_in/repeats), T, lr, pi_ref, warm_start, train_all=True)

    final_perf = traj_segment_generate(pis, env, env_id, 200, stochastic=True, pi_ref = None)
    return final_perf


def train_online_2(env,env_id, pis, expert_trajs, H, num_timesteps, num_roll_in, lr):
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


