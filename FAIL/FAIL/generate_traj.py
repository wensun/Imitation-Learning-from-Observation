import numpy as np
import gym
from IPython import embed


def generate_partial_trajectory(pi_list, env, env_id, stochastic):
    H = len(pi_list) #pi_0, pi_1, ... pi_{H-1}

    ob_space = env.observation_space
    ob_dim = ob_space.shape[0]
    ac_space = env.action_space
    #ac_dim = ac_space.shape[0]

    traj_obs = [] #x_0, x_1, x_2, ...,X_{H-1}
    traj_res = [] #a_0 = pi_o(x_0), a_1, ..., a_{H-1}
    traj_act = [] #r(x_i)
    traj_act_logp = [] #log(\pi_i(a_i|x_i))

    is_success = False
    ob = env.reset()
    for pi in pi_list:
        ac = pi.act(stochastic, ob)
        logp = pi.logp(ob, ac)
        ob_next, rew, done, info = env.step(ac)

        #if env_id == "Reacher-v2" or env_id == "Swimmer-v2":
        #    rew -= info["reward_ctrl"]
        if env_id[0:5] == 'Fetch' or env_id[0:5] == 'Reach':
            if is_success is False and info['is_success'] == True:
                is_success = True #achieves goal in this task

        traj_obs.append(ob)
        traj_res.append(rew)
        traj_act.append(ac)
        traj_act_logp.append(logp)

        ob = ob_next

    assert len(traj_obs) == len(traj_act) and len(traj_obs)==len(pi_list)
    traj_info = {'traj_obs':np.array(traj_obs), 'traj_rew':np.array(traj_res),
                 'traj_act':np.array(traj_act), 'traj_act_log':np.array(traj_act_logp),
                 'obs_after_pilist':ob, 'is_success':is_success}
    return traj_info


def traj_segment_generate(pi_list, env, env_id, batch_size, stochastic, pi_ref = None):

    obs_after_roll_in = []
    act_ref = []
    act_ref_logp = []
    obs_after_ref =[]

    trajs_total_rew = []
    trajs_obs = []
    trajs_act = []
    trajs_act_logp = []

    success_times = 0

    for i in range(batch_size):

        return_info = generate_partial_trajectory(pi_list, env, env_id, stochastic)

        traj_total_rew = np.sum(return_info['traj_rew'])
        success_times += return_info['is_success']
        trajs_total_rew.append(traj_total_rew)

        traj_obs = return_info['traj_obs']
        trajs_obs.append(traj_obs)

        traj_act = return_info['traj_act']
        trajs_act.append(traj_act)

        traj_act_logp = return_info['traj_act_log']
        trajs_act_logp.append(traj_act_logp)

        ob = return_info['obs_after_pilist']
        obs_after_roll_in.append(ob)
        if pi_ref is not None:
            ac_ref = pi_ref.act(stochastic, ob)
            act_ref.append(ac_ref)
            act_ref_logp.append(pi_ref.logp(ob, np.array(ac_ref)))
            next_ob, rew, done, _ = env.step(ac_ref)
            obs_after_ref.append(next_ob)

    info = {'obs_after_roll_in': np.array(obs_after_roll_in), 'act_ref':np.array(act_ref),
            'act_ref_logp':np.array(act_ref_logp), 'obs_after_ref':np.array(obs_after_ref),
            'avg_rew':np.mean(trajs_total_rew), 'trajs_obs':np.array(trajs_obs),
            'trajs_act':np.array(trajs_act), 'trajs_act_logp':np.array(trajs_act_logp),
            'success_rate':success_times*1./batch_size}

    return info


def traj_segment_generator(pi_list, pi_ref, env,env_id, batch_size,stochastic, intermediate = False):
    '''
    pi_list: a list of policies, 
    pi_ref: the policy to generate test actions. 
    env: gym environment
    batch_size: # of trajectories to generate
    intermediate: if it is true, return whole trajectory-wise information
    '''
    print('num of policies: {0}'.format(len(pi_list)))

    num_policies = len(pi_list)
    #pi_0, ... pi_{num_policies-1}
    #x_0, a_0 = pi_0, x_1, a_1 = pi_1, x_2 ...

    ob_space = env.observation_space
    ob_dim = ob_space.shape[0]

    #ob = env.reset()
    #ac = env.action_space.sample()

    obs_after_roll_in = []
    act_ref = []
    act_ref_prob = []
    act_ref_logp = []
    obs_after_ref = []


    if intermediate == True:
        X_history = []
        A_history = []
        r_history = []
        A_logp_history = []
        X_next_history = []
        for h in range(num_policies):
            X_history.append(np.zeros((batch_size, ob_dim)))
            if isinstance(env.action_space, gym.spaces.Discrete):
                A_history.append(np.zeros(batch_size))
            else:
                A_history.append(np.zeros((batch_size, env.action_space.shape[0])))
            A_logp_history.append(np.zeros(batch_size))
            r_history.append(np.zeros(batch_size)) 
            X_next_history.append(np.zeros((batch_size, ob_dim)))

        valid_nums = [0]*num_policies

    sucess_times = 0
    for i in range(batch_size):
        #roll-in
        ob = env.reset()
        completed = True
        succeed = False
        for h in range(len(pi_list)): #x_h, execute pi_h to generate a_h, and then call dynamics to generate x_{h+1}
            pi = pi_list[h]
            ac = pi.act(stochastic, ob)
            ob_next, rew, done, info = env.step(ac)
            if env_id == "Reacher-v2":
                rew -= info["reward_ctrl"]

            if intermediate is True: #collect x_h, a_h, log(pi(a_h|x_h)), x_{h+1}
                X_history[h][valid_nums[h]] = np.copy(ob)
                A_history[h][valid_nums[h]] = ac #double check for continuous control setting
                A_logp_history[h][valid_nums[h]] = pi.logp(ob, ac)
                X_next_history[h][valid_nums[h]] = np.copy(ob_next)
                valid_nums[h] += 1

            ob = ob_next
            if done is True:
                completed = False
                succeed = True
                break

        if completed is True: #roll-in succssefull, exeucted pi_0, ... pi_-1
        #if succeed is True:
            #sucess_times += 1
            obs_after_roll_in.append(np.copy(ob))
            ac_ref = pi_ref.act(stochastic, ob)
            act_ref.append(ac_ref)
            act_ref_logp.append(pi_ref.logp(ob, ac_ref))
            next_ob, rew, done, _ = env.step(ac_ref)
            obs_after_ref.append(np.copy(next_ob))

    if intermediate is False:
        return np.array(obs_after_roll_in), np.array(act_ref), np.array(act_ref_logp),  np.array(obs_after_ref)
    else:
        return np.array(obs_after_roll_in), np.array(act_ref), np.array(act_ref_logp),  np.array(obs_after_ref), X_history, A_history, A_logp_history, X_next_history, valid_nums, sucess_times/(1.*batch_size)

#test policies:
def test_policies(pi_list, env, stochastic, num_traj = 20):
    total_rews = []
    for i in range(num_traj):
        ob = env.reset()
        trew = 0
        for pi in pi_list:
            ac = pi.act(stochastic, ob)
            ob, rew, done, _ = env.step(ac)
            trew = trew + rew
            if done is True:
                break
        total_rews.append(trew)

    return np.mean(total_rews), np.std(total_rews), total_rews










