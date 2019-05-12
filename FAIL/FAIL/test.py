import numpy as np
from mlp_policy import *
import gym
from generate_traj import traj_segment_generator, test_policies

#env = gym.make("Hopper-v2")
env = gym.make("CartPole-v1")
#env = gym.make("MountainCar-v0")
ob_space = env.observation_space
ac_space = env.action_space

print(ac_space)


#pi_ref = MlpPolicy('pi_ref', reuse = False, ob_space = ob_space, ac_space = ac_space, hid_size = 10, num_hid_layers=2)
pi_ref = uniform_policy_discrete(ac_space.n)
#pi2 = MlpPolicy('oldpi', reuse = False, ob_space = ob_space, ac_space = ac_space, hid_size = 10, num_hid_layers=2)
pis = []
for i in range(3):
    pi = MlpPolicy('pi'+str(i), reuse = False, ob_space = ob_space, ac_space = ac_space, hid_size = 10, num_hid_layers = 0)
    pis.append(pi)

U.initialize()

X, A, A_ref_logp, X_next = traj_segment_generator(pis, pi_ref, env, 10, True)

mean_rew, rew_std, total_rews = test_policies(pis, env, stochastic = True, num_traj = 10)





#theta = pi.get_traniable_variables_flat()
#theta_old = pi2.get_traniable_variables_flat()

#print(theta)
#print(theta_old)

#pi2.set_trainable_variable_flat(theta)
#print(pi2.get_traniable_variables_flat())


#obs = np.random.randn(5, ob_space.shape[0])
#act = [0, 0, 0, 1, 1]

#ratios = np.random.rand(5)
#probs = np.exp(pi.logps(obs, act))

#print(pi.dot_p_r(obs, act, ratios))
#print(ratios.dot(probs)/5.)

#print(np.exp(pi.logps(np.array(obs),np.array(act))))


#for i in range(100):
#    val = pi.update_policy(obs, act, ratios)
#    print(val)





#stoch = True
#ob = np.random.randn(ob_space.shape[0])
#print(ob)
#act = pi.act(stoch, ob)
#act = act[0]
#act = mean[0]

#print(act.shape)
#print(act)
#print(logits)
#prob = np.exp(logits[0])
#prob = prob/np.sum(prob)
#print(prob[act])




#logp = pi.logp(ob, act)
#print(np.exp(logp))

#d = mean[0] - act
#cov = np.eye(ac_space.shape[0])

#log_likelihood = -0.5*d.dot(d) - np.log((2*np.pi)**1.5) + np.log(np.linalg.det(cov))

#-0.5*(np.log(np.linalg.det(cov)) + d.T.dot(d) + np.log(2*np.pi))
#print(np.exp(log_likelihood))

#print()

