'''
from baselines/ppo1/mlp_policy.py and add simple modification
(1) add reuse argument
(2) cache the `stochastic` placeholder
'''
import tensorflow as tf
import gym
import numpy as np

import baselines.common.tf_util as U
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common.distributions import make_pdtype
from baselines.acktr.utils import dense


class MlpPolicy(object):
    recurrent = False

    def __init__(self, name, reuse=False, *args, **kwargs):
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

        #print(self.scope)
        #set up for functions to get and set policy parameters:
        var = self.get_trainable_variables()
        var_list = [v for v in var]
        self.flatten_var = tf.concat(axis=0, values=[tf.reshape(v, [U.numel(v)]) for v in var_list])

        self.get_flat = U.GetFlat(var_list)
        self.set_from_flat = U.SetFromFlat(var_list)

        self.setupadam(self.l2_reg)


    def _init(self, ob_space, l2_lambda, ac_space, hid_size, num_hid_layers):
        #assert isinstance(ob_space, gym.spaces.Box)

        self.l2_reg = l2_lambda

        self.pdtype = pdtype = make_pdtype(ac_space) #probablity type. 
        sequence_length = None

        self.ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        #obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
        obz = self.ob #no-normalization

        last_out = obz
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(dense(last_out, hid_size, "polfc%i" % (i+1), weight_init=U.normc_initializer(1.0)))

        if isinstance(ac_space, gym.spaces.Box): #double check if the diag is changed during training
            mean = dense(last_out, pdtype.param_shape()[0]//2, "polfinal", U.normc_initializer(0.01))
            #logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
            logstd = tf.get_variable(name="logstd", initializer=tf.ones([1,pdtype.param_shape()[0]//2])*(-0.0)) #-1
            pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
        else:
            pdparam = dense(last_out, pdtype.param_shape()[0], "polfinal", U.normc_initializer(0.01))

        self.pd = pdtype.pdfromflat(pdparam) #a probability distribution, parameterized by pdparam

        self.state_in = []
        self.state_out = []

        stochastic = U.get_placeholder(name="stochastic", dtype=tf.bool, shape=()) 
        #stocahstic sample action or deterministically pick mode
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self.ac = ac
        self._act = U.function([stochastic, self.ob], ac) #return the action and the 

        #compute logp:
        if isinstance(ac_space, gym.spaces.Box):
            self.tmp_ac = tf.placeholder(tf.float32, shape = [sequence_length]+list(ac_space.shape))
        else:
            self.tmp_ac = tf.placeholder(tf.uint8, shape = [sequence_length]+list(ac_space.shape))
        self._tf_logp = self.pd.logp(self.tmp_ac)
        self._logp = U.function([self.tmp_ac, self.ob], self._tf_logp)


    def setupadam(self, l2_lambda):
        #set up objective function:
        ratios = tf.placeholder(tf.float32, shape = [None])
        probs = tf.exp(self._tf_logp)
        avg_value = tf.reduce_mean(tf.multiply(probs, ratios)) #avg
        #probs_dot_ratios = tf.tensordot(ratios, probs, 1)
        pi_var_list = [v for v in self.get_trainable_variables()]
        self.dot_p_r = U.function([self.ob, self.tmp_ac, ratios], avg_value)
        avg_value += l2_lambda*tf.tensordot(self.flatten_var, self.flatten_var,1) # a small l2 regularization
        self.compute_pi_grad = U.function([self.ob, self.tmp_ac, ratios],
                                          [avg_value, U.flatgrad(avg_value, pi_var_list)])
        #take ob, ac, and ratios as input, output value and grad0
        self.adam = MpiAdam(pi_var_list, comm = None)


    def act(self, stochastic, ob):
        assert ob.ndim == 1
        ac1 = self._act(stochastic, ob[None])
        return ac1[0]  #return a action

    def logps(self, obs, acs):
        assert obs.ndim == 2 #multiple (x,a) pairs
        logps = self._logp(acs, obs)
        return logps #return a list


    def logp(self, ob, ac):
        assert ob.ndim == 1
        logp = self._logp(ac[None], ob[None]);
        return logp[0] #return a scalar

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []

    #get flatten trainable variaables from the policy network
    def get_traniable_variables_flat(self):
        #return self.get_flat()
        return tf.get_default_session().run(self.flatten_var)

    #set a given flatten variables to the policy network
    def set_trainable_variable_flat(self, theta): #flatten variables theta:
        self.set_from_flat(theta)

    #take observations, actions, and ratios as inputs, do a adam gradient descent on policy parameters. 
    def update_policy(self, obs, acts, ratios, step_size = 1e-3):
        value, g = self.compute_pi_grad(obs, acts, ratios)
        self.adam.update(g, step_size)
        return value


class uniform_policy_discrete(object):

    def __init__(self, num_acts):
        self.num_acts = num_acts

    def act(self, stochastic, ob):
        act = np.random.randint(self.num_acts)
        return act


    def logp(self, ob, ac):
        return np.log(1./self.num_acts)

class uniform_policy_MultiDiscrete(object):

    def __init__(self, nvec):
        self.act_space = gym.spaces.MultiDiscrete(nvec)
        self.total_num = np.prod(nvec)

    def act(self, stochastic, ob):
        act = self.act_space.sample()
        act = act.astype('int')
        return act

    def logp(self, ob, ac):
        return np.log(1./self.total_num)




class uniform_policy_continuous(object):

    def __init__(self, dim, low, high):
        self.dim = dim
        self.low = low
        self.high = high

        self.volume = (np.abs(high-low))**self.dim

    def act(self, stochastic, ob):
        act = np.random.uniform(low = self.low/2., high = self.high/2., size = self.dim)
        return act

    def logp(self, ob, ac):
        return np.log(1./(self.volume*1.))

class mix_policy_continuous(object):

    def __init__(self, pi_1, pi_2, epsilon = 0.5):
        self.epsilon = epsilon
        self.pi_1 = pi_1
        self.pi_2 = pi_2

    def act(self, stochastic, ob):
        coin = np.random.rand()
        if coin < self.epsilon:
            return self.pi_1.act(stochastic, ob)
        else:
            return self.pi_2.act(stochastic, ob)

    def logp(self, ob, ac):
        p1 = np.exp(self.pi_1.logp(ob,ac))
        p2 = np.exp(self.pi_2.logp(ob,ac))
        return np.log(p1*self.epsilon + (1.-self.epsilon)*p2)









