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
from IPython import embed

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


    def _init(self, ob_space, K,l2_lambda, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True):
        assert isinstance(ob_space, gym.spaces.Box)
        self.l2_reg = l2_lambda

        self.a_dim = ac_space.shape[0]
        self.highs = []
        self.deltas = []
        self.lows = []
        self.actions = np.zeros((self.a_dim, K))
        self.pdtypes = []
        for a_i in range(self.a_dim):
            high = ac_space.high[a_i]
            low = ac_space.low[a_i]
            delta = (high - low)/(K-1.)
            self.deltas.append(delta)
            self.highs.append(high)
            self.lows.append(low)
            self.actions[a_i] = np.array([low + k*delta for k in range(K)])
            tmp_ac_space = gym.spaces.Discrete(K)
            pdtype = make_pdtype(tmp_ac_space)
            self.pdtypes.append(pdtype)

        self.ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[None] + list(ob_space.shape))
        obz = self.ob #no-normalization
        last_out = obz
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(dense(last_out,hid_size,"polfc%i" % (i+1), weight_init=U.normc_initializer(1.0)))

        self.pds = []
        for a_i in range(self.a_dim):
            pdtype = self.pdtypes[a_i]
            pdparam = dense(last_out,pdtype.param_shape()[0],"polfinal_{0}".format(a_i), U.normc_initializer(0.01))
            pd = pdtype.pdfromflat(pdparam)
            self.pds.append(pd)

        self.state_in = []
        self.state_out = []

        stochastic = U.get_placeholder(name="stochastic", dtype=tf.bool, shape=()) 
        #stocahstic sample action or deterministically pick mode
        self.acs = []
        self._acts = []
        #self.tmp_ac = tf.placeholder(tf.uint8, shape = [None])
        self.tmp_ac = tf.placeholder(tf.uint8, shape = [None, ac_space.shape[0]])
        self._tf_logp = None
        logits = []
        for a_i in range(self.a_dim):
            ac = U.switch(stochastic, self.pds[a_i].sample(), self.pds[a_i].mode())
            logits.append(self.pds[a_i].logits)
            self.acs.append(ac)
            _act = U.function([stochastic, self.ob], ac)
            self._acts.append(_act)

            _tf_logp_i = self.pds[a_i].logp(self.tmp_ac[:,a_i])
            if a_i == 0:
                self._tf_logp = _tf_logp_i
            else:
                self._tf_logp += _tf_logp_i

        self._logp = U.function([self.tmp_ac, self.ob], self._tf_logp)
        self._logits = U.function([self.ob], logits)

    def setupadam(self, l2_lambda):
        #set up objective function:
        ratios = tf.placeholder(tf.float32, shape = [None])
        probs = tf.exp(self._tf_logp)
        avg_value = tf.reduce_mean(tf.multiply(probs, ratios)) #avg
        pi_var_list = [v for v in self.get_trainable_variables()]
        self.dot_p_r = U.function([self.ob, self.tmp_ac, ratios], avg_value)
        avg_value += l2_lambda*tf.tensordot(self.flatten_var, self.flatten_var,1) # a small l2 regularization
        self.compute_pi_grad = U.function([self.ob, self.tmp_ac, ratios],
                                          [avg_value, U.flatgrad(avg_value, pi_var_list)])
        #take ob, ac, and ratios as input, output value and grad0
        self.adam = MpiAdam(pi_var_list, comm = None)


    def act(self, stochastic, ob):
        assert ob.ndim == 1
        act_index = np.zeros(self.a_dim).astype(int)
        for a_i in range(self.a_dim):
            ind_a_i = self._acts[a_i](stochastic, ob[None])[0]
            act_index[a_i] = ind_a_i
        return act_index

    def logps(self, obs, acs_index):
        assert obs.ndim == 2 #multiple (x,a) pairs
        assert acs_index.ndim == 2 and acs_index.dtype==int
        logps = self._logp(acs_index, obs)
        return logps #return a list


    def logp(self, ob, ac):
        assert ob.ndim == 1 and ac.ndim == 1 and ac.dtype == int
        logp = self._logp(ac[None], ob[None])

        return logp[0] #return a scalar

    def transfor_to_real(self, act_index):
        assert act_index.ndim == 1 and act_index.dtype == int
        act = np.zeros(self.a_dim)
        for i in range(self.a_dim):
            act[i] = self.actions[i,act_index[i]]
        return act

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





