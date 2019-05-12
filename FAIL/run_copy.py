import sys
import pickle
import multiprocessing
import os.path as osp
import gym
from collections import defaultdict
import tensorflow as tf
import numpy as np
from IPython import  embed
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env
from baselines.common.tf_util import get_session
from baselines import bench, logger
from importlib import import_module

from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common import atari_wrappers, retro_wrappers

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import roboschool
except ImportError:
    roboschool = None

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_type = env._entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}


def train(args, extra_args):
    env_type, env_id = get_env_type(args.env)
    print('env_type: {}'.format(env_type))

    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)

    env = build_env(args)

    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)

    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))

    model = learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        **alg_kwargs
    )
    
    return model, env


def build_env(args):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu
    alg = args.alg
    rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    seed = args.seed

    env_type, env_id = get_env_type(args.env)

    if env_type == 'atari':
        if alg == 'acer':
            env = make_vec_env(env_id, env_type, nenv, seed)
        elif alg == 'deepq':
            env = atari_wrappers.make_atari(env_id)
            env.seed(seed)
            env = bench.Monitor(env, logger.get_dir())
            env = atari_wrappers.wrap_deepmind(env, frame_stack=True)
        elif alg == 'trpo_mpi':
            env = atari_wrappers.make_atari(env_id)
            env.seed(seed)
            env = bench.Monitor(env, logger.get_dir() and osp.join(logger.get_dir(), str(rank)))
            env = atari_wrappers.wrap_deepmind(env)
            # TODO check if the second seeding is necessary, and eventually remove
            env.seed(seed)
        else:
            frame_stack_size = 4
            env = VecFrameStack(make_vec_env(env_id, env_type, nenv, seed), frame_stack_size)

    elif env_type == 'retro':
        import retro
        gamestate = args.gamestate or retro.State.DEFAULT
        env = retro_wrappers.make_retro(game=args.env, state=gamestate, max_episode_steps=10000,
                                        use_restricted_actions=retro.Actions.DISCRETE)
        env.seed(args.seed)
        env = bench.Monitor(env, logger.get_dir())
        env = retro_wrappers.wrap_deepmind_retro(env)

    else:
       get_session(tf.ConfigProto(allow_soft_placement=True,
                                   intra_op_parallelism_threads=1,
                                   inter_op_parallelism_threads=1))

       env = make_vec_env(env_id, env_type, args.num_env or 1, seed, reward_scale=args.reward_scale)

       if env_type == 'mujoco' or env_type == 'robotics':
           env = VecNormalize(env)

    return env


def get_env_type(env_id):
    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type, env_id


def get_default_network(env_type):
    if env_type == 'atari':
        return 'cnn'
    else:
        return 'mlp'

def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs



def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''
    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k,v in parse_unknown_args(args).items()}


def record_traj(args, env, env_type, model, num_expert_traj, max_time_step):
    trajs = []
    traj_rews = []
    success_times = 0
    for repeat in range(num_expert_traj):
        succeed = False
        traj = []
        traj_total_rew = 0.
        obs,unnorm_obs = env.reset(return_old_obs=True)

        while True:
            actions,_,_,_ = model.step_deterministic(obs)
            #actions,_,_,_ = model.step(obs)
            next_obs, r, done, info, unnorm_next_obs=env.step(actions,return_old_obs=True)
            if env_type == "robotics":
                if info[0]['is_success'] == True and succeed==False:
                    succeed = True
                    success_times+=1
            traj_total_rew += r
            obs_act_rew = (np.copy(unnorm_obs), np.copy(actions), np.copy(r), np.copy(unnorm_next_obs))
            traj.append(obs_act_rew)
            obs = next_obs
            unnorm_obs = unnorm_next_obs

            if len(traj) >= max_time_step:
                trajs.append(traj)
                traj_rews.append(traj_total_rew)
                break
    print("success_rate is {0}, avg_rew {1}".format(success_times*1./num_expert_traj, np.mean(traj_rews)))
    pickle.dump(trajs, open(args.env+"expert_traj.p", 'wb'))



def main():
    # configure logger, disable logging in child MPI processes (with rank > 0)

    #args.num_expert_traj: number of recorded expert traj during replay


    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args()
    extra_args = parse_cmdline_kwargs(unknown_args)


    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        logger.configure()
    else:
        logger.configure(format_strs=[])
        rank = MPI.COMM_WORLD.Get_rank()


    env_type, env_id = get_env_type(args.env)


    max_time_step = 100
    model, env = train(args, extra_args)
    record_traj(args, env, env_type, model, 1000, env.venv.envs[0].spec.max_episode_steps)
    env.close()


    if args.save_path is not None and rank == 0:
        save_path = osp.expanduser(args.save_path)
        model.save(save_path)

    num_expert_traj = 1000
    #max_time_step = 100 #100 #for acrobot, mountaincar and cartpole; 50: reacher
    #max_time_step = 100 #for acrobot, cheetah
    #max_time_step = 50 #reacher 50




    if args.play:
        logger.log("Running trained model")
        trajs = []; #a list of trajectories, where each trajectory itself is a list, containing a sequence of tuple
        env = build_env(args)

        sucess_times = 0
        for repeat in range(num_expert_traj):

            succeed = False

            print("AT ITER {0}".format(repeat))
            traj = []
            traj_total_rew = 0

            if env_type == "classic_control" or env_type == "mujoco" or env_type == "robotics":
                obs, unnorm_obs = env.reset(return_old_obs = True)
            else:
                obs = env.reset()

            def initialize_placeholders(nlstm=128,**kwargs):
                return np.zeros((args.num_env or 1, 2*nlstm)), np.zeros((1))
            state, dones = initialize_placeholders(**extra_args)
            while True:
                actions, _, state, _ = model.step_deterministic(obs,S=state, M=dones)
                #actions, _, state, _ = model.step(obs,S=state, M=dones)

                if env_type == "classic_control" or env_type == "mujoco" or env_type == 'robotics':
                    next_obs, r, done, info, unnorm_next_obs = env.step(actions, return_old_obs = True)
                    #if env_type == "mujoco": #delete reward that relates to control input
                        #embed()
                        #r -= 0# info[0]['reward_ctrl']
                    #    pass
                else:
                    next_obs, r, done, info = env.step(actions)


                if env_type == "robotics":
                    if info[0]['is_success'] == True and succeed==False:
                        succeed= True
                        sucess_times+=1

                traj_total_rew += r
                if env_type == "classic_control" or env_type == "mujoco" or env_type == 'robotics':
                    obs_act_rew = (np.copy(unnorm_obs), np.copy(actions), np.copy(r), np.copy(unnorm_next_obs)) #state-action-reward-next_state tuple
                else:
                    obs_act_rew = (np.copy(obs), np.copy(actions), np.copy(r), np.copy(next_obs))


                traj.append(obs_act_rew)
                obs = next_obs
                if env_type == "classic_control" or env_type == "mujoco" or env_type == 'robotics':
                    unnorm_obs = unnorm_next_obs

                #env.render()
                done = done.any() if isinstance(done, np.ndarray) else done
                if done:
                    print(len(traj))
                #if done:
                if len(traj) >= max_time_step:
                    trajs.append(traj)
                    print(len(traj), traj_total_rew, max_time_step)
                    break;


        env.close()
        print('success rate is {0}'.format(sucess_times*1./num_expert_traj))
        pickle.dump(trajs, open(args.env+"expert_traj.p", "wb"))

if __name__ == '__main__':
    main()
