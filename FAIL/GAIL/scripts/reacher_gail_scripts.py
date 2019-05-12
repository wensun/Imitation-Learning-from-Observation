import os
import numpy as np
import argparse

parser = argparse.ArgumentParser("Scripts for reproducing FAIL's results on FetchReachDiscrete")
parser.add_argument('--num_expert_trajs', type = int, help='number of expert trajs', default=25)
parser.add_argument('--num_timesteps', type=int, help='total number of training samples', default=1000000)
args = parser.parse_args()


seeds = [128038,470925,491264,791625, 100880, 203246,437783,491756,620105,875689]

print("training Reacher GAIL with 10 random seeds...")

env_id = 'ReacherDiscrete-v2'
expert_path = 'ReacherDiscrete-v2expert_traj_1_64_deterministic.p'
policy_hidden_size = 100
num_hid_layers=1
horizon = 50
num_expert_trajs = args.num_expert_trajs
num_timesteps = args.num_timesteps  #500000


for i in range(len(seeds)):
    print("training at {0} th seed, with seed {1}".format(i, seeds[i]))
    seed = seeds[i]
    command = "python GAIL/run_mujoco.py --env_id={0} --expert_path={1} --policy_hidden_size={2} --num_layers={3} --num_expert_trajs={4} --num_timesteps={5} --seed={6} --horizon={7}".format(
                        env_id, expert_path, policy_hidden_size, num_hid_layers,  num_expert_trajs, num_timesteps,  seed, horizon)

    print(command)
    os.system(command)




