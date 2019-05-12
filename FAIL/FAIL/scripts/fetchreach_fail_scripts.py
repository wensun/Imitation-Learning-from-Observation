import os
import numpy as np
import argparse

parser = argparse.ArgumentParser("Scripts for reproducing FAIL's results on FetchReachDiscrete")
parser.add_argument('--num_expert_trajs', type = int, help='number of expert trajs', default=25)
parser.add_argument('--num_timesteps', type=int, help='total number of training samples', default = 500000)
args = parser.parse_args()

seeds = [128038,470925,491264,791625, 100880, 203246,437783,491756,620105,875689]
seeds =[128038]

print("training FetchReachDiscrete FAIL with 10 random seeds...")

env_id = 'FetchReachDiscreteDense-v1'
expert_path = 'FetchReachDiscreteDense-v1expert_traj_1_64_deterministic.p'
policy_hidden_size = 100
num_hid_layers=1
adam_lr = 1e-3
horizon = 50
num_expert_trajs = args.num_expert_trajs#25
num_timesteps =  args.num_timesteps #250000 #500000
min_max_game_iteration = 100
mixing=0
warm_start=0

for i in range(len(seeds)):
    print("training at {0} th seed, with seed {1}".format(i, seeds[i]))
    seed = seeds[i]
    command = "python FAIL/run.py --env_id={0} --expert_path={1} --policy_hidden_size={2} --num_hid_layers={3} --adam_lr={4} --horizon={5} --num_expert_trajs={6} --num_timesteps={7} --min_max_game_iteration={8} --mixing={9} --warm_start={10} --seed={11}".format(
                        env_id, expert_path, policy_hidden_size, num_hid_layers, adam_lr, horizon, num_expert_trajs, num_timesteps, min_max_game_iteration, mixing, warm_start, seed)

    print(command)
    os.system(command)



