#!/bin/bash

# Number of expert trajectories experiment
numexperttrajs=(6 12 25)

for net in ${numexperttrajs[@]}
do
    python FAIL/scripts/reacher_fail_scripts.py --num_timesteps=1000000 --num_expert_trajs=$net &
    python GAIL/scripts/reacher_gail_scripts.py --num_timesteps=1000000 --num_expert_trajs=$net &
done
wait

# Number of samples experiment
numtimesteps=(250000 500000 1000000)

for nt in ${numtimesteps[@]}
do
    python FAIL/scripts/reacher_fail_scripts.py --num_timesteps=$nt --num_expert_trajs=25 &
    python GAIL/scripts/reacher_gail_scripts.py --num_timesteps=$nt --num_expert_trajs=25 &
done
wait

