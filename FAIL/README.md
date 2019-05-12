## FAIL: Forward Adversarial Imitation Learning

Dowload and install gym from [repository](https://github.com/wensun/gym)

Install Baselines from [repository](https://github.com/openai/baselines) and add pathto/baselines to PYTHONPATH

## Training FAIL:
```
python FAIL/run.py -h
```


### Example for reproducing results SwimmerDiscrete-v2 with 12 expert trajectories and 250000 many training samples
```bash
python FAIL/scripts/swimmer_fail_scripts.py --num_timesteps=250000 --num_expert_trajs=12
```

### Example for Reproducing results on ReacherDiscrete-v2 with 12 expert trajectories and 250000 many training samples
```bash
python FAIL/scripts/reacher_fail_scripts.py --num_timesteps=250000 --num_expert_trajs=12
```

### Example for reproducing results on  HopperDiscrete-v2 with 12 expert trajectories and 250000 many training samples
```bash
python FAIL/scripts/hopper_fail_scripts.py --num_timesteps=250000 --num_expert_trajs=12
```
### Example for reproducing results on FetchReachDiscrete-v1 with 12 expert trajectories and 250000 many training samples
```bash
python FAIL/scripts/fetchreach_fail_scripts.py --num_timesteps=250000 --num_expert_trajs=12
```

## Training GAIL:

``` bash
python GAIL/run_mujoco.py -h
```


### Example for reproducing results on HopperDiscrete-v2 with 12 expert trajectories and 250000 many training samples
``` bash
python GAIL/scripts/hopper_gail_scripts.py --num_expert_trajs=12 --num_timesteps=250000
```

### Example for reproducing results on ReacherDiscrete-v2 with 12 expert trajectories and 250000 many training samples
```bash
python GAIL/scripts/reacher_gail_scripts.py --num_expert_trajs=12 --num_timesteps=250000
```

### Example for reproducing results on SwimmerDiscrete-v2 with 12 expert trajectories and 250000 many training samples
```bash
python GAIL/scripts/swimmer_gail_scripts.py --num_expert_trajs=12 --num_timesteps=250000
```

### Example for reproducing results on FetchReachDiscrete-v1 with 12 expert trajectories and 250000 many training samples
```bash
python GAIL/scripts/fetchreach_gail_scripts.py --num_expert_trajs=12 --num_timesteps=250000
```
