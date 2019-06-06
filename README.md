# Forward Adversarial Imitation Learning

This repository implements the algorithms presented in the paper (published in ICML 2019)

# Dependencies

* We advise the reader to use virtualenv so that installing dependencies is easy
* Install the gym package given with the code. To do this, activate your virtual env and go inside `gym/` and run `pip install -e .`
* Install tensorflow (version 1.12.0)
* Install the OpenAI baselines package given with the code. To do this, activate your virtual env and go inside `baselines/` and run `pip install -e .`
* Install other dependencies. To do this, activate your virtual env and go inside `FAIL/` and run `pip install -r requirements.txt`
* Install Mujoco-py: `pip install -U 'mujoco-py<1.50.2,>=1.50.1'`
* We expect the reader to have a mujoco license to run the mujoco experiments in the paper

# Running the code

To download datasets, go to the directory `FAIL/`
  * run `python download_datasets.py`

To run the experiments, go to the directory `FAIL/` and for the environment
  * Fetchreach, run `./scripts/fetchreachexperiments.sh`
  * Swimmer, run `./scripts/swimmerdiscreteexperiments.sh`
  * Reacher, run `./scripts/reacherdiscreteexperiments.sh`
  
The results are generated and stored in the `FAIL/data` folder (for all the 10 random seeds in each experiment for every method)
