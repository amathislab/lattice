# Implementation of Lattice (Latent Exploration for Reinforcement Learning)

This repository includes the code to train the agents as in the paper `Latent Exploration for Reinforcement Learning`, submitted to NeurIPS 2023. The repository is structured as follows:
* src/
  * envs/
    * Modified MyoSuite environments, when we used different parameters from the default (cf. manuscript)
    * Factory to instantiate all the environemtns used in the project
  * metrics/
    * Stable Baselines 3 callbacks to register useful information during the training
  * models/
    * Implementation of Lattice
    * Adaptation of SAC and PPO to use Lattice
  * train/
    * Trainer class used to manage the trainings
  * main_*
    * One main file per environment, to start a training
* data/
  * configuration files for the MyoSuite environments
* docker-cuda
  * Definition of the Dockerfile to create the image used to run the experiments, with GPU support


## Requirements

Listed in `docker/requirements.txt`. Note that there is a version error with some packages, e.g. `stable_baselines3`, requiring later versions of `gym` which `myosuite` is incompatible with. For this reason, we recommend following the instructions in outlined in `docker/Dockerfile`, where additional passages after the installation of the requirements are outlined. We recommend creating a Docker image from the docker file, as it is the procedure that ensures the best compatibility.

## Usage

Run `python src/main_*.py` to start a training, replacing `*` with the name of one fo the environments. The output of the training, including the configuration used to select the hyperparameters and the tensorboard logs, are saved in a subfolder of `output/`, named as the current date. The different configuration hyperparameters can be set from the command line, e.g., by running `python src/main_humanoid.py --use_sde --use_lattice --freq=8`. In this case, a policy will be trained with SAC in the Humanoid environment, using lattice with update period 8. 
N.B. In order to use Lattice both the flags --use_sde and --use_lattice are needed, as Lattice is implemented as an extension of gSDE.
