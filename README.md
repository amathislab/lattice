# Lattice (Latent Exploration for Reinforcement Learning)

This repository includes the implementation of Lattice exploration from the paper [Latent Exploration for Reinforcement Learning](https://arxiv.org/abs/2305.20065), published at NeurIPS 2023.

This project was developed by Alberto Silvio Chiappa, Alessandro Marin Vargas, Ann Zixiang Huang and Alexander Mathis.

## MyoChallenge 2023
We used Lattice to win the NeurIPS 2023 competition [MyoChallenge](https://sites.google.com/view/myosuite/myochallenge/myochallenge-2023?authuser=0), where our team won the object manipulation track. With curriculum learning, reward shaping and Lattice exploration we trained a policy to control a biologically-realistic arm with 63 muscles and 27 degrees of freedom to place random objects inside a box of variable shapes:

![relocate](/data/images/myochallenge_2023.gif)

We outperformed the other best solutions both in performance and effort:

<img src="/data/images/myochallenge_ranking.png" alt="drawing" width="70%"/>

We have also created [a dedicated repository](https://github.com/amathislab/myochallenge_2023eval) for the solution, where we have released the pretrained weights of all the curriculum steps.

## Installation

We recommend using a Docker container to execute the code of this repository. We provide both the docker image [albertochiappa/myo-cuda-pybullet](https://hub.docker.com/repository/docker/albertochiappa/myo-cuda-pybullet) in DockerHub and the Dockerfile in the [docker](/docker/) folder to create the same docker image locally.

If you prefer to manually create a Conda environment, you can do so with the commands:

```bash
conda create --name lattice python=3.8.10
conda activate lattice
pip install -r docker/requirements.txt
pip install myosuite==1.2.4 
pip install --upgrade cloudpickle==2.2.0 pickle5==0.0.11 pybullet==3.2.5
```

Please note that there is a version error with some packages, e.g. `stable_baselines3`, requiring a later version of `gym` which `myosuite` is incompatible with. For this reason we could not include all the requirements in `docker/requirements.txt`. In our experiments the stated incompatibility did not cause any error.

## Usage

Run `python src/main_*.py` to start a training, replacing `*` with the name of one fo the environments. The output of the training, including the configuration used to select the hyperparameters and the tensorboard logs, are saved in a subfolder of `output/`, named as the current date. The different configuration hyperparameters can be set from the command line, e.g., by running `python src/main_humanoid.py --use_sde --use_lattice --freq=8`. In this case, a policy will be trained with SAC in the Humanoid environment, using lattice with update period 8. 
N.B. In order to use Lattice both the flags --use_sde and --use_lattice are needed, as Lattice is implemented as an extension of gSDE.

 The repository is structured as follows:
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


