# Lattice (Latent Exploration for Reinforcement Learning)

This repository includes the implementation of Lattice exploration from the paper [Latent Exploration for Reinforcement Learning](https://arxiv.org/abs/2305.20065), published at NeurIPS 2023. Lattice introduces random perturbations in the latent state of the policy network, which result in correlated noise across the system's actuators. This form of latent noise can help exploration in high-dimensional systems, especially with redundant actuation, and often finds low-effort solutions.

Lattice was build on top of Stable Baselines 3 (version 1.6.1) and it is here implemented for Recurrent PPO and SAC. Integration with a more recent version of Stable Baselines 3 and compatibility with more algorithms is [currently under development](https://github.com/albertochiappa/stable-baselines3).

This project was developed by Alberto Silvio Chiappa, Alessandro Marin Vargas, Ann Zixiang Huang and Alexander Mathis.

## MyoChallenge 2023
We used Lattice to win the NeurIPS 2023 competition [MyoChallenge](https://sites.google.com/view/myosuite/myochallenge/myochallenge-2023?authuser=0), where our team won the object manipulation track. With curriculum learning, reward shaping and Lattice exploration we trained a policy to control a biologically-realistic arm with 63 muscles and 27 degrees of freedom to place random objects inside a box of variable shape:

![relocate](/data/images/myochallenge_2023.gif)

We outperformed the other best solutions both in score and effort:

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

## Start a training

We have prepared some main scripts for various environments from [MyoSuite](https://sites.google.com/view/myosuite) and [PyBullet](https://pybullet.org). Starting a training is as easy as

```bash
python main_pose_elbow.py --use_lattice
```

if you have created a conda environment, or

```bash
docker run --rm --gpus all -it \ 
--mount type=bind,src="$(pwd)/src",target=/src \
--mount type=bind,src="$(pwd)/data",target=/data \ 
--mount type=bind,src="$(pwd)/output",target=/output \ 
albertochiappa/myo-cuda-pybullet \
python3 src/main_pose_elbow.py --use_lattice
```

if you prefer to use the readily available docker container. The previous command will start a training in the `Elbow Pose` enviornment using Recurrent PPO. Simply change the main script name to start a training for a different environment. The output of the training, including the configuration used to select the hyperparameters and the tensorboard logs, are saved in a subfolder of `output/`, named as the current date. The code outputs useful information to monitor the training in Tensorboard format. You can run Tensorboard in the output folder to visualize the learning curves and much more. The different configuration hyperparameters can be set from the command line, e.g., by running 

```bash
python src/main_humanoid.py --use_sde --use_lattice --freq=8
```

In this case, a policy will be trained with SAC in the Humanoid environment, using state-dependent Lattice with update period 8.

## Structure of the repository

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

## Reference

If our work was useful to your research, please cite us!

```
@article{chiappa2023latent,
  title={Latent exploration for reinforcement learning},
  author={Chiappa, Alberto Silvio and Vargas, Alessandro Marin and Huang, Ann Zixiang and Mathis, Alexander},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2023}
}
```
