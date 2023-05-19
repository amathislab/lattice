import os
import shutil
import argparse
import torch.nn as nn
from datetime import datetime
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from datetime import datetime
from definitions import ROOT_DIR
from envs.environment_factory import EnvironmentFactory
from metrics.custom_callbacks import EnvDumpCallback, TensorboardCallback
from train.trainer import MyoTrainer
from models.ppo.policies import LatticeRecurrentActorCriticPolicy


parser = argparse.ArgumentParser(description='Main script to train an agent')

parser.add_argument('--seed', type=int, default=0,
                    help='Seed for random number generator')
parser.add_argument('--freq', type=int, default=1,
                    help='SDE sample frequency')
parser.add_argument('--use_sde', action='store_true', default=False,
                    help='Flag to use SDE')
parser.add_argument('--use_lattice', action='store_true', default=False,
                    help='Flag to use lattice')
parser.add_argument('--log_std_init', type=float, default=0.0,
                    help='Initial log standard deviation')
parser.add_argument('--env_path', type=str,
                    help='Path to environment file')
parser.add_argument('--model_path', type=str,
                    help='Path to model file')
parser.add_argument('--num_envs', type=int, default=16,
                    help='Number of parallel environments')
parser.add_argument('--device', type=str, default="cuda",
                    help='Device, cuda or cpu')
parser.add_argument('--std_reg', type=float, default=0.0,
                    help='Additional independent std for the multivariate gaussian (only for lattice)')

args = parser.parse_args()

# define constants
ENV_NAME = "MyoFingerReachRandom"

now = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")

if args.model_path is not None:
    model_name = args.model_path.split("/")[-2]
else:
    model_name = None
    
TENSORBOARD_LOG = (
    os.path.join(ROOT_DIR, "output", "training", now)
    + f"_finger_reach_sde_{args.use_sde}_lattice_{args.use_lattice}_freq_{args.freq}_log_std_init_{args.log_std_init}_std_reg_{args.std_reg}_recurrent_ppo_seed_{args.seed}_resume_{model_name}"
)

# Reward structure and task parameters:
config = {
    "seed": args.seed,
}

max_episode_steps = 100  # default: 100
num_envs = 16  # 16 for training, fewer for debugging

model_config = dict(
    policy=LatticeRecurrentActorCriticPolicy,
    device=args.device,
    batch_size=32,
    n_steps=128,
    learning_rate=2.55673e-05,
    ent_coef=3.62109e-06,
    clip_range=0.3,
    gamma=0.99,
    gae_lambda=0.9,
    max_grad_norm=0.7,
    vf_coef=0.835671,
    n_epochs=10,
    use_sde=args.use_sde,
    sde_sample_freq=args.freq,  # number of steps
    policy_kwargs=dict(
        use_lattice=args.use_lattice,
        use_expln=True,
        ortho_init=False,
        log_std_init=args.log_std_init,  # TODO: tune
        activation_fn=nn.ReLU,
        net_arch=[dict(pi=[256, 256], vf=[256, 256])],
        std_clip=(1e-3, 10),
        expln_eps=1e-6,
        full_std=False,
        std_reg=args.std_reg
    ),
)

# Function that creates and monitors vectorized environments:
def make_parallel_envs(env_config, num_env, start_index=0):
    def make_env(_):
        def _thunk():
            env = EnvironmentFactory.create(ENV_NAME, **env_config)
            env.seed(args.seed)
            env._max_episode_steps = max_episode_steps
            env = Monitor(env, TENSORBOARD_LOG)
            return env

        return _thunk

    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])


if __name__ == "__main__":
    # ensure tensorboard log directory exists and copy this file to track
    os.makedirs(TENSORBOARD_LOG, exist_ok=True)
    shutil.copy(os.path.abspath(__file__), TENSORBOARD_LOG)

    # Create and wrap the training and evaluations environments
    envs = make_parallel_envs(config, num_envs)

    if args.env_path is not None:
        envs = VecNormalize.load(args.env_path, envs)
    else:
        envs = VecNormalize(envs)

    # Define callbacks for evaluation and saving the agent
    eval_callback = EvalCallback(
        eval_env=envs,
        callback_on_new_best=EnvDumpCallback(TENSORBOARD_LOG, verbose=0),
        n_eval_episodes=10,
        best_model_save_path=TENSORBOARD_LOG,
        log_path=TENSORBOARD_LOG,
        eval_freq=10_000,
        deterministic=True,
        render=False,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,
        save_path=TENSORBOARD_LOG,
        save_vecnormalize=True,
        verbose=1,
    )

    tensorboard_callback = TensorboardCallback(
        info_keywords=(
            "rwd_dense",
            "rwd_sparse",
            "solved",
            "done",
        )
    )

    # Define trainer
    trainer = MyoTrainer(
        algo="recurrent_ppo",
        envs=envs,
        env_config=config,
        load_model_path=args.model_path,
        log_dir=TENSORBOARD_LOG,
        model_config=model_config,
        callbacks=[eval_callback, checkpoint_callback, tensorboard_callback],
        timesteps=500_000_000,
    )

    # Train agent
    trainer.train(total_timesteps=trainer.timesteps)
    trainer.save()
