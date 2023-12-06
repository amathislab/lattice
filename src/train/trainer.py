import json
import os
from abc import ABC
from dataclasses import dataclass, field
from typing import List
from stable_baselines3 import PPO, SAC, TD3
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize


class Trainer(ABC):
    """
    Protocol to train a library-independent RL algorithm on a gym environment.
    """

    envs: VecNormalize
    env_config: dict
    model_config: dict
    model_path: str
    total_timesteps: int
    log: bool

    def _init_agent(self) -> None:
        """Initialize the agent."""

    def train(self, total_timesteps: int) -> None:
        """Train agent on environment for total_timesteps episdodes."""


@dataclass
class MyoTrainer:
    algo: str
    envs: VecNormalize
    env_config: dict
    load_model_path: str
    log_dir: str
    model_config: dict = field(default_factory=dict)
    callbacks: List[BaseCallback] = field(default_factory=list)
    timesteps: int = 10_000_000

    def __post_init__(self):
        self.dump_configs(path=self.log_dir)
        self.agent = self._init_agent()

    def dump_configs(self, path: str) -> None:
        with open(os.path.join(path, "env_config.json"), "w", encoding="utf8") as f:
            json.dump(self.env_config, f, indent=4, default=lambda _: '<not serializable>')
        with open(os.path.join(path, "model_config.json"), "w", encoding="utf8") as f:
            json.dump(self.model_config, f, indent=4, default=lambda _: '<not serializable>')

    def _init_agent(self):
        algo_class = self.get_algo_class()
        if self.load_model_path is not None:
            return algo_class.load(
                self.load_model_path,
                env=self.envs,
                tensorboard_log=self.log_dir,
                custom_objects=self.model_config,
            )
        print("\nNo model path provided. Initializing new model.\n")
        return algo_class(
            env=self.envs,
            verbose=2,
            tensorboard_log=self.log_dir,
            **self.model_config,
        )

    def train(self, total_timesteps: int) -> None:
        self.agent.learn(
            total_timesteps=total_timesteps,
            callback=self.callbacks,
            reset_num_timesteps=True,
        )

    def save(self) -> None:
        self.agent.save(os.path.join(self.log_dir, "final_model.pkl"))
        self.envs.save(os.path.join(self.log_dir, "final_env.pkl"))

    def get_algo_class(self):
        if self.algo == "ppo":
            return PPO
        elif self.algo == "recurrent_ppo":
            return RecurrentPPO
        elif self.algo == "sac":
            return SAC
        elif self.algo == "td3":
            return TD3
        else:
            raise ValueError("Unknown algorithm ", self.algo)


if __name__ == "__main__":
    print("This is a module. Run main.py to train the agent.")
