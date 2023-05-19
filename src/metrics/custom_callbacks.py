import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class EnvDumpCallback(BaseCallback):
    def __init__(self, save_path, verbose=0):
        super().__init__(verbose=verbose)
        self.save_path = save_path

    def _on_step(self):
        env_path = os.path.join(self.save_path, "training_env.pkl")
        if self.verbose > 0:
            print("Saving the training environment to path ", env_path)
        self.training_env.save(env_path)
        return True
    
    
class TensorboardCallback(BaseCallback):
    def __init__(self, info_keywords, verbose=0):
        super().__init__(verbose=verbose)
        self.info_keywords = info_keywords
        self.rollout_info = {}
        
    def _on_rollout_start(self):
        self.rollout_info = {key: [] for key in self.info_keywords}
        
    def _on_step(self):
        for key in self.info_keywords:
            vals = [info[key] for info in self.locals["infos"]]
            self.rollout_info[key].extend(vals)
        return True
    
    def _on_rollout_end(self):
        for key in self.info_keywords:
            self.logger.record("rollout/" + key, np.mean(self.rollout_info[key]))