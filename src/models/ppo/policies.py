from stable_baselines3.common.preprocessing import get_action_dim
from models.distributions import (
    LatticeNoiseDistribution,
    LatticeStateDependentNoiseDistribution,
)
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy


class LatticeRecurrentActorCriticPolicy(RecurrentActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        use_lattice=True,
        std_clip=(1e-3, 10),
        expln_eps=1e-6,
        std_reg=0,
        alpha=1,
        **kwargs
    ):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        if use_lattice:
            if self.use_sde:
                self.dist_kwargs.update(
                    {
                        "epsilon": expln_eps,
                        "std_clip": std_clip,
                        "std_reg": std_reg,
                        "alpha": alpha,
                    }
                )
                self.action_dist = LatticeStateDependentNoiseDistribution(
                    get_action_dim(self.action_space), **self.dist_kwargs
                )
                self._build(lr_schedule)
            else:
                self.action_dist = LatticeNoiseDistribution(
                    get_action_dim(self.action_space)
                )
