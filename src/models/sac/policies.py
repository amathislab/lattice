from typing import Any, Dict, List, Optional, Type
import torch
from torch import nn
from models.distributions import (
    LatticeStateDependentNoiseDistribution,
    SquashedLatticeNoiseDistribution,
)
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac.policies import SACPolicy, Actor


class LatticeActor(Actor):
    def __init__(
        self,
        observation_space,
        action_space,
        net_arch,
        features_extractor,
        features_dim,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
        use_lattice=False,
        std_clip=(1e-3, 10),
        expln_eps=1e-6,
        std_reg=0,
        alpha=1,
    ):
        super().__init__(
            observation_space,
            action_space,
            net_arch,
            features_extractor,
            features_dim,
            activation_fn=activation_fn,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            sde_net_arch=sde_net_arch,
            use_expln=use_expln,
            clip_mean=clip_mean,
            normalize_images=normalize_images,
        )
        self.use_lattice = use_lattice
        self.std_clip = std_clip
        self.expln_eps = expln_eps
        self.std_reg = std_reg
        self.alpha = alpha
        if use_lattice:
            if self.use_sde:
                last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim
                action_dim = get_action_dim(self.action_space)
                self.action_dist = LatticeStateDependentNoiseDistribution(
                    action_dim,
                    full_std=full_std,
                    use_expln=use_expln,
                    squash_output=True,
                    learn_features=True,
                    epsilon=expln_eps,
                    std_clip=std_clip,
                    std_reg=std_reg,
                    alpha=alpha,
                )
                self.mu, self.log_std = self.action_dist.proba_distribution_net(
                    latent_dim=last_layer_dim,
                    latent_sde_dim=last_layer_dim,
                    log_std_init=log_std_init,
                )
                if clip_mean > 0.0:
                    self.mu = nn.Sequential(
                        self.mu, nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean)
                    )
            else:
                self.action_dist = SquashedLatticeNoiseDistribution(action_dim)
                self.mu = nn.Linear(last_layer_dim, action_dim)
                self.log_std = nn.Linear(last_layer_dim, action_dim)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                use_lattice=self.use_lattice,
                std_clip=self.std_clip,
                expln_eps=self.expln_eps,
                std_reg=self.std_reg,
                alpha=self.alpha,
            )
        )
        return data

    def get_std(self) -> torch.Tensor:
        std = super().get_std()
        if self.use_lattice:
            std = torch.cat(std, dim=1)
        return std


class LatticeSACPolicy(SACPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        use_lattice=False,
        std_clip=(1e-3, 10),
        expln_eps=1e-6,
        std_reg=0,
        use_sde=False,
        alpha=1,
        **kwargs
    ):
        super().__init__(
            observation_space, action_space, lr_schedule, use_sde=use_sde, **kwargs
        )
        self.lattice_kwargs = {
            "use_lattice": use_lattice,
            "expln_eps": expln_eps,
            "std_clip": std_clip,
            "std_reg": std_reg,
            "alpha": alpha,
        }
        self.actor_kwargs.update(self.lattice_kwargs)
        if use_lattice:
            assert use_sde
            self._build(lr_schedule)

    def make_actor(
        self, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> Actor:
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, features_extractor
        )
        return LatticeActor(**actor_kwargs).to(self.device)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(self.lattice_kwargs)
        return data
