import torch
import numpy as np
from typing import Optional
from torch import nn
from torch.distributions import MultivariateNormal
from typing import Tuple
from torch.distributions import Normal
from stable_baselines3.common.distributions import DiagGaussianDistribution, TanhBijector, StateDependentNoiseDistribution


class LatticeNoiseDistribution(DiagGaussianDistribution):
    """
    Like Lattice noise distribution, non-state-dependent. Does not allow time correlation, but
    it is more efficient.

    :param action_dim: Dimension of the action space.
    """

    def __init__(self, action_dim: int):
        super().__init__(action_dim=action_dim)

    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0, state_dependent: bool = False) -> Tuple[nn.Module, nn.Parameter]:
        self.mean_actions = nn.Linear(latent_dim, self.action_dim)
        self.std_init = torch.tensor(log_std_init).exp()
        if state_dependent:
            log_std = nn.Linear(latent_dim, self.action_dim + latent_dim)
        else:
            log_std = nn.Parameter(torch.zeros(self.action_dim + latent_dim), requires_grad=True)
        return self.mean_actions, log_std

    def proba_distribution(self, mean_actions: torch.Tensor, log_std: torch.Tensor) -> "LatticeNoiseDistribution":
        """
        Create the distribution given its parameters (mean, std)

        :param mean_actions:
        :param log_std:
        :return:
        """
        std = log_std.exp() * self.std_init
        action_variance = std[..., : self.action_dim] ** 2
        latent_variance = std[..., self.action_dim :] ** 2

        sigma_mat = (self.mean_actions.weight * latent_variance[..., None, :]).matmul(self.mean_actions.weight.T)
        sigma_mat[..., range(self.action_dim), range(self.action_dim)] += action_variance
        self.distribution = MultivariateNormal(mean_actions, sigma_mat)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions)

    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy()


class SquashedLatticeNoiseDistribution(LatticeNoiseDistribution):
    """
    Lattice noise distribution, followed by a squashing function (tanh) to ensure bounds.

    :param action_dim: Dimension of the action space.
    :param epsilon: small value to avoid NaN due to numerical imprecision.
    """
    def __init__(self, action_dim: int, epsilon: float = 1e-6):
        super().__init__(action_dim)
        self.epsilon = epsilon
        self.gaussian_actions: Optional[torch.Tensor] = None
        
    def log_prob(self, actions: torch.Tensor, gaussian_actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Inverse tanh
        # Naive implementation (not stable): 0.5 * torch.log((1 + x) / (1 - x))
        # We use numpy to avoid numerical instability
        if gaussian_actions is None:
            # It will be clipped to avoid NaN when inversing tanh
            gaussian_actions = TanhBijector.inverse(actions)

        # Log likelihood for a Gaussian distribution
        log_prob = super().log_prob(gaussian_actions)
        # Squash correction (from original SAC implementation)
        # this comes from the fact that tanh is bijective and differentiable
        log_prob -= torch.sum(torch.log(1 - actions**2 + self.epsilon), dim=1)
        return log_prob
    
    def entropy(self) -> Optional[torch.Tensor]:
        # No analytical form,
        # entropy needs to be estimated using -log_prob.mean()
        return None

    def sample(self) -> torch.Tensor:
        # Reparametrization trick to pass gradients
        self.gaussian_actions = super().sample()
        return torch.tanh(self.gaussian_actions)

    def mode(self) -> torch.Tensor:
        self.gaussian_actions = super().mode()
        # Squash the output
        return torch.tanh(self.gaussian_actions)

    def log_prob_from_params(self, mean_actions: torch.Tensor, log_std: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        action = self.actions_from_params(mean_actions, log_std)
        log_prob = self.log_prob(action, self.gaussian_actions)
        return action, log_prob


class LatticeStateDependentNoiseDistribution(StateDependentNoiseDistribution):
    """
    Distribution class of Lattice exploration.
    Paper: Latent Exploration for Reinforcement Learning https://arxiv.org/abs/2305.20065

    It creates correlated noise across actuators, with a covariance matrix induced by
    the network weights. Can improve exploration in high-dimensional systems.

    :param action_dim: Dimension of the action space.
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,), defaults to True
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
        Defaults to False
    :param squash_output:  Whether to squash the output using a tanh function,
        this ensures bounds are satisfied, defaults to False
    :param learn_features: Whether to learn features for gSDE or not, defaults to False
        This will enable gradients to be backpropagated through the features, defaults to False
    :param epsilon: small value to avoid NaN due to numerical imprecision, defaults to 1e-6
    :param std_clip: clip range for the standard deviation, can be used to prevent extreme values,
        defaults to (1e-3, 1.0)
    :param std_reg: optional regularization to prevent collapsing to a deterministic policy,
        defaults to 0.0
    :param alpha: relative weight between action and latent noise, 0 removes the latent noise,
        defaults to 1 (equal weight)
    """
    def __init__(
        self,
        action_dim: int,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        learn_features: bool = False,
        epsilon: float = 1e-6,
        std_clip: Tuple[float, float] = (1e-3, 1.0),
        std_reg: float = 0.0,
        alpha: float = 1,
    ):
        super().__init__(
            action_dim=action_dim,
            full_std=full_std,
            use_expln=use_expln,
            squash_output=squash_output,
            epsilon=epsilon,
            learn_features=learn_features,
        )
        self.min_std, self.max_std = std_clip
        self.std_reg = std_reg
        self.alpha = alpha

    def get_std(self, log_std: torch.Tensor) -> torch.Tensor:
        """
        Get the standard deviation from the learned parameter
        (log of it by default). This ensures that the std is positive.

        :param log_std:
        :return:
        """
        # Apply correction to remove scaling of action std as a function of the latent
        # dimension (see paper for details)
        log_std = log_std.clip(min=np.log(self.min_std), max=np.log(self.max_std))
        log_std = log_std - 0.5 * np.log(self.latent_sde_dim)

        if self.use_expln:
            # From gSDE paper, it allows to keep variance
            # above zero and prevent it from growing too fast
            below_threshold = torch.exp(log_std) * (log_std <= 0)
            # Avoid NaN: zeros values that are below zero
            safe_log_std = log_std * (log_std > 0) + self.epsilon
            above_threshold = (torch.log1p(safe_log_std) + 1.0) * (log_std > 0)
            std = below_threshold + above_threshold
        else:
            # Use normal exponential
            std = torch.exp(log_std)

        if self.full_std:
            assert std.shape == (
                self.latent_sde_dim,
                self.latent_sde_dim + self.action_dim,
            )
            corr_std = std[:, : self.latent_sde_dim]
            ind_std = std[:, -self.action_dim :]
        else:
            # Reduce the number of parameters:
            assert std.shape == (self.latent_sde_dim, 2), std.shape
            corr_std = torch.ones(self.latent_sde_dim, self.latent_sde_dim).to(log_std.device) * std[:, 0:1]
            ind_std = torch.ones(self.latent_sde_dim, self.action_dim).to(log_std.device) * std[:, 1:]
        return corr_std, ind_std

    def sample_weights(self, log_std: torch.Tensor, batch_size: int = 1) -> None:
        """
        Sample weights for the noise exploration matrix,
        using a centered Gaussian distribution.

        :param log_std:
        :param batch_size:
        """
        corr_std, ind_std = self.get_std(log_std)
        self.corr_weights_dist = Normal(torch.zeros_like(corr_std), corr_std)
        self.ind_weights_dist = Normal(torch.zeros_like(ind_std), ind_std)

        # Reparametrization trick to pass gradients
        self.corr_exploration_mat = self.corr_weights_dist.rsample()
        self.ind_exploration_mat = self.ind_weights_dist.rsample()

        # Pre-compute matrices in case of parallel exploration
        self.corr_exploration_matrices = self.corr_weights_dist.rsample((batch_size,))
        self.ind_exploration_matrices = self.ind_weights_dist.rsample((batch_size,))

    def proba_distribution_net(
        self,
        latent_dim: int,
        log_std_init: float = 0,
        latent_sde_dim: Optional[int] = None,
        clip_mean: float = 0,
    ) -> Tuple[nn.Module, nn.Parameter]:
        """
        Create the layers and parameter that represent the distribution:
        one output will be the deterministic action, the other parameter will be the
        standard deviation of the distribution that control the weights of the noise matrix,
        both for the action perturbation and the latent perturbation.

        :param latent_dim: Dimension of the last layer of the policy (before the action layer)
        :param log_std_init: Initial value for the log standard deviation
        :param latent_sde_dim: Dimension of the last layer of the features extractor
            for gSDE. By default, it is shared with the policy network.
        :param clip_mean: From SB3 implementation of SAC, add possibility to hard clip the
            mean of the actions.
        :return:
        """
        # Note: we always consider that the noise is based on the features of the last
        # layer, so latent_sde_dim is the same as latent_dim
        self.mean_actions_net = nn.Linear(latent_dim, self.action_dim)
        if clip_mean > 0:
            self.clipped_mean_actions_net = nn.Sequential(
                self.mean_actions_net,
                nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean))
        else:
            self.clipped_mean_actions_net = self.mean_actions_net
        self.latent_sde_dim = latent_dim if latent_sde_dim is None else latent_sde_dim

        log_std = (
            torch.ones(self.latent_sde_dim, self.latent_sde_dim + self.action_dim)
            if self.full_std
            else torch.ones(self.latent_sde_dim, 2)
        )

        # Transform it into a parameter so it can be optimized
        log_std = nn.Parameter(log_std * log_std_init, requires_grad=True)
        # Sample an exploration matrix
        self.sample_weights(log_std)
        return self.clipped_mean_actions_net, log_std

    def proba_distribution(
        self,
        mean_actions: torch.Tensor,
        log_std: torch.Tensor,
        latent_sde: torch.Tensor,
    ) -> "LatticeNoiseDistribution":
        # Detach the last layer features because we do not want to update the noise generation
        # to influence the features of the policy
        self._latent_sde = latent_sde if self.learn_features else latent_sde.detach()
        corr_std, ind_std = self.get_std(log_std)
        latent_corr_variance = torch.mm(self._latent_sde**2, corr_std**2)  # Variance of the hidden state
        latent_ind_variance = torch.mm(self._latent_sde**2, ind_std**2) + self.std_reg**2  # Variance of the action

        # First consider the correlated variance
        sigma_mat = self.alpha**2 * (self.mean_actions_net.weight * latent_corr_variance[:, None, :]).matmul(
            self.mean_actions_net.weight.T
        )
        # Then the independent one, to be added to the diagonal
        sigma_mat[:, range(self.action_dim), range(self.action_dim)] += latent_ind_variance
        self.distribution = MultivariateNormal(loc=mean_actions, covariance_matrix=sigma_mat, validate_args=False)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self.bijector is not None:
            gaussian_actions = self.bijector.inverse(actions)
        else:
            gaussian_actions = actions
        log_prob = self.distribution.log_prob(gaussian_actions)

        if self.bijector is not None:
            # Squash correction
            log_prob -= torch.sum(self.bijector.log_prob_correction(gaussian_actions), dim=1)
        return log_prob

    def entropy(self) -> torch.Tensor:
        if self.bijector is not None:
            return None
        return self.distribution.entropy()

    def get_noise(
        self,
        latent_sde: torch.Tensor,
        exploration_mat: torch.Tensor,
        exploration_matrices: torch.Tensor,
    ) -> torch.Tensor:
        latent_sde = latent_sde if self.learn_features else latent_sde.detach()
        # Default case: only one exploration matrix
        if len(latent_sde) == 1 or len(latent_sde) != len(exploration_matrices):
            return torch.mm(latent_sde, exploration_mat)
        # Use batch matrix multiplication for efficient computation
        # (batch_size, n_features) -> (batch_size, 1, n_features)
        latent_sde = latent_sde.unsqueeze(dim=1)
        # (batch_size, 1, n_actions)
        noise = torch.bmm(latent_sde, exploration_matrices)
        return noise.squeeze(dim=1)

    def sample(self) -> torch.Tensor:
        latent_noise = self.alpha * self.get_noise(self._latent_sde, self.corr_exploration_mat, self.corr_exploration_matrices)
        action_noise = self.get_noise(self._latent_sde, self.ind_exploration_mat, self.ind_exploration_matrices)
        actions = self.clipped_mean_actions_net(self._latent_sde + latent_noise) + action_noise
        if self.bijector is not None:
            return self.bijector.forward(actions)
        return actions
