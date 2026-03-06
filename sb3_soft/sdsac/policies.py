"""Policies for Stable Discrete Soft Actor-Critic (SDSAC).

Implements a discrete-action actor-critic architecture following
Zhou et al. (2024), "Revisiting Discrete Soft Actor-Critic".

The actor outputs a categorical distribution over discrete actions.
Twin critics each output Q-values for all actions given a state.
"""

from typing import Any, Optional, Union

import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import (
    CategoricalDistribution,
    Distribution,
    MultiCategoricalDistribution,
)
from stable_baselines3.common.policies import BaseModel, BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from torch import nn


class DiscreteActor(BasePolicy):
    """Actor (policy) network for discrete-action SAC.

    Outputs a categorical distribution over the discrete action space
    via a softmax over learned logits.

    Parameters
    ----------
    observation_space : spaces.Space
        Observation space.
    action_space : spaces.Discrete
        Discrete action space.
    net_arch : list[int]
        Network architecture (list of hidden layer sizes).
    features_extractor : nn.Module
        Network used to extract features from observations.
    features_dim : int
        Dimensionality of extracted features.
    activation_fn : type[nn.Module], default=nn.ReLU
        Activation function.
    normalize_images : bool, default=True
        Whether to normalize images by dividing by 255.
    """

    action_space: spaces.Discrete
    action_dist: Union[CategoricalDistribution, MultiCategoricalDistribution]

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        net_arch: list[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn

        latent_net = create_mlp(features_dim, -1, net_arch, activation_fn)
        self.latent_pi = nn.Sequential(*latent_net)
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim

        if isinstance(action_space, spaces.Discrete):
            self.action_dist = CategoricalDistribution(int(action_space.n))
        elif isinstance(action_space, spaces.MultiDiscrete):
            self.action_dist = MultiCategoricalDistribution(list(action_space.nvec))
        else:
            raise NotImplementedError(
                f"Unsupported action space type for DiscreteActor: {type(action_space)}"
            )

        self.action_net = self.action_dist.proba_distribution_net(
            latent_dim=last_layer_dim
        )

    def get_action_dist_params(self, obs: PyTorchObs) -> th.Tensor:
        """Compute action logits from observations.

        Parameters
        ----------
        obs : PyTorchObs
            Batched observations.

        Returns
        -------
        th.Tensor
            Raw logits of shape ``(batch, n_actions)``.
        """
        features = self.extract_features(obs, self.features_extractor)
        latent = self.latent_pi(features)
        return self.action_net(latent)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        action_logits = self.action_net(latent_pi)
        if isinstance(self.action_dist, CategoricalDistribution):
            return self.action_dist.proba_distribution(action_logits=action_logits)
        if isinstance(self.action_dist, MultiCategoricalDistribution):
            return self.action_dist.proba_distribution(action_logits=action_logits)
        raise ValueError("Invalid action distribution")

    def get_action_probs(
        self, obs: PyTorchObs, epsilon: float = 1e-8
    ) -> tuple[th.Tensor, th.Tensor]:
        """Get action probabilities and log-probabilities.

        Parameters
        ----------
        obs : PyTorchObs
            Batched observations.
        epsilon : float, default=1e-8
            Unused placeholder for API compatibility.

        Returns
        -------
        tuple[th.Tensor, th.Tensor]
            Tuple ``(probs, log_probs)``, each of shape ``(batch, n_actions)``.
        """
        del epsilon
        logits = self.get_action_dist_params(obs)
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        if isinstance(distribution, CategoricalDistribution):
            probs = distribution.distribution.probs
            log_probs = distribution.distribution.logits
        elif isinstance(distribution, MultiCategoricalDistribution):
            # SDSAC currently supports only Discrete action spaces.
            probs = th.cat([dist.probs for dist in distribution.distribution], dim=1)
            log_probs = th.cat(
                [dist.logits for dist in distribution.distribution], dim=1
            )
        else:
            raise ValueError("Invalid action distribution")

        # Clamp for numerical stability in entropy/loss computations.
        log_probs = th.clamp(log_probs, min=th.finfo(log_probs.dtype).min)
        return probs, log_probs

    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        """Select actions from observations.

        Parameters
        ----------
        obs : PyTorchObs
            Batched observations.
        deterministic : bool, default=False
            If ``True``, return greedy actions (argmax).

        Returns
        -------
        th.Tensor
            Selected action indices of shape ``(batch,)``.
        """
        distribution = self.get_distribution(obs)
        return distribution.get_actions(deterministic=deterministic)

    def get_distribution(self, obs: PyTorchObs) -> Distribution:
        """Get the current policy distribution given observations."""
        features = super().extract_features(obs, self.features_extractor)
        latent_pi = self.latent_pi(features)
        return self._get_action_dist_from_latent(latent_pi)

    def _predict(
        self, observation: PyTorchObs, deterministic: bool = False
    ) -> th.Tensor:
        return self(observation, deterministic)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data


class DiscreteCritic(BaseModel):
    """Twin Q-network critic for discrete-action SAC.

    Each Q-network takes a state as input and outputs Q-values for every
    discrete action.  Multiple networks (default: 2) are used to reduce
    overestimation via clipped double Q-learning.

    Parameters
    ----------
    observation_space : spaces.Space
        Observation space.
    action_space : spaces.Discrete
        Discrete action space.
    net_arch : list[int]
        Network architecture for each Q-network.
    features_extractor : BaseFeaturesExtractor
        Network used to extract features from observations.
    features_dim : int
        Dimensionality of extracted features.
    activation_fn : type[nn.Module], default=nn.ReLU
        Activation function.
    normalize_images : bool, default=True
        Whether to normalize images by dividing by 255.
    n_critics : int, default=2
        Number of Q-networks to create.
    share_features_extractor : bool, default=True
        Whether the features extractor is shared with the actor. If ``True``,
        gradients through it are blocked in the critic forward pass.
    """

    features_extractor: BaseFeaturesExtractor

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        net_arch: list[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        n_actions = int(action_space.n)
        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks: list[nn.Module] = []
        for idx in range(n_critics):
            q_net_list = create_mlp(features_dim, n_actions, net_arch, activation_fn)
            q_net = nn.Sequential(*q_net_list)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: th.Tensor) -> tuple[th.Tensor, ...]:
        """Compute Q-values for all actions from all critic networks.

        Parameters
        ----------
        obs : th.Tensor
            Batched observations.

        Returns
        -------
        tuple[th.Tensor, ...]
            Q-value tensors, one per critic, each of shape
            ``(batch, n_actions)``.
        """
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        return tuple(q_net(features) for q_net in self.q_networks)

    def q1_forward(self, obs: th.Tensor) -> th.Tensor:
        """Compute Q-values using only the first critic network.

        Parameters
        ----------
        obs : th.Tensor
            Batched observations.

        Returns
        -------
        th.Tensor
            Q-values of shape ``(batch, n_actions)``.
        """
        with th.no_grad():
            features = self.extract_features(obs, self.features_extractor)
        return self.q_networks[0](features)


class SDSACPolicy(BasePolicy):
    """Policy class (actor + twin critics) for discrete-action SAC.

    Parameters
    ----------
    observation_space : spaces.Space
        Observation space.
    action_space : spaces.Discrete
        Discrete action space.
    lr_schedule : Schedule
        Learning rate schedule.
    net_arch : Optional[Union[list[int], dict[str, list[int]]]], default=None
        Network architecture specification. Can be a list of integers (shared)
        or a dictionary with ``"pi"`` and ``"qf"`` keys.
    activation_fn : type[nn.Module], default=nn.ReLU
        Activation function.
    features_extractor_class : type[BaseFeaturesExtractor], default=FlattenExtractor
        Features extractor class.
    features_extractor_kwargs : Optional[dict[str, Any]], default=None
        Keyword arguments for the features extractor.
    normalize_images : bool, default=True
        Whether to normalize images by dividing by 255.
    optimizer_class : type[th.optim.Optimizer], default=th.optim.Adam
        Optimizer class.
    optimizer_kwargs : Optional[dict[str, Any]], default=None
        Additional optimizer keyword arguments.
    n_critics : int, default=2
        Number of critic networks.
    share_features_extractor : bool, default=False
        Whether to share the features extractor between actor and critic.
    """

    actor: DiscreteActor
    critic: DiscreteCritic
    critic_target: DiscreteCritic

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [256, 256]

        actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self.actor_kwargs = self.net_args.copy()
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
            }
        )

        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        self.actor = self.make_actor()
        self.actor.optimizer = self.optimizer_class(
            self.actor.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        if self.share_features_extractor:
            self.critic = self.make_critic(
                features_extractor=self.actor.features_extractor
            )
            # Don't optimize the shared features extractor with the critic loss
            critic_parameters = [
                param
                for name, param in self.critic.named_parameters()
                if "features_extractor" not in name
            ]
        else:
            self.critic = self.make_critic(features_extractor=None)
            critic_parameters = list(self.critic.parameters())

        # Target network gets its own features extractor
        self.critic_target = self.make_critic(features_extractor=None)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic.optimizer = self.optimizer_class(
            critic_parameters,
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        # Target should always be in eval mode
        self.critic_target.set_training_mode(False)

    def make_actor(
        self, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> DiscreteActor:
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, features_extractor
        )
        return DiscreteActor(**actor_kwargs).to(self.device)

    def make_critic(
        self, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> DiscreteCritic:
        critic_kwargs = self._update_features_extractor(
            self.critic_kwargs, features_extractor
        )
        return DiscreteCritic(**critic_kwargs).to(self.device)

    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(
        self, observation: PyTorchObs, deterministic: bool = False
    ) -> th.Tensor:
        return self.actor(observation, deterministic)

    def evaluate_actions(
        self, obs: PyTorchObs, actions: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """Evaluate actions using current actor distribution and critics.

        Returns the mean twin-critic Q-value for the provided actions,
        action log-probabilities under the current actor and distribution entropy.
        """
        distribution = self.get_distribution(obs)
        action_indices = actions.long().reshape(-1)
        log_prob = distribution.log_prob(action_indices)
        entropy = distribution.entropy()

        q_values_all = th.stack(self.critic(obs), dim=0).mean(dim=0)
        values = th.gather(q_values_all, dim=1, index=action_indices.unsqueeze(-1))
        return values, log_prob, entropy

    def get_distribution(self, obs: PyTorchObs) -> Distribution:
        """Get the current actor distribution given observations."""
        return self.actor.get_distribution(obs)

    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        """Estimate state values under the current policy.

        Computes V(s) = sum_a pi(a|s) * mean_i Q_i(s, a).
        """
        probs, _ = self.actor.get_action_probs(obs)
        q_values_all = th.stack(self.critic(obs), dim=0).mean(dim=0)
        return (probs * q_values_all).sum(dim=1, keepdim=True)

    def set_training_mode(self, mode: bool) -> None:
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.net_args["activation_fn"],
                n_critics=self.critic_kwargs["n_critics"],
                lr_schedule=self._dummy_schedule,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                share_features_extractor=self.share_features_extractor,
            )
        )
        return data


MlpPolicy = SDSACPolicy


class CnnPolicy(SDSACPolicy):
    """SDSAC policy with ``NatureCNN`` features extractor."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        features_extractor_class: type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )


class MultiInputPolicy(SDSACPolicy):
    """SDSAC policy with ``CombinedExtractor`` for dict observations."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        features_extractor_class: type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )
