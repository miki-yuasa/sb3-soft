"""Stable Discrete Soft Actor-Critic (SD-SAC).

Actor-critic algorithm for discrete action spaces based on
Zhou et al. (2024), "Revisiting Discrete Soft Actor-Critic".

Key differences from continuous SAC and naive discrete SAC:

- The actor outputs a categorical distribution over discrete actions.
- Twin critics output Q-values for *all* actions given a state (no action
  input), enabling exact expectation computation.
- **Double-average Q-learning**: the target uses ``mean`` (not ``min``)
  of the twin target critics.
- **Q-clip**: the critic loss is
  ``max((Q - y)², (Q' + clip(Q - Q', -c, c) - y)²)``.
- **Entropy-penalty**: the actor loss includes
  ``β · ½ · (H_πold − H_π)²`` where ``H_πold`` is stored in the
  replay buffer at collection time.
"""

from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update

from .buffers import (
    SDSACDictReplayBuffer,
    SDSACReplayBuffer,
    SDSACReplayBufferSamples,
)
from .policies import (
    CnnPolicy,
    DiscreteActor,
    DiscreteCritic,
    MlpPolicy,
    MultiInputPolicy,
    SDSACPolicy,
)

SelfSDSAC = TypeVar("SelfSDSAC", bound="SDSAC")


class SDSAC(OffPolicyAlgorithm):
    """Stable Discrete Soft Actor-Critic (SD-SAC).

    An off-policy actor-critic algorithm for discrete action spaces that
    maintains separate actor and twin-critic networks and performs
    entropy-regularized updates using full-distribution expectations.

    Compared with a naive discrete adaptation of SAC, SD-SAC adds three
    stabilisation mechanisms (Algorithm 1 in the paper):

    1. **Double-average Q-learning** – the Bellman target uses
       ``mean(Q'_1, Q'_2)`` of the twin target critics instead of ``min``.
    2. **Q-clip** – the critic loss is
       ``max((Q - y)^2, (Q' + clip(Q - Q', -c, c) - y)^2)``.
    3. **Entropy penalty** – the actor loss adds
       ``beta * 0.5 * (H_pi_old - H_pi)^2`` where ``H_pi_old`` is the
       policy entropy stored in the replay buffer at collection time.

    Reference: Zhou et al.  (2024) "Revisiting Discrete Soft Actor-Critic".

    Parameters
    ----------
    policy : str | type[SDSACPolicy]
        Policy model to use (``"MlpPolicy"``, ``"CnnPolicy"``, …).
    env : GymEnv | str
        Environment to learn from.
    learning_rate : float | Schedule, default=3e-4
        Learning rate for all networks (actor, critic, and optionally alpha).
    buffer_size : int, default=1_000_000
        Replay buffer capacity.
    learning_starts : int, default=100
        Number of environment steps to collect before training starts.
    batch_size : int, default=256
        Mini-batch size for each gradient update.
    tau : float, default=0.005
        Polyak averaging coefficient for target network updates.
    gamma : float, default=0.99
        Discount factor.
    train_freq : int | tuple[int, str], default=1
        How often to update the model (in steps or episodes).
    gradient_steps : int, default=1
        Gradient updates per rollout step.  ``-1`` means as many as
        environment steps collected.
    replay_buffer_class : type[ReplayBuffer] | None, default=None
        Custom replay buffer class.  Defaults to
        :class:`~sb3_soft.sdsac.buffers.SDSACReplayBuffer`.
    replay_buffer_kwargs : dict | None, default=None
        Keyword arguments for the replay buffer.
    optimize_memory_usage : bool, default=False
        Memory-efficient replay buffer variant.
    n_steps : int, default=1
        Steps for n-step returns.
    ent_coef : str | float, default="auto"
        Entropy coefficient (temperature) :math:`\\alpha`.  ``"auto"``
        enables automatic tuning (``"auto_0.1"`` sets the initial value).
    target_update_interval : int, default=1
        Gradient steps between target network updates.
    target_entropy : str | float, default="auto"
        Target entropy for automatic :math:`\\alpha` tuning.  ``"auto"``
        uses :math:`0.98 \\log |\\mathcal{A}|`.
    beta : float, default=0.1
        Entropy-penalty coefficient :math:`\\beta` in the actor loss.
    clip_range : float, default=0.5
        Clipping range :math:`c` for the Q-clip critic loss.
    stats_window_size : int, default=100
        Window size for rollout statistics.
    tensorboard_log : str | None, default=None
        TensorBoard log directory.
    policy_kwargs : dict | None, default=None
        Extra keyword arguments for policy construction.
    verbose : int, default=0
        Verbosity level (0: silent, 1: info, 2: debug).
    seed : int | None, default=None
        Random seed.
    device : str | th.device, default="auto"
        Computation device.
    _init_setup_model : bool, default=True
        Whether to build networks on construction.
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    policy: SDSACPolicy
    actor: DiscreteActor
    critic: DiscreteCritic
    critic_target: DiscreteCritic

    def __init__(
        self,
        policy: Union[str, type[SDSACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 1,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        n_steps: int = 1,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        beta: float = 0.5,
        clip_range: float = 0.5,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            n_steps=n_steps,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            supported_action_spaces=(spaces.Discrete,),
            support_multi_env=True,
        )

        self.target_entropy = target_entropy
        self.log_ent_coef: Optional[th.Tensor] = None
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer: Optional[th.optim.Adam] = None
        self.beta = beta
        self.clip_range = clip_range

        if _init_setup_model:
            self._setup_model()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_model(self) -> None:
        if self.replay_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                self.replay_buffer_class = SDSACDictReplayBuffer
            else:
                self.replay_buffer_class = SDSACReplayBuffer

        super()._setup_model()
        self._create_aliases()

        # Batch-norm running statistics for Polyak updates
        self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(
            self.critic_target, ["running_"]
        )

        # Target entropy
        if self.target_entropy == "auto":
            assert isinstance(self.action_space, spaces.Discrete)
            # 0.98 * log(|A|) as suggested in the paper
            self.target_entropy = float(0.98 * np.log(self.action_space.n))
        else:
            self.target_entropy = float(self.target_entropy)

        # Entropy coefficient (alpha)
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            init_value = 0.1
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, (
                    "The initial value of ent_coef must be greater than 0"
                )

            self.log_ent_coef = th.log(
                th.ones(1, device=self.device) * init_value
            ).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam(
                [self.log_ent_coef], lr=self.lr_schedule(1)
            )
        else:
            self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor  # type: ignore[assignment]
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        """Store a transition and record the current-policy entropy.

        Computes :math:`H_{\\pi_{\\text{old}}}` for each environment and
        writes it to the :class:`SDSACReplayBuffer` at the current
        position *before* the base class increments the write pointer.
        """
        if isinstance(replay_buffer, (SDSACReplayBuffer, SDSACDictReplayBuffer)):
            with th.no_grad():
                assert self._last_obs is not None
                obs_tensor, _ = self.policy.obs_to_tensor(self._last_obs)
                probs, log_probs = self.actor.get_action_probs(obs_tensor)
                # H = -sum_a pi(a|s) log pi(a|s), shape (n_envs,)
                entropy = -(probs * log_probs).sum(dim=-1).cpu().numpy()
            replay_buffer.set_entropy(entropy)

        super()._store_transition(
            replay_buffer, buffer_action, new_obs, reward, dones, infos
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode
        self.policy.set_training_mode(True)

        # Prepare optimizers
        optimizers: list[th.optim.Optimizer] = [
            self.actor.optimizer,
            self.critic.optimizer,
        ]
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)

        # Update learning rates
        self._update_learning_rate(optimizers)

        ent_coef_losses: list[float] = []
        ent_coefs: list[float] = []
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        ent_penalties: list[float] = []
        q_value_means: list[float] = []
        q_value_means_qf0: list[float] = []
        q_value_means_qf1: list[float] = []

        for gradient_step in range(gradient_steps):
            # ---- Sample replay buffer ----
            replay_data = self.replay_buffer.sample(  # type: ignore[union-attr]
                batch_size, env=self._vec_normalize_env
            )
            discounts = (
                replay_data.discounts
                if replay_data.discounts is not None
                else self.gamma
            )

            # ---- Current policy distribution ----
            probs, log_probs = self.actor.get_action_probs(
                replay_data.observations
            )  # (B, |A|)

            # Per-state policy entropy: H = -sum_a pi(a|s) log pi(a|s)
            entropy = -(probs * log_probs).sum(dim=1, keepdim=True)  # (B, 1)
            entropy = th.nan_to_num(entropy, nan=0.0, posinf=1e6, neginf=0.0)

            # ---- Entropy coefficient (alpha) ----
            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                # J(alpha) = alpha * (H(pi) - H_target)
                ent_coef_loss = (
                    self.log_ent_coef * (entropy.detach() - self.target_entropy)
                ).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize alpha
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            # ---- Critic update ----
            with th.no_grad():
                # Next-state policy distribution
                next_probs, next_log_probs = self.actor.get_action_probs(
                    replay_data.next_observations
                )  # (B, |A|)

                # Double-average Q-learning: use *mean* of twin target
                # critics instead of min (Algorithm 1, line 8).
                next_q_values = th.stack(
                    self.critic_target(replay_data.next_observations), dim=0
                )  # (n_critics, B, |A|)
                next_q_values_avg = next_q_values.mean(dim=0)  # (B, |A|)

                # V(s') = sum_a pi(a|s') * [Q_target(s',a) - alpha * log pi(a|s')]
                next_v = (
                    next_probs * (next_q_values_avg - ent_coef * next_log_probs)
                ).sum(dim=1, keepdim=True)  # (B, 1)

                # TD target y
                target_q_values = (
                    replay_data.rewards + (1 - replay_data.dones) * discounts * next_v
                )  # (B, 1)
                target_q_values = th.nan_to_num(
                    target_q_values, nan=0.0, posinf=1e6, neginf=-1e6
                )

                # Target-critic Q-values for Q-clip (need per-critic)
                target_q_all = self.critic_target(
                    replay_data.observations
                )  # tuple of (B, |A|)

            # Current Q-values for taken actions
            actions_long = replay_data.actions.long()
            current_q_all = self.critic(replay_data.observations)  # tuple of (B, |A|)

            # Q-clip loss (Algorithm 1, line 10):
            # L(theta_i) = max((Q_i - y)^2, (Q'_i + clip(Q_i - Q'_i, -c, c) - y)^2)
            critic_loss = th.zeros(1, device=self.device)
            q_taken_means: list[th.Tensor] = []
            for q_local, q_target in zip(current_q_all, target_q_all):
                q_local_a = th.gather(q_local, dim=1, index=actions_long)  # (B, 1)
                q_target_a = th.gather(q_target, dim=1, index=actions_long)  # (B, 1)
                q_local_a = th.nan_to_num(q_local_a, nan=0.0, posinf=1e6, neginf=-1e6)
                q_target_a = th.nan_to_num(q_target_a, nan=0.0, posinf=1e6, neginf=-1e6)
                q_taken_means.append(q_local_a.mean())
                loss_plain = (q_local_a - target_q_values).pow(2)  # (B, 1)
                q_clipped = q_target_a + th.clamp(
                    q_local_a - q_target_a,
                    -self.clip_range,
                    self.clip_range,
                )
                loss_clipped = (q_clipped - target_q_values).pow(2)  # (B, 1)
                critic_loss = critic_loss + th.max(loss_plain, loss_clipped).mean()

            if len(q_taken_means) > 0:
                q_value_means.append(th.stack(q_taken_means).mean().item())
                q_value_means_qf0.append(q_taken_means[0].item())
                if len(q_taken_means) > 1:
                    q_value_means_qf1.append(q_taken_means[1].item())

            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            # Optimize critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            th.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
            self.critic.optimizer.step()

            # ---- Actor update ----
            # Re-compute probs with fresh graph (critic was just updated)
            probs_pi, log_probs_pi = self.actor.get_action_probs(
                replay_data.observations
            )

            # Q-values from all critics (no grad through critic)
            with th.no_grad():
                q_values_all = th.stack(
                    self.critic(replay_data.observations), dim=0
                )  # (n_critics, B, |A|)
                q_values_min, _ = q_values_all.min(dim=0)  # (B, |A|)
                q_values_min = th.nan_to_num(
                    q_values_min, nan=0.0, posinf=1e6, neginf=-1e6
                )

            # J_pi = E_s [ sum_a pi(a|s) * (alpha * log pi(a|s) - Q(s,a)) ]
            actor_loss = (
                (probs_pi * (ent_coef * log_probs_pi - q_values_min)).sum(dim=1).mean()
            )
            actor_loss = th.nan_to_num(actor_loss, nan=0.0, posinf=1e6, neginf=-1e6)

            # Entropy-penalty (Algorithm 1, line 12):
            # J_pi += beta * 0.5 * (H_pi_old - H_pi)^2
            current_entropy = -(probs_pi * log_probs_pi).sum(
                dim=1, keepdim=True
            )  # (B, 1)
            assert isinstance(replay_data, SDSACReplayBufferSamples)
            entropy_penalty = (
                self.beta
                * 0.5
                * (replay_data.old_entropies - current_entropy).pow(2).mean()
            )
            actor_loss = actor_loss + entropy_penalty

            actor_losses.append(actor_loss.item())
            ent_penalties.append(entropy_penalty.item())

            # Optimize actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            th.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
            self.actor.optimizer.step()

            # ---- Target network update ----
            if gradient_step % self.target_update_interval == 0:
                polyak_update(
                    self.critic.parameters(),
                    self.critic_target.parameters(),
                    self.tau,
                )
                polyak_update(
                    self.batch_norm_stats,
                    self.batch_norm_stats_target,
                    1.0,
                )

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/ent_penalty", np.mean(ent_penalties))
        if len(q_value_means) > 0:
            self.logger.record("train/q_value_mean", np.mean(q_value_means))
        if len(q_value_means_qf0) > 0:
            self.logger.record("train/q_value_mean_qf0", np.mean(q_value_means_qf0))
        if len(q_value_means_qf1) > 0:
            self.logger.record("train/q_value_mean_qf1", np.mean(q_value_means_qf1))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    # ------------------------------------------------------------------
    # Predict / Learn / Save helpers
    # ------------------------------------------------------------------

    def predict(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        return self.policy.predict(observation, state, episode_start, deterministic)

    def learn(
        self: SelfSDSAC,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "SDSAC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfSDSAC:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> list[str]:
        return [
            *super()._excluded_save_params(),
            "actor",
            "critic",
            "critic_target",
        ]

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables
