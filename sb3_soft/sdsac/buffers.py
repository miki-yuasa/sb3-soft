"""Custom replay buffer for SD-SAC.

Extends the standard SB3 :class:`ReplayBuffer` with per-transition storage
of the policy entropy at collection time (``H_πold``).  This is required by
the entropy-penalty term in the SD-SAC actor loss.
"""

from typing import Any, NamedTuple, Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize


class SDSACReplayBufferSamples(NamedTuple):
    """Replay-buffer samples with an extra ``old_entropies`` field.

    Attributes
    ----------
    observations : th.Tensor
        Batch of observations.
    actions : th.Tensor
        Batch of actions.
    next_observations : th.Tensor
        Batch of next observations.
    dones : th.Tensor
        Batch of done flags.
    rewards : th.Tensor
        Batch of rewards.
    old_entropies : th.Tensor
        Policy entropy at the time the transition was collected,
        shape ``(batch, 1)``.
    discounts : th.Tensor | None
        Per-sample discount factors (used by n-step buffers).
    """

    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    old_entropies: th.Tensor
    discounts: Optional[th.Tensor] = None


class SDSACReplayBuffer(ReplayBuffer):
    """Replay buffer that additionally stores per-transition policy entropy.

    The entropy is set via :meth:`set_entropy` *before* each :meth:`add`
    call.  During sampling, the stored entropy is returned alongside the
    standard replay-buffer fields.

    Parameters
    ----------
    buffer_size : int
        Maximum number of transitions to store.
    observation_space : spaces.Space
        Observation space.
    action_space : spaces.Space
        Action space.
    device : Union[th.device, str], default="auto"
        Device for returned tensors.
    n_envs : int, default=1
        Number of parallel environments.
    optimize_memory_usage : bool, default=False
        Memory-efficient variant (see SB3 docs).
    handle_timeout_termination : bool, default=True
        Whether to handle ``TimeLimit.truncated`` in ``infos``.
    """

    old_entropies: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ) -> None:
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )
        # Allocate storage for per-transition entropy.
        # Shape mirrors self.rewards: (effective_buffer_size, n_envs).
        self.old_entropies = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self._pending_entropy: Optional[np.ndarray] = None

    def set_entropy(self, entropy: np.ndarray) -> None:
        """Stage entropy values to be written on the next :meth:`add` call.

        Parameters
        ----------
        entropy : np.ndarray
            Entropy for each environment, shape ``(n_envs,)`` or ``(n_envs, 1)``.
        """
        self._pending_entropy = np.asarray(entropy, dtype=np.float32).reshape(
            self.n_envs
        )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        """Store a transition, including any staged entropy."""
        if self._pending_entropy is not None:
            self.old_entropies[self.pos] = self._pending_entropy
            self._pending_entropy = None
        else:
            # Fallback: store zero (e.g. during warmup if entropy isn't set).
            self.old_entropies[self.pos] = 0.0

        super().add(obs, next_obs, action, reward, done, infos)

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> SDSACReplayBufferSamples:
        """Return a batch of transitions including old entropies."""
        # Randomly select one env per sample (matches parent behaviour).
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(
                self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :],
                env,
            )
        else:
            next_obs = self._normalize_obs(
                self.next_observations[batch_inds, env_indices, :], env
            )

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            (
                self.dones[batch_inds, env_indices]
                * (1 - self.timeouts[batch_inds, env_indices])
            ).reshape(-1, 1),
            self._normalize_reward(
                self.rewards[batch_inds, env_indices].reshape(-1, 1), env
            ),
            self.old_entropies[batch_inds, env_indices].reshape(-1, 1),
        )
        return SDSACReplayBufferSamples(*tuple(map(self.to_torch, data)))
