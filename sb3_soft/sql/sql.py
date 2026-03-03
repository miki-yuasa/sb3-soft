from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.dqn.policies import QNetwork
from torch.nn import functional as F

from .policies import CnnPolicy, MlpPolicy, MultiInputPolicy, SQLPolicy

SelfSQL = TypeVar("SelfSQL", bound="SQL")


class SQL(OffPolicyAlgorithm):
    """Discrete-action Soft Q-Learning.

    Extends SB3's ``OffPolicyAlgorithm`` with an entropy-regularized Bellman
    backup and Boltzmann (softmax) action sampling.

    Parameters
    ----------
    policy : str | type[SQLPolicy]
        Policy model to use (e.g., ``"MlpPolicy"``, ``"CnnPolicy"``).
    env : GymEnv | str
        Environment to learn from.
    learning_rate : float | Schedule, default=1e-4
        Learning rate (constant or schedule).
    buffer_size : int, default=1_000_000
        Replay buffer capacity.
    learning_starts : int, default=100
        Number of steps of random exploration before learning starts.
    batch_size : int, default=32
        Minibatch size for each gradient update.
    tau : float, default=1.0
        Polyak update coefficient for target network updates.
    gamma : float, default=0.99
        Discount factor.
    train_freq : int | tuple[int, str], default=4
        Update frequency in steps or episodes.
    gradient_steps : int, default=1
        Number of gradient updates after each rollout.
    action_noise : ActionNoise | None, default=None
        Action noise for exploration (only applicable to continuous action spaces).
    replay_buffer_class : type[ReplayBuffer] | None, default=None
        Optional replay buffer implementation override.
    replay_buffer_kwargs : dict[str, Any] | None, default=None
        Additional keyword arguments for replay buffer creation.
    optimize_memory_usage : bool, default=False
        Whether to use the memory-efficient replay buffer variant.
    n_steps : int, default=1
        Number of steps for n-step returns.
    target_update_interval : int, default=10_000
        Environment steps between target network updates.
    max_grad_norm : float, default=10
        Maximum gradient norm for clipping.
    ent_coef : str | float, default="auto"
        Temperature :math:`\\alpha` used in the soft Bellman target
        :math:`V(s') = \\alpha \\log \\sum_a \\exp(Q(s', a) / \\alpha)`.
        Set to ``"auto"`` (or ``"auto_0.1"``) to learn it automatically.
    target_entropy : str | float, default="auto"
        Target policy entropy used when ``ent_coef`` is learned automatically.
        If ``"auto"``, uses :math:`0.98 \\log(|\\mathcal{A}|)`.
    action_temperature : float | None, default=None
        Temperature :math:`\tau` for Boltzmann action sampling
        :math:`\\pi(a \\mid s) \\propto \\exp(Q(s, a) / \\tau)`.
        If ``None``, uses ``ent_coef``.
    use_sde : bool, default=False
        Whether to use State-Dependent Exploration (SDE) instead of action noise.
    sde_sample_freq : int, default=-1
        Sample a new noise matrix every n steps when using SDE.
    use_sde_at_warmup : bool, default=False
        Whether to use SDE noise during the warmup phase (before learning starts).
    stats_window_size : int, default=100
        Window size for rollout statistics logging.
    tensorboard_log : str | None, default=None
        TensorBoard log directory.
    policy_kwargs : dict[str, Any] | None, default=None
        Additional keyword arguments passed to the policy.
    verbose : int, default=0
        Verbosity level (0: no output, 1: info, 2: debug).
    seed : int | None, default=None
        Random seed.
    device : torch.device | str, default="auto"
        Device to run the model on.
    _init_setup_model : bool, default=True
        Whether to build networks and optimizers during construction.
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }

    q_net: QNetwork
    q_net_target: QNetwork

    def __init__(
        self,
        policy: Union[str, type[SQLPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 4,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        n_steps: int = 1,
        target_update_interval: int = 10_000,
        max_grad_norm: float = 10,
        ent_coef: Union[str, float] = "auto",
        target_entropy: Union[str, float] = "auto",
        action_temperature: float | None = None,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        self.target_entropy = target_entropy
        self.ent_coef = ent_coef
        self.log_ent_coef: Optional[th.Tensor] = None
        self.ent_coef_optimizer: Optional[th.optim.Adam] = None

        self._temperature_follows_ent_coef = action_temperature is None
        init_ent_coef: float
        if isinstance(ent_coef, str):
            if not ent_coef.startswith("auto"):
                raise ValueError(
                    "ent_coef must be a float or start with 'auto' (e.g. 'auto_0.1'). "
                    f"Got: {ent_coef}"
                )
            init_ent_coef = 1.0
            if "_" in ent_coef:
                init_ent_coef = float(ent_coef.split("_")[1])
                if init_ent_coef <= 0:
                    raise ValueError(
                        f"The initial value of ent_coef must be > 0, got {init_ent_coef}"
                    )
        else:
            init_ent_coef = float(ent_coef)
            if init_ent_coef <= 0:
                raise ValueError(f"ent_coef must be > 0, got {ent_coef}")

        if self._temperature_follows_ent_coef:
            self.action_temperature = init_ent_coef
        else:
            assert action_temperature is not None
            self.action_temperature = float(action_temperature)
        if self.action_temperature <= 0:
            raise ValueError(
                f"action_temperature must be > 0 when provided, got {self.action_temperature}"
            )

        policy_kwargs = {} if policy_kwargs is None else dict(policy_kwargs)
        policy_kwargs.setdefault("temperature", self.action_temperature)

        self.target_update_interval = target_update_interval
        self.max_grad_norm = max_grad_norm
        self._n_calls = 0

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            n_steps=n_steps,
            policy_kwargs=policy_kwargs,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            supported_action_spaces=(spaces.Discrete,),
            support_multi_env=True,
        )

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        self.batch_norm_stats = get_parameters_by_name(self.q_net, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(
            self.q_net_target, ["running_"]
        )

        if self.target_entropy == "auto":
            assert isinstance(self.action_space, spaces.Discrete)
            self.target_entropy = float(0.98 * np.log(self.action_space.n))
        else:
            self.target_entropy = float(self.target_entropy)

        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            init_value = 1.0
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

    def _set_action_temperature(self, value: float) -> None:
        value = max(float(value), 1e-8)
        self.action_temperature = value
        if isinstance(self.policy, SQLPolicy):
            self.policy.temperature = value
            setattr(self.policy.q_net, "temperature", value)
            setattr(self.policy.q_net_target, "temperature", value)

    def _create_aliases(self) -> None:
        assert isinstance(self.policy, SQLPolicy)
        self.q_net = self.policy.q_net
        self.q_net_target = self.policy.q_net_target

    def _on_step(self) -> None:
        """Update the target network if needed."""
        self._n_calls += 1
        if self._n_calls % max(self.target_update_interval // self.n_envs, 1) == 0:
            polyak_update(
                self.q_net.parameters(), self.q_net_target.parameters(), self.tau
            )
            polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        optimizers: list[th.optim.Optimizer] = [self.policy.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)
        self._update_learning_rate(optimizers)

        losses: list[float] = []
        ent_coef_losses: list[float] = []
        ent_coefs: list[float] = []
        last_batch_entropy: float | None = None

        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(  # type: ignore[union-attr]
                batch_size, env=self._vec_normalize_env
            )
            q_obs = self.q_net(replay_data.observations)
            log_probs = th.log_softmax(q_obs / self.action_temperature, dim=1)
            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum(dim=1, keepdim=True)

            ent_coef_loss: Optional[th.Tensor] = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef_tensor = th.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = (
                    self.log_ent_coef * (entropy.detach() - self.target_entropy)
                ).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef_tensor = self.ent_coef_tensor

            ent_coefs.append(ent_coef_tensor.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
                assert self.log_ent_coef is not None
                ent_coef_tensor = th.exp(self.log_ent_coef.detach())

            if self._temperature_follows_ent_coef:
                self._set_action_temperature(ent_coef_tensor.item())

            discounts = (
                replay_data.discounts
                if replay_data.discounts is not None
                else self.gamma
            )

            with th.no_grad():
                next_q_values = self.q_net_target(replay_data.next_observations)
                next_v_values = ent_coef_tensor * th.logsumexp(
                    next_q_values / ent_coef_tensor, dim=1
                )
                next_v_values = next_v_values.reshape(-1, 1)
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * discounts * next_v_values
                )

            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(
                current_q_values, dim=1, index=replay_data.actions.long()
            )

            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(float(loss.item()))

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
            last_batch_entropy = entropy.mean().item()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
        if last_batch_entropy is not None:
            self.logger.record("train/entropy", last_batch_entropy)
        self.logger.record("train/action_temperature", self.action_temperature)

    def predict(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        return self.policy.predict(observation, state, episode_start, deterministic)

    def learn(
        self: SelfSQL,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "SQL",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfSQL:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> list[str]:
        return [*super()._excluded_save_params(), "q_net", "q_net_target"]

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "policy.optimizer"]
        if self.ent_coef_optimizer is not None:
            state_dicts.append("ent_coef_optimizer")
            saved_pytorch_variables = ["log_ent_coef"]
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables
