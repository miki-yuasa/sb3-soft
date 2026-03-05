# sb3-soft

`sb3-soft` provides reinforcement learning algorithms with soft Q-targets,
implemented on top of
[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3).

Current scope:

- Discrete action spaces
- SQL (Soft Q-Learning)
- SDSAC (Stable Discrete Soft Actor-Critic)

## Why sb3-soft?

- Familiar SB3-style API (`learn`, `predict`, `save`, `load`)
- Drop-in usage for Gymnasium discrete environments
- Strong algorithm-focused implementation with clean class-level docstrings

## Installation

Install from PyPI:

```bash
pip install sb3-soft
# or
uv add sb3-soft
```

Install the latest development version:

```bash
pip install git+https://github.com/miki-yuasa/sb3-soft.git
# or
uv add git+https://github.com/miki-yuasa/sb3-soft.git
```

## Quick Start

### SQL

```python
from sb3_soft import SQL
from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env("CartPole-v1", n_envs=1)

model = SQL(
    "MlpPolicy",
    env,
    learning_rate=1e-4,
    buffer_size=100_000,
    verbose=1,
)
model.learn(total_timesteps=100_000)
model.save("sql_cartpole")
```

### SDSAC

```python
from sb3_soft import SDSAC
from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env("CartPole-v1", n_envs=1)

model = SDSAC(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    buffer_size=100_000,
    batch_size=256,
    verbose=1,
)
model.learn(total_timesteps=100_000)
model.save("sdsac_cartpole")
```

## Algorithms

### SQL (Soft Q-Learning)

- Entropy-regularized Bellman backups via soft value targets
- Boltzmann (softmax) sampling over Q-values
- Optional automatic entropy-coefficient tuning

### SDSAC (Stable Discrete SAC)

- Categorical actor + twin critics for discrete actions
- Double-average Q-learning (mean twin target)
- Q-clip critic loss and entropy-penalty term for stability
- Replay buffers that store per-transition old policy entropy

## Documentation

- API and usage docs: https://miki-yuasa.github.io/sb3-soft/
- Documentation is generated from in-code docstrings using Sphinx.

Build docs locally:

```bash
uv sync --group dev
cd docs
uv run sphinx-build -b html . _build/html
```

## Development

Set up a local development environment:

```bash
uv sync --group dev --group lint
```

Run tests:

```bash
uv run pytest
```

## Citation

```bibtex
@misc{yuasa2026sb3soft,
  author = {Yuasa, Mikihisa},
  title = {sb3-soft},
  year = {2026},
  howpublished = {\url{https://github.com/miki-yuasa/sb3-soft}}
}
```