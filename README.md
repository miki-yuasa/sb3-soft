# sb3-soft
Reinforcement learning algorithms for soft Q targets compatible with Stable-Baselines3.
Currently only discrete action spaces are supported.

The implemented algorithms are:
- Soft Q-Learning (SQL),
- Stable Discrete Soft Actor-Critic (SDSAC).


## Installation
You can install the package from PyPI:

```bash
pip install sb3-soft
# or using uv
uv add sb3-soft
```

You can also install directly from GitHub:

```bash
pip install git+https://github.com/miki-yuasa/sb3-soft.git
# or using uv
uv add git+https://github.com/miki-yuasa/sb3-soft.git
```

## Usage

```python
from sb3_soft import SQL

# Create the environment
env = ...

# Create the model
model = SQL("MlpPolicy", env, ...)

# Train the model
model.learn(total_timesteps=10000)

# Save the model
model.save("sql_model")

# Load the model
model = SQL.load("sql_model")
```

## Publishing to PyPI (uv)

1. Bump the version in `pyproject.toml`.
2. Build distributions:

```bash
uv build
```

3. (Optional) Publish to TestPyPI first:

```bash
export UV_PUBLISH_URL="https://test.pypi.org/legacy/"
export UV_PUBLISH_TOKEN="<testpypi-token>"
uv publish
unset UV_PUBLISH_URL
```

4. Publish to PyPI:

```bash
export UV_PUBLISH_TOKEN="<pypi-token>"
uv publish
```

You can also pass the token directly with `uv publish --token <token>`.