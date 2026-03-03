# sb3-soft
Reinforcement learning algorithms for soft Q targets compatible with Stable-Baselines3.
Currently only discrete action spaces are supported.

The implemented algorithms are:
- Soft Q-Learning (SQL),
- Stable Discrete Soft Actor-Critic (SDSAC).


## Installation
You can install the package using pip (or uv):

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