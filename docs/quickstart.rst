Quickstart
==========

Installation
------------

Install from PyPI:

.. code-block:: bash

   pip install sb3-soft

Or install from source:

.. code-block:: bash

   git clone https://github.com/miki-yuasa/sb3-soft.git
   cd sb3-soft
   pip install -e .

Basic SQL usage
---------------

.. code-block:: python

   from sb3_soft import SQL

   env = ...
   model = SQL("MlpPolicy", env, verbose=1)
   model.learn(total_timesteps=100_000)
   model.save("sql_model")

Basic SDSAC usage
-----------------

.. code-block:: python

   from sb3_soft import SDSAC

   env = ...
   model = SDSAC("MlpPolicy", env, verbose=1)
   model.learn(total_timesteps=100_000)
   model.save("sdsac_model")

Notes
-----

- Discrete action spaces are supported.
- APIs follow Stable-Baselines3 conventions.
