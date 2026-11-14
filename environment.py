"""
environment.py

Responsible for creating and configuring the FrozenLake-v1 Gymnasium
environment. This is the only file that knows environment-specific setup
details -- everything else just receives an already-configured env object.
"""

import gymnasium as gym

from config import MAP_NAME, IS_SLIPPERY, MAX_STEPS_PER_EPISODE

# Gymnasium's own default step limits per built-in map.
DEFAULT_MAX_STEPS = {"4x4": 100, "8x8": 200}


def make_env(render_mode=None, map_name=None, is_slippery=None, max_steps=None):
    """
    Create and return a configured FrozenLake-v1 environment.

    All arguments are optional and fall back to the values in config.py, so
    existing callers (main.py) behave exactly as before. The API backend
    passes explicit values to build environments on demand.

    render_mode=None    -> no rendering (fast, used for training/evaluation)
    render_mode="human" -> opens a visual window (manual demo run)
    """
    map_name = MAP_NAME if map_name is None else map_name
    is_slippery = IS_SLIPPERY if is_slippery is None else is_slippery
    if max_steps is None:
        max_steps = (
            MAX_STEPS_PER_EPISODE
            if map_name == MAP_NAME
            else DEFAULT_MAX_STEPS.get(map_name, MAX_STEPS_PER_EPISODE)
        )

    return gym.make(
        "FrozenLake-v1",
        map_name=map_name,
        is_slippery=is_slippery,
        max_episode_steps=max_steps,
        render_mode=render_mode,
    )


def get_layout(env):
    """Return the map as a list of strings, e.g. ['SFFF', 'FHFH', ...]."""
    desc = env.unwrapped.desc
    return ["".join(c.decode() for c in row) for row in desc]
