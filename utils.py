"""
utils.py

Small reusable helpers: reproducibility (seeding) and saving/loading the
Q-table and training metrics to/from disk.
"""

import os
import numpy as np


def set_seed(seed):
    """
    Create and return a seeded numpy random Generator.

    We use this single Generator everywhere randomness is needed (agent's
    epsilon-greedy choices) rather than numpy's global random state, which
    is the modern, recommended way to get reproducible randomness in numpy.
    Note: the environment itself is seeded separately, via env.reset(seed=seed),
    since Gymnasium manages its own internal random state.
    """
    return np.random.default_rng(seed)


def ensure_dir(path):
    """Create a directory (and parents) if it doesn't already exist."""
    os.makedirs(path, exist_ok=True)


def save_q_table(q_table, path):
    ensure_dir(os.path.dirname(path))
    np.save(path, q_table)


def load_q_table(path):
    return np.load(path)


def save_training_metrics(rewards, path):
    ensure_dir(os.path.dirname(path))
    np.save(path, np.array(rewards))


def load_training_metrics(path):
    return np.load(path)
