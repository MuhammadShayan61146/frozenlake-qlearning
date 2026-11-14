"""
config.py

Single source of truth for all hyperparameters and settings used across
the FrozenLake Tabular Q-Learning project. No other file should hardcode
these values -- they should import them from here.
"""

import os

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------
MAP_NAME = "4x4"
IS_SLIPPERY = True
MAX_STEPS_PER_EPISODE = 100  # Gymnasium's own default for FrozenLake-v1 (4x4)

# ---------------------------------------------------------------------------
# Q-Learning hyperparameters
# ---------------------------------------------------------------------------
ALPHA = 0.1          # learning rate: how much each update shifts the Q-value
GAMMA = 0.99          # discount factor: importance of future reward

# ---------------------------------------------------------------------------
# Exploration (epsilon-greedy) schedule
# ---------------------------------------------------------------------------
EPSILON_START = 1.0   # fully random at the start of training
EPSILON_MIN = 0.01     # small residual exploration kept even late in training
EPSILON_DECAY = 0.9995  # multiplicative decay applied once per episode

# ---------------------------------------------------------------------------
# Training / evaluation scale
# ---------------------------------------------------------------------------
NUM_TRAINING_EPISODES = 15_000
NUM_EVAL_EPISODES = 1_000

# ---------------------------------------------------------------------------
# Output paths (matches the results/ structure from Step 4)
# ---------------------------------------------------------------------------
RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

Q_TABLE_PATH = os.path.join(RESULTS_DIR, "q_table.npy")
TRAINING_METRICS_PATH = os.path.join(RESULTS_DIR, "training_metrics.npy")

TRAINING_CURVE_PLOT_PATH = os.path.join(PLOTS_DIR, "training_curve.png")
POLICY_GRID_PLOT_PATH = os.path.join(PLOTS_DIR, "policy_grid.png")
Q_HEATMAP_PLOT_PATH = os.path.join(PLOTS_DIR, "q_heatmap.png")
