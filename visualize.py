"""
visualize.py

Generates the three visualizations decided in our Step 3 design:
  1. Training reward / success-rate curve (learning progress over time)
  2. Final policy overlaid on the 4x4 grid (what the agent decided)
  3. Q-value heatmap (how confident/valuable each state is)

This file only plots data handed to it -- it has no knowledge of how
training or evaluation actually happened.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt

from config import TRAINING_CURVE_PLOT_PATH, POLICY_GRID_PLOT_PATH, Q_HEATMAP_PLOT_PATH
from utils import ensure_dir
import os

# FrozenLake action indices -> arrow symbols, for the policy grid
ACTION_ARROWS = {0: "\u2190", 1: "\u2193", 2: "\u2192", 3: "\u2191"}  # Left, Down, Right, Up


def _rolling_average(values, window=100):
    values = np.array(values, dtype=float)
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_training_curve(reward_history, window=100, save_path=TRAINING_CURVE_PLOT_PATH):
    """
    Plots the rolling-average reward (equivalently, rolling success rate,
    since FrozenLake rewards are 0 or 1) over training episodes.
    """
    ensure_dir(os.path.dirname(save_path))
    rolling = _rolling_average(reward_history, window)

    plt.figure(figsize=(8, 5))
    plt.plot(rolling)
    plt.xlabel(f"Episode (rolling window = {window})")
    plt.ylabel("Rolling average reward / success rate")
    plt.title("Training Progress: Rolling Success Rate over Episodes")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_policy_grid(q_table, save_path=POLICY_GRID_PLOT_PATH):
    """
    Draws the 4x4 grid with an arrow in each cell showing the agent's
    learned best action (greedy policy) from that state.
    """
    ensure_dir(os.path.dirname(save_path))
    GRID_SIZE = int(round(len(q_table) ** 0.5))
    best_actions = np.argmax(q_table, axis=1).reshape(GRID_SIZE, GRID_SIZE)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xticks(np.arange(GRID_SIZE + 1))
    ax.set_yticks(np.arange(GRID_SIZE + 1))
    ax.grid(True)
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.invert_yaxis()  # so row 0 is at the top, matching the grid layout

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            action = best_actions[row, col]
            arrow = ACTION_ARROWS[action]
            ax.text(col + 0.5, row + 0.5, arrow, ha="center", va="center", fontsize=20)

    ax.set_title("Learned Policy (best action per state)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_q_heatmap(q_table, save_path=Q_HEATMAP_PLOT_PATH):
    """
    Draws a heatmap of the maximum Q-value in each state, reshaped onto
    the 4x4 grid -- a visual sense of which tiles the agent considers
    valuable.
    """
    ensure_dir(os.path.dirname(save_path))
    GRID_SIZE = int(round(len(q_table) ** 0.5))
    max_q_values = np.max(q_table, axis=1).reshape(GRID_SIZE, GRID_SIZE)

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(max_q_values, cmap="viridis")
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            ax.text(col, row, f"{max_q_values[row, col]:.2f}",
                    ha="center", va="center", color="white", fontsize=9)

    ax.set_title("Max Q-Value per State")
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
