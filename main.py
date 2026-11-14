"""
main.py

Orchestrates the complete pipeline end-to-end:
  seed -> build environment -> build agent -> train -> save -> evaluate
  -> visualize -> print summary
"""

from config import (
    SEED, ALPHA, GAMMA, EPSILON_START, EPSILON_MIN, EPSILON_DECAY,
    NUM_TRAINING_EPISODES, NUM_EVAL_EPISODES,
    Q_TABLE_PATH, TRAINING_METRICS_PATH,
)
from environment import make_env
from agent import QLearningAgent
from utils import set_seed, save_q_table, save_training_metrics
from train import train
from evaluate import evaluate
from visualize import plot_training_curve, plot_policy_grid, plot_q_heatmap


def main():
    # 1. Reproducibility
    rng = set_seed(SEED)

    # 2. Environment
    env = make_env()
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # 3. Agent
    agent = QLearningAgent(
        num_states=num_states,
        num_actions=num_actions,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
        rng=rng,
    )

    # 4. Train
    print(f"Training for {NUM_TRAINING_EPISODES} episodes...")
    reward_history = train(env, agent, NUM_TRAINING_EPISODES, seed=SEED)
    env.close()

    # 5. Save Q-table and metrics
    save_q_table(agent.q_table, Q_TABLE_PATH)
    save_training_metrics(reward_history, TRAINING_METRICS_PATH)
    print(f"Saved trained Q-table to {Q_TABLE_PATH}")

    # 6. Evaluate (fresh env instance, same config, greedy policy)
    eval_env = make_env()
    eval_env.reset(seed=SEED + 1)  # different seed stream than training
    results = evaluate(eval_env, agent, NUM_EVAL_EPISODES)
    eval_env.close()

    # 7. Visualize
    plot_training_curve(reward_history)
    plot_policy_grid(agent.q_table)
    plot_q_heatmap(agent.q_table)

    # 8. Summary
    print("\n--- Evaluation Results (greedy policy, no exploration) ---")
    print(f"Success rate:          {results['success_rate']:.3f}")
    print(f"Average reward:        {results['average_reward']:.3f}")
    print(f"Average episode length:{results['average_episode_length']:.2f}")
    print(f"Final epsilon after training: {agent.epsilon:.4f}")


if __name__ == "__main__":
    main()
