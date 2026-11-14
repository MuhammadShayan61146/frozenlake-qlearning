"""
evaluate.py

Runs evaluation episodes using the trained agent's GREEDY policy (no
exploration, no learning updates) to measure what was actually learned,
separate from behavior seen during training.
"""

import numpy as np


def evaluate(env, agent, num_episodes):
    """
    Evaluate the agent for num_episodes episodes using a purely greedy
    policy (epsilon effectively 0). Does NOT call agent.update() or
    agent.decay_epsilon() -- evaluation must not change the Q-table.

    Returns a dict with:
        success_rate: fraction of episodes that reached the goal
        average_reward: mean reward per episode (== success_rate here,
                         since reward is 0 or 1 -- noted explicitly in
                         the results discussion)
        average_episode_length: mean number of steps per episode
    """
    successes = 0
    total_reward = 0.0
    total_steps = 0

    for episode in range(num_episodes):
        state, info = env.reset()
        terminated = False
        truncated = False
        steps = 0

        while not (terminated or truncated):
            # Greedy action: always exploit, ignore agent.epsilon entirely.
            action = int(np.argmax(agent.q_table[state]))
            state, reward, terminated, truncated, info = env.step(action)
            steps += 1
            total_reward += reward

        if terminated and reward == 1.0:
            successes += 1
        total_steps += steps

    return {
        "success_rate": successes / num_episodes,
        "average_reward": total_reward / num_episodes,
        "average_episode_length": total_steps / num_episodes,
    }
