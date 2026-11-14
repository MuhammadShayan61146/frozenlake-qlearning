"""
train.py

Runs the Q-Learning training loop: for each episode, reset the environment,
let the agent act and learn from every step until the episode ends, decay
epsilon, and track the reward earned per episode for later analysis/plotting.

The loop is observable and cancellable through two optional hooks, so the same
code drives both the command-line pipeline and the live web dashboard:

    on_episode_end(episode, reward, steps, mean_abs_td, path)
    should_stop() -> bool
"""


def train(env, agent, num_episodes, seed=None, on_episode_end=None, should_stop=None):
    """
    Train the agent for num_episodes episodes.

    Args:
        seed: if given, seeds the environment on the first reset so that a
              run is fully reproducible (the agent's RNG is seeded separately).
        on_episode_end: optional callback fired after every episode with
              (episode_index, reward, steps, mean_abs_td_error, visited_states).
        should_stop: optional zero-arg callable; training ends early when it
              returns True (checked once per episode).

    Returns:
        reward_history: list of total reward earned in each episode
                         (for FrozenLake, this is 0.0 or 1.0 per episode,
                         since the only reward is +1 at the goal).
    """
    reward_history = []

    for episode in range(num_episodes):
        if should_stop is not None and should_stop():
            break

        state, info = env.reset(seed=seed if episode == 0 else None)
        episode_reward = 0.0
        terminated = False
        truncated = False
        steps = 0
        td_sum = 0.0
        path = [int(state)]

        while not (terminated or truncated):
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)

            td_error = agent.update(state, action, reward, next_state, terminated)

            state = next_state
            episode_reward += reward
            steps += 1
            td_sum += abs(td_error)
            path.append(int(state))

        agent.decay_epsilon()
        reward_history.append(episode_reward)

        if on_episode_end is not None:
            on_episode_end(episode, episode_reward, steps, td_sum / max(steps, 1), path)

    return reward_history
