"""Unit tests for the learning code and the engine (no web framework needed)."""
import json
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent import QLearningAgent  # noqa: E402
from engine import Engine, EpisodeFinished, TrainParams  # noqa: E402
from environment import make_env  # noqa: E402
from train import train  # noqa: E402


def make_agent(seed=0, **kw):
    args = dict(num_states=16, num_actions=4, alpha=0.5, gamma=0.9, epsilon_start=1.0,
                epsilon_min=0.01, epsilon_decay=0.99, rng=np.random.default_rng(seed))
    args.update(kw)
    return QLearningAgent(**args)


def wait_until(engine, predicate, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        snap = json.loads(engine.current())
        if predicate(snap):
            return snap
        time.sleep(0.05)
    raise AssertionError("timed out waiting for engine")


def test_update_returns_td_error_and_moves_q_toward_target():
    a = make_agent()
    td = a.update(0, 1, 1.0, 1, terminated=False)
    assert td == pytest.approx(1.0)
    assert a.q_table[0, 1] == pytest.approx(0.5)


def test_terminal_transitions_do_not_bootstrap():
    a = make_agent()
    a.q_table[5] = [9, 9, 9, 9]
    a.update(0, 0, 0.0, 5, terminated=True)
    assert a.q_table[0, 0] == 0.0


def test_actions_are_plain_ints():
    a = make_agent()
    assert all(isinstance(a.choose_action(0), int) for _ in range(50))


def test_training_is_reproducible_with_a_seed():
    def run():
        env = make_env(is_slippery=True)
        agent = make_agent(seed=7)
        rewards = train(env, agent, 300, seed=7)
        return rewards, agent.q_table.copy()

    (r1, q1), (r2, q2) = run(), run()
    assert r1 == r2 and np.array_equal(q1, q2)


def test_train_hooks_fire_and_can_cancel():
    env, agent, seen = make_env(), make_agent(), []
    train(env, agent, 1000, on_episode_end=lambda ep, *_: seen.append(ep),
          should_stop=lambda: len(seen) >= 10)
    assert seen == list(range(10))


def test_engine_run_publishes_complete_snapshot():
    eng = Engine()
    eng.start_training(TrainParams(episodes=1500, pace=0))
    snap = wait_until(eng, lambda s: s["status"] == "done")
    assert snap["episode"] == snap["total_episodes"] == 1500
    assert len(snap["q_table"]) == 16 and len(snap["policy"]) == 16
    assert 0.0 <= snap["evaluation"]["success_rate"] <= 1.0
    n = len(snap["series"]["episodes"])
    assert n > 0 and all(len(snap["series"][k]) == n for k in ("success_100", "success_500", "epsilon"))
    assert snap["epsilon"] < 1.0


def test_stop_ends_run_early():
    eng = Engine()
    eng.start_training(TrainParams(episodes=100_000, pace=500))
    wait_until(eng, lambda s: s["episode"] > 50)
    eng.stop_training()
    snap = wait_until(eng, lambda s: s["status"] == "stopped")
    assert snap["episode"] < 100_000


def test_restart_replaces_the_previous_run():
    eng = Engine()
    first = eng.start_training(TrainParams(episodes=100_000, pace=500))
    second = eng.start_training(TrainParams(episodes=200, pace=0))
    assert second > first
    snap = wait_until(eng, lambda s: s["status"] == "done")
    assert snap["run_id"] == second and snap["episode"] == 200


def test_play_steps_until_done_then_refuses():
    eng = Engine(TrainParams(slippery=False))
    eng.play_reset()
    for _ in range(300):
        try:
            v = eng.play_step(action=2)  # keep walking right
        except EpisodeFinished:
            break
        assert 0 <= v["state"] < 16
        if v["done"]:
            with pytest.raises(EpisodeFinished):
                eng.play_step()
            break


def test_no_slip_flag_on_non_slippery_ice():
    eng = Engine(TrainParams(slippery=False))
    eng.play_reset()
    assert eng.play_step(action=2)["slipped"] is False


def test_changing_map_rebuilds_the_board():
    eng = Engine()
    eng.reset(TrainParams(map_name="8x8"))
    snap = json.loads(eng.current())
    assert len(snap["layout"]) == 8 and len(snap["q_table"]) == 64
