"""
engine.py

Framework-free service layer between the Q-learning code and the web API.

It owns one "run" at a time (environment + agent + metric history), trains in
a background thread using the *unchanged* train() loop (through its hooks),
publishes JSON-ready snapshots to any number of subscribers, and exposes
step-by-step play so the UI can watch the learned policy move across the ice.

Nothing in here imports FastAPI, so it can be unit-tested and reused from a
CLI or a notebook.
"""

import json
import math
import threading
import time
from dataclasses import asdict, dataclass
from itertools import count

import numpy as np

import config as cfg
from agent import QLearningAgent
from environment import get_layout, make_env
from evaluate import evaluate
from train import train
from utils import set_seed

MAX_POINTS = 240          # chart points sent per series (downsampled)
EMIT_INTERVAL = 0.10      # seconds between live snapshots (~10 Hz)
EVAL_EPISODES = 1_000     # greedy evaluation after each completed run
ACTION_NAMES = ["Left", "Down", "Right", "Up"]
MOVES = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # (d_row, d_col) per action


class EpisodeFinished(Exception):
    """Raised when stepping an episode that already ended."""


@dataclass
class TrainParams:
    map_name: str = cfg.MAP_NAME
    slippery: bool = cfg.IS_SLIPPERY
    alpha: float = cfg.ALPHA
    gamma: float = cfg.GAMMA
    epsilon_start: float = cfg.EPSILON_START
    epsilon_min: float = cfg.EPSILON_MIN
    epsilon_decay: float = cfg.EPSILON_DECAY
    episodes: int = cfg.NUM_TRAINING_EPISODES
    seed: int = cfg.SEED
    pace: int = 3000      # max episodes/second so the run is watchable; 0 = full speed


class Run:
    """Everything belonging to one training run."""

    def __init__(self, run_id, params):
        self.run_id = run_id
        self.params = params
        self.status = "idle"
        self.error = None
        self.stop = threading.Event()
        self.thread = None

        env = make_env(map_name=params.map_name, is_slippery=params.slippery)
        self.layout = get_layout(env)
        self.size = len(self.layout)
        env.close()

        n_states = self.size * self.size
        self.agent = QLearningAgent(
            num_states=n_states, num_actions=4,
            alpha=params.alpha, gamma=params.gamma,
            epsilon_start=params.epsilon_start, epsilon_min=params.epsilon_min,
            epsilon_decay=params.epsilon_decay, rng=set_seed(params.seed),
        )

        total = max(int(params.episodes), 0)
        self.n = 0                                   # completed episodes
        self.rewards = np.zeros(total, dtype=np.float32)
        self.steps = np.zeros(total, dtype=np.float32)
        self.td = np.zeros(total, dtype=np.float32)
        self.epsilons = np.zeros(total, dtype=np.float32)
        self.visits = [0] * n_states
        self.last_path = []
        self.qconv = []                              # [episode, mean max-Q]
        self.evaluation = None
        self.t0 = None
        self.elapsed = 0.0


class _Play:
    def __init__(self, state):
        self.state = state
        self.steps = 0
        self.reward = 0.0
        self.done = False
        self.truncated = False


class Engine:
    def __init__(self, params=None):
        self.lock = threading.RLock()
        self.cond = threading.Condition()
        self.version = 0
        self.snapshot_json = "{}"
        self._ids = count(1)
        self._play = None
        self._play_env = None
        self.run = None
        self.reset(params or TrainParams())

    # ------------------------------------------------------------------ run
    def _stop_current(self):
        run = self.run
        if run is not None and run.thread is not None and run.thread.is_alive():
            run.stop.set()
            run.thread.join(timeout=10)

    def _install(self, params):
        run = Run(next(self._ids), params)
        self.run = run
        if self._play_env is not None:
            self._play_env.close()
        self._play_env = make_env(map_name=params.map_name, is_slippery=params.slippery)
        self._play = None
        return run

    def reset(self, params):
        """Discard any run and show a fresh, untrained agent."""
        with self.lock:
            self._stop_current()
            run = self._install(params)
            self._publish(run)

    def start_training(self, params):
        with self.lock:
            self._stop_current()
            run = self._install(params)
            run.status = "training"
            run.t0 = time.perf_counter()
            run.thread = threading.Thread(target=self._worker, args=(run,), daemon=True)
            run.thread.start()
            return run.run_id

    def stop_training(self):
        with self.lock:
            self._stop_current()

    def shutdown(self):
        """Non-blocking: ask any running job to stop (used when a session expires)."""
        run = self.run
        if run is not None:
            run.stop.set()

    @property
    def is_training(self):
        return self.run is not None and self.run.status == "training"

    # --------------------------------------------------------------- worker
    def _worker(self, run):
        p = run.params
        total = len(run.rewards)
        pace = max(int(p.pace), 0)
        chunk = max(1, pace // 100) if pace else 0
        last_emit = [0.0]
        t0 = run.t0
        env = make_env(map_name=p.map_name, is_slippery=p.slippery)

        def on_episode(ep, reward, steps, mean_td, path):
            run.rewards[ep] = reward
            run.steps[ep] = steps
            run.td[ep] = mean_td
            run.epsilons[ep] = run.agent.epsilon
            visits = run.visits
            for s in path:
                visits[s] += 1
            run.last_path = path
            run.n = ep + 1

            now = time.perf_counter()
            if chunk and run.n % chunk == 0:
                delay = t0 + run.n / pace - now
                if delay > 0:
                    time.sleep(delay)
                    now = time.perf_counter()
            if now - last_emit[0] >= EMIT_INTERVAL or run.n == total:
                last_emit[0] = now
                run.elapsed = now - t0
                self._publish(run)

        try:
            train(env, run.agent, total, seed=p.seed,
                  on_episode_end=on_episode, should_stop=run.stop.is_set)
            run.elapsed = time.perf_counter() - t0
            if run.stop.is_set():
                run.status = "stopped"
            else:
                run.status = "done"
                run.evaluation = self._evaluate(run, EVAL_EPISODES)
        except Exception as exc:  # surface failures to the UI instead of dying silently
            run.status = "error"
            run.error = f"{type(exc).__name__}: {exc}"
        finally:
            env.close()
            self._publish(run)

    @staticmethod
    def _evaluate(run, episodes):
        p = run.params
        env = make_env(map_name=p.map_name, is_slippery=p.slippery)
        env.reset(seed=p.seed + 1)
        try:
            return evaluate(env, run.agent, episodes)
        finally:
            env.close()

    def evaluate_policy(self, episodes=EVAL_EPISODES):
        with self.lock:
            run = self.run
            run.evaluation = self._evaluate(run, int(min(max(episodes, 1), 10_000)))
            self._publish(run)
            return run.evaluation

    # ------------------------------------------------------------ snapshots
    @staticmethod
    def _series(run):
        n = run.n
        if n == 0:
            return {k: [] for k in
                    ("episodes", "success_100", "success_500", "epsilon", "avg_steps", "mean_abs_td")}
        stride = max(1, math.ceil(n / MAX_POINTS))
        idx = np.arange(stride - 1, n, stride)
        if idx[-1] != n - 1:
            idx = np.append(idx, n - 1)

        def moving(values, window):
            cs = np.concatenate(([0.0], np.cumsum(values[:n], dtype=np.float64)))
            lo = np.maximum(0, idx + 1 - window)
            return np.round((cs[idx + 1] - cs[lo]) / (idx + 1 - lo), 4).tolist()

        return {
            "episodes": (idx + 1).tolist(),
            "success_100": moving(run.rewards, 100),
            "success_500": moving(run.rewards, 500),
            "epsilon": np.round(run.epsilons[idx], 4).tolist(),
            "avg_steps": moving(run.steps, 100),
            "mean_abs_td": [round(v, 5) for v in moving(run.td, 100)],
        }

    def _snapshot(self, run):
        q = run.agent.q_table
        n = run.n
        policy = [int(np.argmax(row)) if row.any() else -1 for row in q]
        series = self._series(run)
        elapsed = run.elapsed if run.status != "training" else time.perf_counter() - run.t0
        return {
            "run_id": run.run_id,
            "status": run.status,
            "error": run.error,
            "params": asdict(run.params),
            "layout": run.layout,
            "episode": n,
            "total_episodes": len(run.rewards),
            "epsilon": round(float(run.agent.epsilon), 5),
            "success_100": series["success_100"][-1] if n else 0.0,
            "avg_steps_100": series["avg_steps"][-1] if n else 0.0,
            "elapsed": round(elapsed, 2),
            "eps_per_sec": round(n / elapsed, 1) if elapsed > 0 else 0.0,
            "series": series,
            "qconv": run.qconv,
            "q_table": np.round(q, 5).tolist(),
            "policy": policy,
            "visits": run.visits,
            "last_path": run.last_path,
            "evaluation": run.evaluation,
        }

    def _publish(self, run):
        if run.n:
            run.qconv.append([run.n, round(float(run.agent.q_table.max(axis=1).mean()), 5)])
            if len(run.qconv) > 1_000:
                run.qconv = run.qconv[::2]
        payload = self._snapshot(run)
        with self.cond:
            self.version += 1
            payload["version"] = self.version
            self.snapshot_json = json.dumps(payload)
            self.cond.notify_all()

    def wait(self, last_version, timeout=10.0):
        """Block until a snapshot newer than last_version exists (or timeout)."""
        with self.cond:
            if self.version == last_version:
                self.cond.wait(timeout)
            return self.version, self.snapshot_json

    def current(self):
        with self.cond:
            return self.snapshot_json

    # ----------------------------------------------------------------- play
    def _view(self, **extra):
        ps, size, layout = self._play, self.run.size, self.run.layout
        row, col = divmod(ps.state, size)
        tile = layout[row][col]
        return {
            "state": ps.state, "row": row, "col": col, "tile": tile,
            "steps": ps.steps, "reward": ps.reward, "done": ps.done,
            "won": ps.done and tile == "G", "fell": ps.done and tile == "H",
            "truncated": ps.truncated, **extra,
        }

    def play_reset(self):
        with self.lock:
            state, _ = self._play_env.reset()
            self._play = _Play(int(state))
            return self._view(action=None, slipped=False)

    def play_step(self, action=None):
        """Advance one step. Uses the greedy action unless one is supplied."""
        with self.lock:
            if self._play is None:
                self.play_reset()
            ps = self._play
            if ps.done:
                raise EpisodeFinished("Episode finished - reset to play again.")

            size = self.run.size
            intended = int(action) if action is not None else int(np.argmax(self.run.agent.q_table[ps.state]))
            row, col = divmod(ps.state, size)
            dr, dc = MOVES[intended]
            expected = min(max(row + dr, 0), size - 1) * size + min(max(col + dc, 0), size - 1)

            nxt, reward, terminated, truncated, _ = self._play_env.step(intended)
            nxt = int(nxt)
            nrow, ncol = divmod(nxt, size)
            actual = MOVES.index((nrow - row, ncol - col)) if (nrow - row, ncol - col) in MOVES else None

            ps.state, ps.steps = nxt, ps.steps + 1
            ps.reward += float(reward)
            ps.done = bool(terminated or truncated)
            ps.truncated = bool(truncated and not terminated)
            return self._view(
                action=intended, action_name=ACTION_NAMES[intended],
                slipped=nxt != expected, actual_action=actual,
            )

    @staticmethod
    def defaults():
        return {
            "defaults": asdict(TrainParams()),
            "maps": ["4x4", "8x8"],
            "actions": ACTION_NAMES,
        }
