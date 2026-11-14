/* Frozen Lake Lab: in-browser engine.
   A JavaScript port of agent.py, train.py, evaluate.py, environment.py and engine.py.
   It exposes the same commands and the same snapshot shape as the FastAPI backend,
   so the dashboard runs unchanged with no server (for example on GitHub Pages).

   Random numbers come from a seeded mulberry32 generator, so runs are reproducible
   but not bit-identical to NumPy's generator in the Python version. */
(function (root) {
  "use strict";

  const MAPS = {
    "4x4": ["SFFF", "FHFH", "FFFH", "HFFG"],
    "8x8": ["SFFFFFFF", "FFFFFFFF", "FFFHFFFF", "FFFFFHFF", "FFFHFFFF", "FHHFFFHF", "FHFFHFHF", "FFFHFFFG"],
  };
  const MAX_STEPS = { "4x4": 100, "8x8": 200 };
  const MOVES = [[0, -1], [1, 0], [0, 1], [-1, 0]]; // Left, Down, Right, Up as [d_row, d_col]
  const ACTION_NAMES = ["Left", "Down", "Right", "Up"];
  const MAX_POINTS = 240;      // chart points per series
  const EMIT_MS = 100;         // live snapshot interval
  const BUDGET_MS = 12;        // max work per tick so the page stays responsive
  const EVAL_EPISODES = 1000;

  const DEFAULTS = {
    map_name: "4x4", slippery: true, alpha: 0.1, gamma: 0.99, epsilon_start: 1.0,
    epsilon_min: 0.01, epsilon_decay: 0.9995, episodes: 15000, seed: 42, pace: 3000,
  };

  const r4 = (v) => Math.round(v * 1e4) / 1e4;
  const r5 = (v) => Math.round(v * 1e5) / 1e5;

  function rngFrom(seed) {                       // mulberry32
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /* ---------------------------------------------------------- environment */
  class FrozenLake {
    constructor(mapName, slippery, rng) {
      this.layout = MAPS[mapName];
      this.n = this.layout.length;
      this.flat = this.layout.join("");
      this.slippery = slippery;
      this.maxSteps = MAX_STEPS[mapName];
      this.rng = rng;
      this.s = 0;
      this.t = 0;
    }
    reset() { this.s = 0; this.t = 0; return 0; }
    step(action) {
      let a = action;
      if (this.slippery) a = [(action + 3) % 4, action, (action + 1) % 4][Math.floor(this.rng() * 3)];
      const n = this.n, r = Math.floor(this.s / n), c = this.s % n;
      const nr = Math.min(Math.max(r + MOVES[a][0], 0), n - 1);
      const nc = Math.min(Math.max(c + MOVES[a][1], 0), n - 1);
      this.s = nr * n + nc;
      this.t += 1;
      const tile = this.flat[this.s];
      return {
        state: this.s,
        reward: tile === "G" ? 1 : 0,
        terminated: tile === "G" || tile === "H",
        truncated: this.t >= this.maxSteps,
      };
    }
  }

  /* ---------------------------------------------------------------- agent */
  class Agent {
    constructor(nStates, p, rng) {
      this.q = new Float64Array(nStates * 4);
      this.alpha = p.alpha; this.gamma = p.gamma;
      this.epsilon = p.epsilon_start; this.epsilonMin = p.epsilon_min; this.epsilonDecay = p.epsilon_decay;
      this.rng = rng;
    }
    greedy(s) {                                   // first maximum wins, like numpy.argmax
      const q = this.q, o = s * 4;
      let best = 0, v = q[o];
      for (let a = 1; a < 4; a++) if (q[o + a] > v) { v = q[o + a]; best = a; }
      return best;
    }
    act(s) { return this.rng() < this.epsilon ? Math.floor(this.rng() * 4) : this.greedy(s); }
    update(s, a, reward, s2, terminated) {
      // Terminal transitions must not bootstrap: there is no next action beyond the goal or a hole.
      let bestNext = 0;
      if (!terminated) {
        const o = s2 * 4;
        bestNext = Math.max(this.q[o], this.q[o + 1], this.q[o + 2], this.q[o + 3]);
      }
      const td = reward + this.gamma * bestNext - this.q[s * 4 + a];
      this.q[s * 4 + a] += this.alpha * td;
      return td;
    }
    decay() { this.epsilon = Math.max(this.epsilonMin, this.epsilon * this.epsilonDecay); }
    anyLearned(s) { const o = s * 4; return this.q[o] !== 0 || this.q[o + 1] !== 0 || this.q[o + 2] !== 0 || this.q[o + 3] !== 0; }
  }

  /* ------------------------------------------------------------------ run */
  function makeRun(id, p) {
    const layout = MAPS[p.map_name], size = layout.length, nStates = size * size;
    const total = Math.max(0, Math.floor(p.episodes));
    return {
      id, params: p, status: "idle", error: null, layout, size,
      agent: new Agent(nStates, p, rngFrom(p.seed * 2)),
      env: new FrozenLake(p.map_name, p.slippery, rngFrom(p.seed * 2 + 1)),
      n: 0, total,
      rewards: new Float32Array(total), steps: new Float32Array(total),
      td: new Float32Array(total), eps: new Float32Array(total),
      visits: new Array(nStates).fill(0), lastPath: [], qconv: [],
      evaluation: null, t0: 0, elapsed: 0, lastEmit: 0,
    };
  }

  function runEpisode(run) {
    const { env, agent } = run;
    let s = env.reset(), reward = 0, steps = 0, tdSum = 0, done = false;
    const path = [s];
    while (!done) {
      const a = agent.act(s);
      const r = env.step(a);
      tdSum += Math.abs(agent.update(s, a, r.reward, r.state, r.terminated));
      s = r.state;
      reward += r.reward;
      steps += 1;
      path.push(s);
      done = r.terminated || r.truncated;
    }
    agent.decay();
    const i = run.n;
    run.rewards[i] = reward; run.steps[i] = steps; run.td[i] = tdSum / Math.max(steps, 1); run.eps[i] = agent.epsilon;
    for (const st of path) run.visits[st] += 1;
    run.lastPath = path;
    run.n = i + 1;
  }

  function evaluate(run, episodes) {
    const p = run.params;
    const env = new FrozenLake(p.map_name, p.slippery, rngFrom(p.seed + 1000003));
    let successes = 0, totalReward = 0, totalSteps = 0;
    for (let e = 0; e < episodes; e++) {
      let s = env.reset(), done = false, last = null, steps = 0;
      while (!done) {
        last = env.step(run.agent.greedy(s));
        s = last.state; steps += 1; totalReward += last.reward;
        done = last.terminated || last.truncated;
      }
      if (last.terminated && last.reward === 1) successes += 1;
      totalSteps += steps;
    }
    return { success_rate: successes / episodes, average_reward: totalReward / episodes, average_episode_length: totalSteps / episodes };
  }

  /* ------------------------------------------------------------ snapshots */
  function series(run) {
    const n = run.n;
    if (n === 0) return { episodes: [], success_100: [], success_500: [], epsilon: [], avg_steps: [], mean_abs_td: [] };
    const stride = Math.max(1, Math.ceil(n / MAX_POINTS));
    const idx = [];
    for (let i = stride - 1; i < n; i += stride) idx.push(i);
    if (idx[idx.length - 1] !== n - 1) idx.push(n - 1);
    const cum = (arr) => { const c = new Float64Array(n + 1); for (let i = 0; i < n; i++) c[i + 1] = c[i] + arr[i]; return c; };
    const moving = (c, w) => idx.map((i) => { const lo = Math.max(0, i + 1 - w); return (c[i + 1] - c[lo]) / (i + 1 - lo); });
    const cr = cum(run.rewards), cs = cum(run.steps), ct = cum(run.td);
    return {
      episodes: idx.map((i) => i + 1),
      success_100: moving(cr, 100).map(r4),
      success_500: moving(cr, 500).map(r4),
      epsilon: idx.map((i) => r4(run.eps[i])),
      avg_steps: moving(cs, 100).map(r4),
      mean_abs_td: moving(ct, 100).map(r5),
    };
  }

  function snapshot(run) {
    const nStates = run.size * run.size, q = run.agent.q;
    const se = series(run), n = run.n;
    const training = run.status === "training";
    const elapsed = training ? (performance.now() - run.t0) / 1000 : run.elapsed;
    const qTable = [], policy = [];
    for (let s = 0; s < nStates; s++) {
      qTable.push([r5(q[s * 4]), r5(q[s * 4 + 1]), r5(q[s * 4 + 2]), r5(q[s * 4 + 3])]);
      policy.push(run.agent.anyLearned(s) ? run.agent.greedy(s) : -1);
    }
    return {
      run_id: run.id, status: run.status, error: run.error, params: { ...run.params },
      layout: run.layout, episode: n, total_episodes: run.total, epsilon: r5(run.agent.epsilon),
      success_100: n ? se.success_100[se.success_100.length - 1] : 0,
      avg_steps_100: n ? se.avg_steps[se.avg_steps.length - 1] : 0,
      elapsed: Math.round(elapsed * 100) / 100,
      eps_per_sec: elapsed > 0 ? Math.round((n / elapsed) * 10) / 10 : 0,
      series: se, qconv: run.qconv.slice(), q_table: qTable, policy,
      visits: run.visits.slice(), last_path: run.lastPath.slice(), evaluation: run.evaluation,
    };
  }

  /* ------------------------------------------------------------- the lab */
  let run = null, playEnv = null, play = null, runCounter = 0;
  const subscribers = [];

  function publish(r) {
    if (r.n) {
      const q = r.agent.q, nStates = r.size * r.size;
      let sum = 0;
      for (let s = 0; s < nStates; s++) sum += Math.max(q[s * 4], q[s * 4 + 1], q[s * 4 + 2], q[s * 4 + 3]);
      r.qconv.push([r.n, r5(sum / nStates)]);
      if (r.qconv.length > 1000) r.qconv = r.qconv.filter((_, i) => i % 2 === 0);
    }
    const snap = snapshot(r);
    for (const fn of subscribers) fn(snap);
  }

  function finish(r, status) {
    r.elapsed = (performance.now() - r.t0) / 1000;
    r.status = status;
    if (status === "done") r.evaluation = evaluate(r, EVAL_EPISODES);
    publish(r);
  }

  function tick(r) {
    if (r !== run || r.status !== "training") return;
    const start = performance.now(), p = r.params;
    const goal = p.pace > 0 ? Math.min(r.total, Math.floor(((start - r.t0) / 1000) * p.pace)) : r.total;
    while (r.n < goal && performance.now() - start < BUDGET_MS) runEpisode(r);
    const now = performance.now();
    r.elapsed = (now - r.t0) / 1000;
    if (r.n >= r.total) { finish(r, "done"); return; }
    if (now - r.lastEmit >= EMIT_MS) { r.lastEmit = now; publish(r); }
    setTimeout(() => tick(r), p.pace > 0 ? 16 : 0);
  }

  function parseParams(body) {
    const p = { ...DEFAULTS, ...body };
    const bad = (m) => { const e = new Error(m); e.status = 422; throw e; };
    if (!MAPS[p.map_name]) bad("map_name: must be 4x4 or 8x8");
    const num = (k, lo, hi, loOpen) => {
      const v = Number(p[k]);
      if (!Number.isFinite(v) || v > hi || (loOpen ? v <= lo : v < lo)) bad(`${k}: must be between ${lo} and ${hi}`);
      p[k] = v;
    };
    num("alpha", 0, 1, true); num("gamma", 0, 1); num("epsilon_start", 0, 1); num("epsilon_min", 0, 1);
    num("epsilon_decay", 0, 1, true); num("episodes", 1, 100000); num("seed", 0, 2147483647); num("pace", 0, 1e6);
    p.episodes = Math.floor(p.episodes); p.seed = Math.floor(p.seed); p.slippery = !!p.slippery;
    return p;
  }

  function install(p) {
    run = makeRun(++runCounter, p);
    playEnv = new FrozenLake(p.map_name, p.slippery, rngFrom(p.seed * 2 + 7));
    play = null;
    return run;
  }

  function stopCurrent() { if (run && run.status === "training") run.status = "stopped"; }

  /* ----------------------------------------------------------------- play */
  function view(extra) {
    const n = run.size, s = play.state, row = Math.floor(s / n), col = s % n, tile = run.layout[row][col];
    return {
      state: s, row, col, tile, steps: play.steps, reward: play.reward, done: play.done,
      won: play.done && tile === "G", fell: play.done && tile === "H", truncated: play.truncated, ...extra,
    };
  }

  function playReset() {
    play = { state: playEnv.reset(), steps: 0, reward: 0, done: false, truncated: false };
    return view({ action: null, slipped: false });
  }

  function playStep(action) {
    if (!play) playReset();
    if (play.done) { const e = new Error("Episode finished - reset to play again."); e.status = 409; throw e; }
    const n = run.size, ps = play;
    const intended = action != null ? action : run.agent.greedy(ps.state);
    const row = Math.floor(ps.state / n), col = ps.state % n, [dr, dc] = MOVES[intended];
    const expected = Math.min(Math.max(row + dr, 0), n - 1) * n + Math.min(Math.max(col + dc, 0), n - 1);
    const res = playEnv.step(intended);
    const nr = Math.floor(res.state / n), nc = res.state % n;
    const actual = MOVES.findIndex((m) => m[0] === nr - row && m[1] === nc - col);
    ps.state = res.state; ps.steps += 1; ps.reward += res.reward;
    ps.done = res.terminated || res.truncated;
    ps.truncated = res.truncated && !res.terminated;
    return view({ action: intended, action_name: ACTION_NAMES[intended], slipped: res.state !== expected, actual_action: actual < 0 ? null : actual });
  }

  /* ------------------------------------------------------------ public API */
  function handle(path, body) {
    body = body || {};
    switch (path) {
      case "/api/train": {
        const p = parseParams(body);
        stopCurrent();
        const r = install(p);
        r.status = "training"; r.t0 = performance.now();
        publish(r);
        setTimeout(() => tick(r), 0);
        return { run_id: r.id, status: "training" };
      }
      case "/api/stop":
        if (run.status === "training") finish(run, "stopped");
        return { status: "stopped" };
      case "/api/reset": {
        stopCurrent();
        publish(install(parseParams(body)));
        return { status: "idle" };
      }
      case "/api/play/reset": return playReset();
      case "/api/play/step": return playStep(body.action);
      case "/api/evaluate": {
        run.evaluation = evaluate(run, Math.min(Math.max(body.episodes || EVAL_EPISODES, 1), 10000));
        publish(run);
        return run.evaluation;
      }
      default: { const e = new Error("Unknown endpoint " + path); e.status = 404; throw e; }
    }
  }

  function subscribe(fn) {
    subscribers.push(fn);
    fn(snapshot(run));
    return () => subscribers.splice(subscribers.indexOf(fn), 1);
  }

  install({ ...DEFAULTS });
  const Lab = { handle, subscribe, defaults: DEFAULTS };
  if (typeof module !== "undefined" && module.exports) module.exports = Lab;
  else root.LocalLab = Lab;
})(typeof window !== "undefined" ? window : globalThis);
