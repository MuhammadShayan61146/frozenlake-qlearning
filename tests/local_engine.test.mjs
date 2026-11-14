// Run with: node --test tests/
import test from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const Lab = createRequire(import.meta.url)("../frontend/static/local-engine.js");
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

function listen() {
  let last = null;
  const off = Lab.subscribe((s) => { last = s; });
  return { get: () => last, off };
}
async function until(l, pred, ms = 20000) {
  const end = Date.now() + ms;
  while (Date.now() < end) { if (pred(l.get())) return l.get(); await wait(10); }
  throw new Error("timeout");
}

test("trains to a sensible policy and publishes a full snapshot", async () => {
  const l = listen();
  Lab.handle("/api/train", { pace: 0, episodes: 15000 });
  const s = await until(l, (x) => x.status === "done");
  assert.equal(s.episode, 15000);
  assert.equal(s.q_table.length, 16);
  assert.ok(s.evaluation.success_rate > 0.6, `greedy success ${s.evaluation.success_rate}`);
  const n = s.series.episodes.length;
  assert.ok(n > 0 && n <= 241);
  for (const k of ["success_100", "success_500", "epsilon", "avg_steps", "mean_abs_td"]) assert.equal(s.series[k].length, n);
  l.off();
});

test("same seed gives identical results", async () => {
  const runOnce = async () => {
    const l = listen();
    Lab.handle("/api/train", { pace: 0, episodes: 2000, seed: 7 });
    const s = await until(l, (x) => x.status === "done");
    l.off();
    return JSON.stringify([s.q_table, s.evaluation]);
  };
  assert.equal(await runOnce(), await runOnce());
});

test("stop ends a paced run early", async () => {
  const l = listen();
  Lab.handle("/api/train", { pace: 200, episodes: 100000 });
  await until(l, (x) => x.episode > 20);
  Lab.handle("/api/stop");
  const s = await until(l, (x) => x.status === "stopped");
  assert.ok(s.episode < 100000);
  l.off();
});

test("8x8 map rebuilds the board; invalid input is rejected", () => {
  const l = listen();
  Lab.handle("/api/reset", { map_name: "8x8" });
  assert.equal(l.get().layout.length, 8);
  assert.equal(l.get().q_table.length, 64);
  assert.throws(() => Lab.handle("/api/train", { alpha: 5 }), /alpha/);
  l.off();
});

test("play: non-slippery walk is deterministic and episodes can end", () => {
  Lab.handle("/api/reset", { slippery: false });
  Lab.handle("/api/play/reset");
  const v = Lab.handle("/api/play/step", { action: 2 });
  assert.equal(v.state, 1);
  assert.equal(v.slipped, false);
  let done = v.done, guard = 0;
  while (!done && guard++ < 300) done = Lab.handle("/api/play/step", { action: 2 }).done;
  assert.ok(done);
  assert.throws(() => Lab.handle("/api/play/step", {}), /finished/);
});
