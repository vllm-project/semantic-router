import assert from "node:assert/strict";
import { setImmediate as tick } from "node:timers/promises";
import { test } from "node:test";
import { DecisionStore } from "../src/decisions.ts";
import { pricesOf } from "../src/prices.ts";
import type { Reply, Result } from "../src/runtime.ts";
import { parseSettings } from "../src/settings.ts";
import { estimateOf, Turns } from "../src/turns.ts";
import type { Price } from "../src/turns.ts";
import { savingWords, summary } from "../src/view.ts";

const STRONG: Price = { input: 10, output: 50, cacheRead: 1, cacheWrite: 12.5 };
const LIGHT: Price = { input: 1, output: 5, cacheRead: 0.1, cacheWrite: 1.25 };
const prices = new Map([
  ["p/strong", STRONG],
  ["p/light", LIGHT],
]);
const million = { input: 1e6, output: 1e6, cacheRead: 0, cacheWrite: 0 };

test("an estimate prices the same tokens on the model that ran them and on the session's", () => {
  assert.deepEqual(estimateOf(prices, "p/light", "p/strong", million), {
    actual: 6,
    baseline: 60,
    on: "p/strong",
  });
  assert.equal(
    estimateOf(prices, "p/medium", "p/strong", million),
    undefined,
    "a model with no known price gives no estimate",
  );
});

test("the price table keeps only models with a known price", () => {
  const table = pricesOf([
    {
      providerID: "p",
      id: "a",
      cost: [{ input: 1, output: 2, cache: { read: 0.1, write: 1.25 } }],
    },
    { providerID: "p", id: "b", cost: [{ input: 3, output: 4 }] },
    { providerID: "p", id: "c" },
    { providerID: "p", id: "d", cost: [] },
  ]);
  assert.deepEqual([...table.keys()], ["p/a", "p/b"]);
  assert.deepEqual(table.get("p/b"), {
    input: 3,
    output: 4,
    cacheRead: 0,
    cacheWrite: 0,
  });
});

test("your turns moved to a lighter tier are priced there and on your own model, once per turn", async () => {
  const settings = parseSettings({
    routing: {
      turns: true,
      goal: "save",
      tiers: ["light", "medium", "strong"].map((id) => ({ model: `p/${id}` })),
    },
  });
  const easy: Result<Reply> = {
    ok: true,
    elapsedMs: 1,
    value: {
      answers: {
        complexity: { score: 0.1, confidence: 0.95 },
        steps: { score: 0.1, confidence: 0.95 },
      },
      spans: {},
    },
  };
  const turns = new Turns(settings, { decide: async () => easy }, () => "1", {
    modelOf: async () => ({ providerID: "p", id: "medium" }),
    switchModel: async () => {},
    // A move is one tier at a time: medium, priced as strong here, to light.
    prices: async () => new Map([...prices, ["p/medium", STRONG]]),
  });
  const long = "EASY " + "please list the files ".repeat(5);
  await turns.prompt("s", "m1", long, async () => false);
  assert.equal(turns.view("s").model, "p/light");
  turns.stepped("s", { input: 1e6, output: 0, cacheRead: 0, cacheWrite: 0 });
  // Reasoning is priced as output.
  turns.stepped(
    "s",
    { input: 0, output: 5e5, cacheRead: 0, cacheWrite: 0 },
    5e5,
  );
  await tick();
  assert.deepEqual(turns.view("s").spend, {
    actual: 6,
    baseline: 60,
    turns: 1,
    on: "p/medium",
  });
  await turns.prompt("s", "m2", long, async () => false);
  turns.stepped("s", { input: 1e6, output: 0, cacheRead: 0, cacheWrite: 0 });
  await tick();
  assert.equal(turns.view("s").spend?.turns, 2);
  assert.equal(turns.view("s").spend?.actual, 7);

  // Back on your own model, nothing is counted; removing the session forgets it.
  settings.routing.turns = false;
  await turns.prompt("s", "m3", long, async () => false);
  turns.stepped("s", { input: 1e6, output: 0, cacheRead: 0, cacheWrite: 0 });
  await tick();
  assert.equal(turns.view("s").spend?.turns, 2);
  turns.remove("s");
  assert.equal(turns.view("s").spend, undefined);
});

test("the routing row adds routed subagents and turns against the session's model", () => {
  const store = new DecisionStore(4, 50);
  const routed = store.record({
    sessionID: "s",
    app: "router",
    outcome: "routed",
    model: "p/light",
  });
  store.observe("s", routed.id, {
    model: "p/light",
    childSessionID: "c1",
    usage: { input: 1, output: 1, cachedInput: 0, reasoning: 0, cost: 0.15 },
    costKnown: true,
    estimate: { actual: 0.15, baseline: 3, on: "p/strong" },
  });
  assert.throws(() =>
    store.observe("s", routed.id, {
      estimate: { actual: -1, baseline: 3, on: "p/strong" },
    }),
  );
  const { routed: subagents } = summary(store.snapshot("s").decisions);
  assert.equal(
    savingWords(subagents, undefined),
    "routing 1 subagent ~$0.1500 vs ~$3.00 on strong · saved 95%",
  );
  assert.equal(
    savingWords(subagents, {
      actual: 0.85,
      baseline: 3,
      turns: 2,
      on: "p/strong",
    }),
    "routing 1 subagent + 2 turns ~$1.00 vs ~$6.00 on strong · saved 83%",
  );
  assert.equal(
    savingWords(subagents, { actual: 0.5, baseline: 1, turns: 1, on: "q/big" }),
    "routing 1 subagent + 1 turn ~$0.6500 vs ~$4.00 on your models · saved 84%",
  );
  assert.equal(
    savingWords({ count: 0, actual: 0, baseline: 0, on: new Set() }, undefined),
    undefined,
    "nothing routed and priced: no row",
  );
  // The row from a session on Luna whose turn moved to a dearer tier, and one a little dearer.
  assert.equal(
    savingWords(
      { count: 0, actual: 0, baseline: 0, on: new Set() },
      {
        actual: 0.0123,
        baseline: 0.0007,
        turns: 1,
        on: "azure-lab/gpt-5.6-luna",
      },
    ),
    "routing 1 turn ~$0.0123 vs ~$0.0007 on gpt-5.6-luna · 18x more",
  );
  assert.equal(
    savingWords(
      { count: 0, actual: 0, baseline: 0, on: new Set() },
      { actual: 1.35, baseline: 1, turns: 2, on: "p/strong" },
    ),
    "routing 2 turns ~$1.35 vs ~$1.00 on strong · 35% more",
  );
});
