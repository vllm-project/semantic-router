import assert from "node:assert/strict";
import { test } from "node:test";
import { SessionArchive } from "../src/archive.ts";
import { DecisionStore } from "../src/decisions.ts";
import type { LogStore } from "../src/log.ts";

function memory(): LogStore & { data: Map<string, unknown> } {
  const data = new Map<string, unknown>();
  return {
    data,
    get: async (key) => structuredClone(data.get(key)),
    set: async (key, value) => void data.set(key, structuredClone(value)),
    remove: async (key) => void data.delete(key),
    scan: async ({ prefix }) => ({
      entries: [...data]
        .filter(([key]) => key.startsWith(prefix))
        .map(([key, value]) => ({ key, value: structuredClone(value) })),
    }),
  };
}

test("a restarted server shows the decisions, outcomes, usage and feedback it had", async () => {
  const store = memory();
  const before = new DecisionStore(4, 10);
  const archive = new SessionArchive(store, before);
  await archive.load();
  const held = before.record({
    sessionID: "s",
    app: "guard",
    outcome: "held",
    score: 0.95,
    subject: "rm -rf build",
    thresholds: [0.9],
  });
  before.settleGuard("s", held.id, "approved");
  const routed = before.record({
    sessionID: "s",
    app: "router",
    outcome: "routed",
    model: "p/light",
  });
  before.observe("s", routed.id, {
    model: "p/light#default",
    childSessionID: "child",
    usage: { input: 10, output: 2, cachedInput: 0, reasoning: 0, cost: 0 },
    costKnown: false,
    childOutcome: "succeeded",
  });
  before.label("s", routed.id, "right-model");
  await archive.close();

  const after = new DecisionStore(4, 10);
  let logged = 0;
  after.sink = {
    recorded: () => logged++,
    labeled: () => logged++,
    settled: () => logged++,
  };
  await new SessionArchive(store, after).load();
  const { decisions, feedback } = after.snapshot("s");
  assert.deepEqual(decisions, before.snapshot("s").decisions);
  assert.equal(decisions[0]?.outcome, "approved");
  assert.deepEqual(decisions[0]?.thresholds, [0.9]);
  assert.equal(decisions[1]?.observedModel, "p/light#default");
  assert.equal(decisions[1]?.usage?.input, 10);
  assert.deepEqual(feedback, { [routed.id]: "right-model" });
  assert.equal(logged, 0, "restored decisions are not logged again");
});

test("the archive drops what fails the checks and follows the store's limits", async () => {
  const store = memory();
  store.data.set("sessions/bad", {
    at: 1,
    decisions: [
      { id: "x", at: 1, sessionID: "bad", app: "nope", outcome: "allowed" },
      { id: "y", at: 1, sessionID: "other", app: "guard", outcome: "allowed" },
      {
        id: "z",
        at: 1,
        sessionID: "bad",
        app: "guard",
        outcome: "allowed",
        raw: "output text",
      },
    ],
    feedback: { x: "wrong", z: "nonsense" },
  });
  for (const [i, id] of ["a", "b", "c"].entries())
    store.data.set(`sessions/${id}`, {
      at: 10 + i,
      decisions: [
        {
          id: `d${id}`,
          at: 10,
          sessionID: id,
          app: "check",
          outcome: "verified",
        },
      ],
      feedback: {},
    });
  const decisions = new DecisionStore(2, 10);
  const archive = new SessionArchive(store, decisions);
  await archive.load();
  // Two sessions fit: the newest, b and c. a and the oldest are gone from storage too.
  assert.equal(decisions.snapshot("c").decisions.length, 1);
  assert.equal(decisions.snapshot("b").decisions.length, 1);
  await archive.flush();
  assert.deepEqual([...store.data.keys()].sort(), ["sessions/b", "sessions/c"]);

  const kept = new DecisionStore(4, 10);
  await new SessionArchive(store, kept).load();
  kept.remove("b");
  // A removed session's key goes after the next write.
  await new Promise((resolve) => setTimeout(resolve, 300));
  assert.ok(!store.data.has("sessions/b"));

  store.data.set("sessions/bad", {
    at: 99,
    decisions: [
      {
        id: "z",
        at: 1,
        sessionID: "bad",
        app: "guard",
        outcome: "allowed",
        raw: "output text",
      },
      { id: "x", at: 1, sessionID: "bad", app: "nope", outcome: "allowed" },
    ],
    feedback: { z: "nonsense", x: "wrong" },
  });
  const fresh = new DecisionStore(4, 10);
  await new SessionArchive(store, fresh).load();
  const bad = fresh.snapshot("bad");
  assert.deepEqual(
    bad.decisions.map((d) => d.id),
    ["z"],
  );
  assert.equal(
    "raw" in bad.decisions[0]!,
    false,
    "unknown fields are not kept",
  );
  assert.deepEqual(bad.feedback, {});
});

test("a restarted server still lists a subagent's decisions under its parent", async () => {
  const store = memory();
  const before = new DecisionStore(4, 10);
  const archive = new SessionArchive(store, before);
  await archive.load();
  before.record({ sessionID: "main", app: "router", outcome: "routed" });
  before.record({ sessionID: "child", app: "shield", outcome: "allowed" });
  before.adopt("child", "main");
  await archive.close();

  const after = new DecisionStore(4, 10);
  await new SessionArchive(store, after).load();
  assert.deepEqual(
    after.snapshot("main").decisions.map((decision) => decision.sessionID),
    ["main", "child"],
  );
});
