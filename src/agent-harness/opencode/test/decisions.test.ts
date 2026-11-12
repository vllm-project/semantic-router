import assert from "node:assert/strict";
import { test } from "node:test";
import { DecisionStore } from "../src/decisions.ts";

test("sessions and feedback are isolated, snapshots cannot mutate internal state", () => {
  const store = new DecisionStore(2, 3);
  const first = store.record({
    sessionID: "one",
    app: "router",
    outcome: "routed",
  });
  store.record({ sessionID: "two", app: "guard", outcome: "held" });
  assert.throws(() => store.label("two", first.id, "wrong"));
  store.label("one", first.id, "needed-stronger");
  const snapshot = store.snapshot("one");
  snapshot.decisions[0]!.model = "mutated";
  snapshot.feedback[first.id] = "wrong";
  assert.equal(store.snapshot("one").decisions[0]?.model, undefined);
  assert.equal(store.snapshot("one").feedback[first.id], "needed-stronger");
  assert.equal(store.snapshot("two").decisions.length, 1);
});

test("bounded retention evicts decisions with their labels and the least recently used session", () => {
  const store = new DecisionStore(2, 1);
  const first = store.record({
    sessionID: "one",
    app: "guard",
    outcome: "held",
  });
  store.label("one", first.id, "right");
  store.record({ sessionID: "one", app: "guard", outcome: "approved" });
  assert.deepEqual(store.snapshot("one").feedback, {});
  store.record({ sessionID: "two", app: "guard", outcome: "held" });
  store.snapshot("one");
  store.record({ sessionID: "three", app: "guard", outcome: "held" });
  assert.equal(store.snapshot("two").decisions.length, 0);
  assert.equal(store.snapshot("one").decisions[0]?.outcome, "approved");
  const before = store.snapshot("one");
  store.remove("one");
  assert.ok(store.snapshot("one").revision > before.revision);
});

test("metadata projection discards raw outputs and validates measurements", () => {
  const store = new DecisionStore(1, 2);
  const input = {
    sessionID: "one",
    app: "redact" as const,
    outcome: "redacted" as const,
    rawOutput: "private fixture",
    labels: ["email"],
  };
  const decision = store.record(input);
  assert.ok(!JSON.stringify(store.snapshot("one")).includes("private fixture"));
  input.labels.push("changed");
  assert.deepEqual(store.snapshot("one").decisions[0]?.labels, ["email"]);
  assert.throws(() => store.label("one", decision.id, "right-model"));
  assert.throws(() =>
    store.record({
      sessionID: "one",
      app: "guard",
      outcome: "held",
      score: NaN,
    }),
  );
});

test("child observations are session-scoped and keep planned and actual model facts separate", () => {
  const store = new DecisionStore(2, 5);
  const decision = store.record({
    sessionID: "one",
    app: "router",
    outcome: "routed",
    model: "fixture/light#low",
    reason: "selected",
  });
  const usage = {
    input: 10,
    output: 2,
    cachedInput: 0,
    reasoning: 0,
    cost: 0,
    rawOutput: "private fixture",
  };
  store.observe("two", decision.id, { model: "wrong-session" });
  assert.equal(store.snapshot("one").decisions[0]?.observedModel, undefined);
  store.observe("one", decision.id, {
    model: "fixture/middle#medium",
    childSessionID: "child",
    usage,
  });
  usage.input = 99;
  const snapshot = store.snapshot("one");
  assert.equal(snapshot.decisions[0]?.model, "fixture/light#low");
  assert.equal(snapshot.decisions[0]?.observedModel, "fixture/middle#medium");
  assert.equal(snapshot.decisions[0]?.usage?.input, 10);
  assert.ok(!JSON.stringify(snapshot).includes("private fixture"));
  assert.throws(() =>
    store.observe("one", decision.id, { usage: { ...usage, input: NaN } }),
  );
  store.observe("one", decision.id, { failed: true });
  assert.equal(store.snapshot("one").decisions[0]?.reason, "spawn-failed");
});
