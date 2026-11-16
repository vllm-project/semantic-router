import assert from "node:assert/strict";
import { test } from "node:test";
import { AnswerCheck } from "../src/check.ts";
import { parseSettings } from "../src/settings.ts";
import { DecisionStore } from "../src/decisions.ts";
import type { Result, Reply } from "../src/runtime.ts";
const answer =
  "This final answer makes a factual claim about the supplied evidence.";
function rig(score = 0.1) {
  const config = parseSettings({ apps: { check: true } }),
    store = new DecisionStore(64, 100);
  const inputs: unknown[] = [];
  const runtime = {
    decide: async (state: unknown): Promise<Result<Reply>> => {
      inputs.push(state);
      return {
        ok: true,
        value: { answers: { hallucination: { noul: score } }, spans: {} },
        elapsedMs: 1,
      };
    },
  };
  const check = new AnswerCheck(config, runtime, (input) =>
    store.record(input),
  );
  function read(session: string, turn: string, key: string, text: string) {
    check.activate(session, turn);
    check.startedTool(session, key, "read");
    check.completedTool(session, key);
    check.delivered(session, key, "read", text);
  }
  return { config, store, inputs, check, read };
}
test("checks use only delivered evidence and do not inherit previous-turn or foreign-session reads", async () => {
  const r = rig();
  r.check.admit("one", "A", "request A");
  r.read("one", "A", "tool-a", "MASKED_A");
  await r.check.complete("one", "A", "answer-a", answer);
  r.check.admit("two", "B", "request B");
  r.read("two", "B", "tool-b", "MASKED_B");
  r.check.admit("one", "A2", "request A2");
  r.check.activate("one", "A2");
  r.check.delivered("one", "tool-a", "read", "OLD_A");
  await r.check.complete("one", "A2", "answer-a2", answer);
  await r.check.complete("two", "B", "answer-b", answer);
  assert.equal(r.inputs.length, 2);
  assert.deepEqual(
    r.inputs.map((input) => (input as { context: string }).context),
    ["read:\nMASKED_A", "read:\nMASKED_B"],
  );
  assert.equal(r.store.snapshot("one").decisions.length, 1);
  assert.equal(r.store.snapshot("one").decisions[0]?.outcome, "verified");
  assert.ok(!JSON.stringify(r.store.snapshot("one")).includes("MASKED_A"));
});
test("queued admissions and tool completion order cannot move evidence into a different turn", async () => {
  const r = rig();
  r.check.admit("one", "A", "A");
  r.check.activate("one", "A");
  r.check.startedTool("one", "call", "shell");
  r.check.admit("one", "B", "B");
  r.check.activate("one", "B");
  r.check.completedTool("one", "call");
  r.check.delivered("one", "call", "shell", "wrong turn");
  await r.check.complete("one", "B", "answer-b", answer);
  assert.equal(r.inputs.length, 0);
  r.check.activate("one", "A");
  r.check.delivered("one", "call", "shell", "correct A");
  await r.check.complete("one", "A", "answer-a", answer);
  assert.equal(
    (r.inputs[0] as { context: string }).context,
    "shell:\ncorrect A",
  );
});
test("undelivered, withheld, failed and synthetic output cannot ground an answer", async () => {
  const r = rig();
  r.check.admit("one", "A", "A");
  r.check.activate("one", "A");
  r.check.startedTool("one", "call", "read");
  r.check.completedTool("one", "call", false);
  r.check.delivered("one", "call", "read", "raw error");
  r.check.startedTool("one", "withheld", "read");
  r.check.completedTool("one", "withheld");
  r.check.delivered(
    "one",
    "withheld",
    "read",
    "vSR: tool content withheld because its checks could not finish.",
  );
  await r.check.complete("one", "A", "answer", answer);
  assert.equal(r.inputs.length, 0);
  r.check.activate("one", "synthetic");
  r.check.startedTool("one", "background", "read");
  r.check.completedTool("one", "background");
  r.check.delivered("one", "background", "read", "synthetic output");
  assert.equal(r.inputs.length, 0);
});
test("duplicate completion is coalesced and unsupported scores do not become verified", async () => {
  const r = rig(0.95);
  r.check.admit("one", "A", "A");
  r.read("one", "A", "call", "facts");
  await Promise.all([
    r.check.complete("one", "A", "answer", answer),
    r.check.complete("one", "A", "answer", answer),
  ]);
  assert.equal(r.inputs.length, 1);
  assert.equal(r.store.snapshot("one").decisions[0]?.outcome, "unverified");
});
test("truncated answer/evidence never earns a full verified result", async () => {
  const r = rig();
  r.check.admit("one", "A", "A");
  r.read("one", "A", "call", "x".repeat(9000));
  await r.check.complete("one", "A", "answer", answer + "y".repeat(9000));
  const decision = r.store.snapshot("one").decisions[0]!;
  assert.equal(decision.outcome, "partial");
  assert.equal(decision.reason, "coverage-limit");
  assert.ok(decision.coverage!.checked < decision.coverage!.total);
});
test("disabled Check, short answers and cancellation cause no review or stale record", async () => {
  const r = rig();
  r.config.apps.check = false;
  r.check.admit("one", "A", "A");
  r.read("one", "A", "call", "facts");
  await r.check.complete("one", "A", "answer", answer);
  assert.equal(r.inputs.length, 0);
  r.config.apps.check = true;
  r.check.admit("one", "B", "B");
  r.read("one", "B", "call", "facts");
  await r.check.complete("one", "B", "short", "done");
  assert.equal(r.inputs.length, 0);
  r.check.interrupt("one");
  await r.check.complete("one", "B", "answer", answer);
  assert.equal(r.inputs.length, 0);
});
