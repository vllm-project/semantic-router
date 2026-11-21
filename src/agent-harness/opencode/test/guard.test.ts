import assert from "node:assert/strict";
import { test } from "node:test";
import { guard } from "../src/guard-policy.ts";
import { parseSettings } from "../src/settings.ts";
import type { Result, Reply } from "../src/runtime.ts";
import { DecisionStore } from "../src/decisions.ts";
const answer = (review: number): Result<Reply> => ({
  ok: true,
  value: {
    answers: {
      kind: { choice: "destructive", probabilities: { destructive: 0.99 } },
      review: { noul: review },
    },
    spans: {},
  },
  elapsedMs: 1,
});

test("fixed shell rules hold offline and over-limit commands hold without partial classification", async () => {
  let calls = 0;
  const runtime = {
    decide: async (): Promise<Result<Reply>> => {
      calls++;
      return { ok: false, code: "offline", elapsedMs: 0 };
    },
  };
  const settings = parseSettings({});
  assert.equal(
    (await guard("echo harmless && sudo rm -rf /", settings, runtime)).reason,
    "fixed-rule",
  );
  assert.equal(
    (await guard("x".repeat(32001), settings, runtime)).reason,
    "coverage-limit",
  );
  assert.equal(
    (
      await guard(
        Array.from({ length: 13 }, (_, i) => `echo ${i}`).join("\n"),
        settings,
        runtime,
      )
    ).reason,
    "coverage-limit",
  );
  assert.equal(calls, 0);
});

test("model guard checks parts in execution order and stops at a held result", async () => {
  const seen: unknown[] = [];
  const runtime = {
    decide: async (state: unknown): Promise<Result<Reply>> => {
      seen.push(state);
      return answer(seen.length === 2 ? 0.98 : 0.1);
    },
  };
  const verdict = await guard(
    "git status && git push --force origin main && echo done",
    parseSettings({}),
    runtime,
  );
  assert.equal(verdict.effect, "ask");
  assert.equal(verdict.reason, "model-risk");
  assert.equal(seen.length, 2);
  assert.equal(seen[1], "Tool: Bash\nCommand: git push --force origin main");
  assert.ok(!JSON.stringify(verdict).includes("git push"));
});

test("one shared deadline covers all parts, with bounded skip and strict hold failures", async () => {
  let now = 0;
  const budgets: number[] = [];
  const runtime = {
    decide: async (
      _state: unknown,
      _questions: unknown,
      budget: number,
    ): Promise<Result<Reply>> => {
      budgets.push(budget);
      now += 600;
      return answer(0.1);
    },
  };
  const bounded = parseSettings({ waits: { guard: 1000 } });
  const verdict = await guard(
    "echo a; echo b; echo c",
    bounded,
    runtime,
    undefined,
    () => now,
  );
  assert.deepEqual(budgets, [1000, 400]);
  assert.equal(verdict.outcome, "skipped");
  assert.equal(verdict.reason, "timeout");
  const strict = parseSettings({ waits: { guard: "done" } });
  const unavailable = {
    decide: async (): Promise<Result<Reply>> => ({
      ok: false,
      code: "unsupported",
      elapsedMs: 0,
    }),
  };
  assert.equal((await guard("echo safe", strict, unavailable)).effect, "ask");
  assert.equal(
    (await guard("echo safe", bounded, unavailable)).effect,
    "allow",
  );
});

test("invalid model replies are unavailable and cancellation always prevents execution", async () => {
  const settings = parseSettings({});
  const malformed = {
    decide: async (): Promise<Result<Reply>> => ({
      ok: true,
      value: {
        answers: {
          kind: { choice: "safe", probabilities: { safe: 1 } },
          review: { noul: 7 },
        },
        spans: {},
      },
      elapsedMs: 0,
    }),
  };
  assert.equal(
    (await guard("echo safe", settings, malformed)).reason,
    "invalid-response",
  );
  const controller = new AbortController();
  const cancelled = {
    decide: async (): Promise<Result<Reply>> => {
      controller.abort();
      return answer(0);
    },
  };
  assert.equal(
    (await guard("echo safe", settings, cancelled, controller.signal)).effect,
    "deny",
  );
  const already = await guard(
    "rm -rf ~",
    settings,
    malformed,
    controller.signal,
  );
  assert.equal(already.reason, "cancelled");
  assert.equal(already.effect, "deny");
});

test("guard settlement is scoped and cannot overwrite router decisions or settled outcomes", () => {
  const store = new DecisionStore(2, 5);
  const held = store.record({
    sessionID: "one",
    app: "guard",
    outcome: "held",
  });
  store.settleGuard("two", held.id, "approved");
  assert.equal(store.snapshot("one").decisions[0]?.outcome, "held");
  store.settleGuard("one", held.id, "declined");
  store.settleGuard("one", held.id, "approved");
  assert.equal(store.snapshot("one").decisions[0]?.outcome, "declined");
  const other = store.record({
    sessionID: "one",
    app: "router",
    outcome: "held",
  });
  store.settleGuard("one", other.id, "approved");
  assert.equal(store.snapshot("one").decisions[1]?.outcome, "held");
});

test("guard options reject misspellings and invalid thresholds", () => {
  for (const guard of [
    null,
    { threshold: 0.9 },
    { reviewThreshold: null },
    { reviewThreshold: NaN },
    { reviewThreshold: -0.1 },
    { reviewThreshold: 1.1 },
  ])
    assert.throws(() => parseSettings({ guard }));
  assert.equal(
    parseSettings({ guard: { reviewThreshold: 0.95 } }).guard.reviewThreshold,
    0.95,
  );
});
