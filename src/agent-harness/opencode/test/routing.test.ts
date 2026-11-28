import assert from "node:assert/strict";
import { test } from "node:test";
import { choose, routeScores } from "../src/routing-policy.ts";
import type { Candidate } from "../src/routing-policy.ts";
import { Routing } from "../src/routing.ts";
import { parseSettings } from "../src/settings.ts";
import { DecisionStore } from "../src/decisions.ts";
import { RuntimeClient } from "../src/runtime.ts";
import type { Result, Reply } from "../src/runtime.ts";
const tiers = ["light", "middle", "strong"].map((model) => ({
  model: `fixture/${model}`,
  efforts: ["low", "medium", "high"],
}));
const settings = (extra: Record<string, unknown> = {}) =>
  parseSettings({ routing: { tiers, ...extra } });
const candidates: Candidate[] = tiers.map((tier) => ({
  providerID: "fixture",
  id: tier.model.split("/")[1]!,
  enabled: true,
  variants: tier.efforts.map((id) => ({ id })),
  capabilities: { tools: true },
}));
const scores = (difficulty = 0.2, confidence = 0.95, steps = 0.1) => ({
  complexity: { score: difficulty, confidence },
  steps: { score: steps, confidence },
});
const reply = (answers: Record<string, unknown>) =>
  Response.json({ answers, spans: {} });
function rig(
  options: { push?: boolean; repeat?: boolean; offline?: boolean } = {},
) {
  const config = settings(),
    store = new DecisionStore(64, 100);
  let calls = 0;
  const runtime = new RuntimeClient(config, {
    fetch: async (_url, init) => {
      calls++;
      if (options.offline) throw new Error("fixture offline");
      const body = JSON.parse(String(init?.body));
      if (body.questions.user_feedback)
        return reply({
          user_feedback: {
            probabilities: {
              WRONG_ANSWER: options.push ? 0.95 : 0.01,
              WANT_DIFFERENT: 0.01,
              NO_FEEDBACK: options.push ? 0.04 : 0.98,
            },
          },
        });
      if (body.questions.reask)
        return reply({ reask: { noul: options.repeat ? 0.95 : 0.01 } });
      return reply(scores());
    },
  });
  const routing = new Routing(
    config,
    runtime,
    (input) => store.record(input).id,
  );
  return { routing, store, getCalls: () => calls };
}

test("tier and effort selection respects confidence, goals, parent identity and available variants", () => {
  const parent = { providerID: "fixture", id: "middle", variant: "medium" };
  assert.equal(
    choose(settings().routing, parent, candidates, scores(), false).model,
    "fixture/light#low",
  );
  assert.equal(
    choose(
      settings().routing,
      parent,
      candidates,
      scores(0.8, 0.95, 0.8),
      false,
    ).model,
    "fixture/middle#medium",
  );
  assert.equal(
    choose(
      settings().routing,
      parent,
      candidates,
      scores(1.8, 0.95, 1.8),
      false,
    ).model,
    "fixture/strong#high",
  );
  assert.equal(
    choose(settings().routing, parent, candidates, scores(0.2, 0.55), false)
      .reason,
    "low-confidence",
  );
  assert.equal(
    choose(
      settings({ goal: "save" }).routing,
      parent,
      candidates,
      scores(0.2, 0.55),
      false,
    ).model,
    "fixture/light#low",
  );
  assert.equal(
    choose(
      settings({ goal: "quality" }).routing,
      parent,
      candidates,
      scores(),
      false,
    ).reason,
    "quality",
  );
  assert.equal(
    choose(
      settings().routing,
      { providerID: "other", id: "middle" },
      candidates,
      scores(),
      false,
    ).reason,
    "unknown-parent",
  );
  assert.equal(
    choose(
      settings().routing,
      parent,
      [{ ...candidates[0]!, variants: [] }],
      scores(),
      false,
    ).reason,
    "invalid-tier",
  );
  assert.equal(
    choose(
      settings().routing,
      parent,
      candidates.map((c) => ({ ...c, enabled: false })),
      scores(),
      false,
    ).reason,
    "invalid-tier",
  );
  assert.equal(
    choose(settings().routing, parent, candidates, scores(0.2, 0.1), false)
      .model,
    undefined,
  );
  assert.equal(
    routeScores({
      answers: { complexity: { score: 7, confidence: 1 } },
      spans: {},
    }),
    undefined,
  );
});

test("escalation raises one tier and effort, never below parent or above configured variants", () => {
  assert.equal(
    choose(
      settings().routing,
      { providerID: "fixture", id: "middle", variant: "high" },
      candidates,
      scores(),
      true,
    ).model,
    "fixture/middle#high",
  );
  assert.equal(
    choose(
      settings().routing,
      { providerID: "fixture", id: "middle" },
      candidates,
      undefined,
      true,
    ).model,
    "fixture/strong#high",
  );
  assert.equal(
    choose(
      settings().routing,
      { providerID: "fixture", id: "strong", variant: "high" },
      candidates,
      scores(1.9),
      true,
    ).model,
    "fixture/strong#high",
  );
});

test("one pushback is consumed by only one of two concurrent child launches", async () => {
  const { routing, store } = rig({ push: true });
  routing.admit("one", "prompt", "That was wrong.");
  assert.equal(
    await routing.activate("one", "prompt", {
      providerID: "fixture",
      id: "middle",
    }),
    true,
  );
  const routes = await Promise.all([
    routing.route("one", "child-a", "easy task", candidates),
    routing.route("one", "child-b", "easy task", candidates),
  ]);
  assert.deepEqual(
    routes.map((route) => route.model),
    ["fixture/middle#medium", "fixture/light#low"],
  );
  assert.ok(
    store
      .snapshot("one")
      .decisions.some((decision) => decision.app === "escalate"),
  );
  // The prompt is kept only as the escalation's clipped subject, which calibration asks about.
  const snapshot = store.snapshot("one");
  const escalation = snapshot.decisions.find(
    (decision) => decision.app === "escalate",
  );
  assert.equal(escalation?.subject, "That was wrong.");
  assert.deepEqual(escalation?.features, [0.95, 0]);
  assert.equal(
    JSON.stringify(snapshot).split("That was wrong").length - 1,
    1,
    "nowhere but the subject",
  );
});

test("queued admissions do not replace the executing prompt, and foreign/synthetic messages do not inherit escalation", async () => {
  const { routing, getCalls } = rig({ push: true });
  routing.admit("one", "first", "Wrong first answer");
  routing.admit("one", "queued", "Queued request");
  await routing.activate("one", "first", {
    providerID: "fixture",
    id: "middle",
  });
  const first = await routing.route("one", "child", "task", candidates);
  assert.equal(first.model, "fixture/middle#medium");
  const calls = getCalls();
  await routing.activate("one", "first", {
    providerID: "fixture",
    id: "middle",
  });
  assert.equal(getCalls(), calls, "same executing message must not reclassify");
  await routing.activate("two", "first", {
    providerID: "fixture",
    id: "middle",
  });
  assert.equal(
    (await routing.route("two", "other", "task", candidates)).model,
    "fixture/light#low",
  );
  await routing.activate("one", "background-without-admission", {
    providerID: "fixture",
    id: "middle",
  });
  assert.equal(
    (await routing.route("one", "background", "task", candidates)).model,
    "fixture/light#low",
  );
});

test("repeated executed request escalates; prior cancelled admission is not treated as the prior human request", async () => {
  const { routing } = rig({ repeat: true });
  routing.admit("one", "cancelled", "different queued topic");
  routing.admit("one", "first", "original topic");
  assert.equal(
    await routing.activate("one", "first", {
      providerID: "fixture",
      id: "middle",
    }),
    false,
  );
  routing.admit("one", "second", "original topic again");
  assert.equal(
    await routing.activate("one", "second", {
      providerID: "fixture",
      id: "middle",
    }),
    true,
  );
});

test("offline classification keeps the parent, while previously established escalation can still raise it", async () => {
  const { routing, store } = rig({ offline: true });
  routing.admit("one", "first", "ordinary request");
  await routing.activate("one", "first", {
    providerID: "fixture",
    id: "middle",
  });
  assert.equal(
    (await routing.route("one", "child", "task", candidates)).model,
    undefined,
  );
  assert.equal(store.snapshot("one").decisions.at(-1)?.reason, "offline");
  const config = settings();
  const pushed = new Routing(
    config,
    {
      decide: async (_state, questions) =>
        questions.user_feedback
          ? {
              ok: true,
              value: {
                answers: {
                  user_feedback: { probabilities: { WRONG_ANSWER: 0.95 } },
                },
                spans: {},
              },
              elapsedMs: 0,
            }
          : { ok: false, code: "offline", elapsedMs: 0 },
    },
    (input) => store.record(input).id,
  );
  pushed.admit("pushed", "first", "wrong");
  assert.equal(
    await pushed.activate("pushed", "first", {
      providerID: "fixture",
      id: "middle",
    }),
    true,
  );
  assert.equal(
    (await pushed.route("pushed", "child", "task", candidates)).model,
    "fixture/strong#high",
  );
});

test("interrupting a routing-only or synthetic turn cancels its request without requiring a human escalation", async () => {
  for (const human of [false, true]) {
    const config = parseSettings({
      apps: { escalate: false },
      routing: { tiers },
    });
    const store = new DecisionStore(2, 10);
    const routing = new Routing(
      config,
      {
        decide: (_state, _questions, _timeout, signal) =>
          new Promise<Result<Reply>>((resolve) => {
            signal?.addEventListener(
              "abort",
              () => resolve({ ok: false, code: "cancelled", elapsedMs: 0 }),
              { once: true },
            );
          }),
      },
      (input) => store.record(input).id,
    );
    if (human) routing.admit("one", "first", "request");
    await routing.activate("one", human ? "first" : undefined, {
      providerID: "fixture",
      id: "middle",
    });
    const pending = routing.route("one", "child", "task", candidates);
    routing.interrupt("one");
    assert.deepEqual(await pending, {});
    assert.deepEqual(store.snapshot("one").decisions, []);
    routing.close();
  }
});

test("interrupting a pending classification stops routing and never leaks its escalation into the next turn", async () => {
  const config = settings(),
    store = new DecisionStore(64, 10);
  const runtime = {
    decide: (
      _state: unknown,
      _questions: unknown,
      _timeout: number,
      signal?: AbortSignal,
    ) =>
      new Promise<Result<Reply>>((resolve) => {
        signal?.addEventListener(
          "abort",
          () => resolve({ ok: false, code: "cancelled", elapsedMs: 0 }),
          { once: true },
        );
      }),
  };
  const routing = new Routing(
    config,
    runtime,
    (input) => store.record(input).id,
  );
  routing.admit("one", "first", "wrong");
  const waiting = routing.activate("one", "first", {
    providerID: "fixture",
    id: "middle",
  });
  routing.interrupt("one");
  assert.equal(await waiting, false);
  assert.equal(store.snapshot("one").decisions.length, 0);
  routing.close();
});

test("a model without variants routes without one: OpenCode's `default` placeholder is not a variant", () => {
  // Found qualifying against a gateway whose models declare no variants: the parent reports variant `default`, and
  // a child asked for `model#default` fails to spawn.
  const plain = ["light", "middle", "strong"].map((model) => ({
    model: `gateway/${model}`,
  }));
  const bare: Candidate[] = plain.map((tier) => ({
    providerID: "gateway",
    id: tier.model.split("/")[1]!,
    enabled: true,
    variants: [],
    capabilities: { tools: true },
  }));
  const parent = { providerID: "gateway", id: "middle", variant: "default" };
  const routing = parseSettings({ routing: { tiers: plain } }).routing;
  assert.equal(
    choose(routing, parent, bare, scores(0.8, 0.95), false).model,
    "gateway/middle",
  );
  assert.equal(
    choose(routing, parent, bare, scores(0.8, 0.95), true).model,
    "gateway/strong",
  );
  // A tier with efforts treats the placeholder as no variant chosen, not as one outside its ladder.
  const laddered = parseSettings({
    routing: {
      tiers: plain.map((tier) => ({ ...tier, efforts: ["low", "high"] })),
    },
  }).routing;
  const withEfforts = bare.map((item) => ({
    ...item,
    variants: [{ id: "low" }, { id: "high" }],
  }));
  assert.equal(
    choose(laddered, parent, withEfforts, scores(0.8, 0.95, 0.1), false).model,
    "gateway/middle#low",
  );
});

test("effort follows the steps answer on its own lines, as the Claude Code plugin's pickEffort", () => {
  const parent = { providerID: "fixture", id: "middle", variant: "medium" };
  const effort = (steps: number, confidence = 0.95) =>
    choose(
      settings().routing,
      parent,
      candidates,
      {
        complexity: { score: 0.8, confidence: 0.95 },
        steps: { score: steps, confidence },
      },
      false,
    ).model;
  // Below the few-steps line 0.5, the lowest; from the many-steps line 1.2, the highest; the middle between.
  assert.equal(effort(0.4), "fixture/middle#low");
  assert.equal(effort(0.8), "fixture/middle#medium");
  assert.equal(
    effort(1.3),
    "fixture/middle#high",
    "1.3 is many steps on the steps line, though under the strong difficulty line",
  );
  assert.equal(
    effort(1.3, 0.2),
    "fixture/middle#medium",
    "under the steps confidence line the answer is not acted on",
  );
});
