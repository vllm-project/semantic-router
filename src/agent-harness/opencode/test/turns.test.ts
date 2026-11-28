import assert from "node:assert/strict";
import { test } from "node:test";
import type { Decision } from "../src/decisions.ts";
import { parseRouting } from "../src/routing-settings.ts";
import type { ModelRef } from "../src/routing-policy.ts";
import type { Reply, Result } from "../src/runtime.ts";
import { parseSettings } from "../src/settings.ts";
import { planTurn, SHORT_REPLY_WORDS, Turns } from "../src/turns.ts";
import type { Price, TurnFacts } from "../src/turns.ts";

const routing = (goal: "balanced" | "quality" | "save" = "balanced") =>
  parseRouting({
    goal,
    tiers: ["light", "medium", "strong"].map((id) => ({
      model: `p/${id}`,
      efforts: ["low", "medium", "high"],
    })),
  });
const sure = 0.95;
const facts = (extra: Partial<TurnFacts>): TurnFacts => ({
  complexity: { score: 1, confidence: sure },
  steps: { score: 1, confidence: sure },
  words: 40,
  own: 1,
  now: undefined,
  escalated: false,
  prices: () => undefined,
  ...extra,
});

test("a hard prompt moves your turns up to the strong tier, an easy one stays off the light tier", () => {
  assert.deepEqual(
    planTurn(
      facts({ complexity: { score: 1.8, confidence: sure } }),
      routing(),
    ),
    { kind: "move", rank: 2, escalated: false, effort: "medium" },
  );
  assert.deepEqual(
    planTurn(
      facts({ complexity: { score: 0.1, confidence: sure } }),
      routing(),
    ),
    { kind: "stay", effort: "medium" },
    "the light tier only with the save goal",
  );
  assert.equal(
    planTurn(
      facts({ complexity: { score: 0.1, confidence: sure }, now: 2 }),
      routing("save"),
    ).kind,
    "move",
  );
  assert.deepEqual(
    planTurn(
      facts({ complexity: { score: 0.1, confidence: sure }, now: 2 }),
      routing("save"),
    ),
    { kind: "move", rank: 0, escalated: false, effort: "medium" },
  );
});

test("an unsure answer keeps the tier; a smaller one needs more confidence and never comes with the quality goal", () => {
  assert.deepEqual(
    planTurn(facts({ complexity: { score: 1.8, confidence: 0.2 } }), routing()),
    {
      kind: "keep",
      reason: "low-confidence",
      would: 2,
      minConfidence: 0.3,
      effort: "medium",
    },
  );
  const down = facts({ complexity: { score: 0.1, confidence: 0.55 }, own: 2 });
  assert.deepEqual(planTurn(down, routing()), {
    kind: "keep",
    reason: "low-confidence",
    would: 1,
    minConfidence: 0.6,
    effort: "medium",
  });
  assert.equal(
    (
      planTurn(
        { ...down, complexity: { score: 0.1, confidence: sure } },
        routing("quality"),
      ) as { reason: string }
    ).reason,
    "quality",
  );
});

test("a short reply never moves your turns down, and keeps the last effort", () => {
  assert.deepEqual(
    planTurn(
      facts({
        complexity: { score: 0.1, confidence: sure },
        now: 2,
        words: SHORT_REPLY_WORDS,
      }),
      routing(),
    ),
    { kind: "keep", reason: "follow-up", would: 1, minConfidence: 0.3 },
  );
});

test("a pushback climbs one tier and one effort from where the last turn ran", () => {
  assert.deepEqual(
    planTurn(
      facts({ escalated: true, now: 1, lastEffort: "medium" }),
      routing(),
    ),
    { kind: "move", rank: 2, escalated: true, effort: "high" },
  );
  assert.deepEqual(
    planTurn(facts({ escalated: true, now: 2, lastEffort: "high" }), routing()),
    { kind: "move", rank: 2, escalated: true, effort: "high" },
    "on the strong tier only the effort can climb, to its top",
  );
  assert.deepEqual(
    planTurn(facts({ escalated: true, own: 3 }), routing()),
    { kind: "stay" },
    "above the tiers there is no step",
  );
});

test("a move down waits while rewriting the cache costs more than three turns would save", () => {
  const prices: Record<number, Price> = {
    1: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
    2: { input: 15, output: 75, cacheRead: 1.5, cacheWrite: 18.75 },
  };
  const warm = facts({
    complexity: { score: 1, confidence: sure },
    own: 2,
    context: 100_000,
    last: { input: 100, output: 100, cacheRead: 20_000, cacheWrite: 0 },
    prices: (rank) => prices[rank],
  });
  const kept = planTurn(warm, routing());
  assert.equal(kept.kind, "keep");
  assert.equal((kept as { reason: string }).reason, "cache");
  const reread = (kept as { reread: { cost: number; saving: number } }).reread;
  assert.ok(reread.cost > reread.saving, JSON.stringify(reread));
  assert.equal(
    planTurn({ ...warm, context: undefined, last: undefined }, routing()).kind,
    "move",
    "with nothing warm, nothing is lost",
  );
});

const settingsWith = (turns: boolean) =>
  parseSettings({
    routing: {
      turns,
      tiers: ["light", "medium", "strong"].map((id) => ({
        model: `p/${id}`,
        efforts: ["low", "medium", "high"],
      })),
    },
  });
// A runtime that answers each prompt with the difficulty its text names (HARD, EASY), sure of it.
const runtime = () => {
  const asked: unknown[] = [];
  return {
    asked,
    decide: async (state: unknown): Promise<Result<Reply>> => {
      asked.push(state);
      const text = String(state);
      const score = text.includes("HARD")
        ? 1.8
        : text.includes("EASY")
          ? 0.1
          : 1;
      return {
        ok: true,
        elapsedMs: 3,
        value: {
          answers: {
            complexity: { score, confidence: 0.95 },
            steps: { score: 1, confidence: 0.95 },
          },
          spans: {},
        },
      };
    },
  };
};

test("your turns move with the router, a model you pick pauses them, and off brings yours back", async () => {
  const settings = settingsWith(true);
  const fake = runtime();
  const recorded: Omit<Decision, "id" | "at">[] = [];
  const switched: ModelRef[] = [];
  const turns = new Turns(
    settings,
    fake,
    (decision) => {
      recorded.push(decision);
      return String(recorded.length);
    },
    {
      modelOf: async () => ({
        providerID: "p",
        id: "medium",
        variant: "medium",
      }),
      switchModel: async (_, model) => void switched.push(model),
      prices: async () => new Map(),
    },
  );
  const long = (word: string) => `${word} ${"please do this ".repeat(5)}`;
  await turns.prompt("s", "m1", long("HARD"), async () => false);
  assert.deepEqual(switched.at(-1), {
    providerID: "p",
    id: "strong",
    variant: "medium",
  });
  assert.deepEqual(
    { ...recorded.at(-1), elapsedMs: undefined },
    {
      sessionID: "s",
      turnID: "m1",
      app: "router",
      scope: "turn",
      subject: long("HARD").trim().replace(/\s+/g, " "),
      elapsedMs: undefined,
      outcome: "routed",
      reason: "selected",
      model: "p/strong#medium",
      score: 1.8,
      confidence: 0.95,
    },
  );
  assert.deepEqual(turns.view("s"), { on: true, model: "p/strong#medium" });
  // The switch vSR made is its own; the next prompt goes on from the strong tier.
  turns.picked("s", { providerID: "p", id: "strong", variant: "medium" });
  assert.equal(turns.view("s").paused, undefined);

  // A pushback climbs on the turn itself.
  await turns.prompt("s", "m2", long("again"), async () => true);
  assert.equal(recorded.at(-1)?.reason, "pushback");
  assert.deepEqual(switched.at(-1), {
    providerID: "p",
    id: "strong",
    variant: "high",
  });

  // A model you pick pauses the router until you pick routed again.
  turns.picked("s", { providerID: "p", id: "light" });
  assert.deepEqual(turns.view("s"), { on: true, paused: "p/light" });
  const asked = fake.asked.length;
  await turns.prompt("s", "m3", long("HARD"), async () => false);
  assert.equal(fake.asked.length, asked, "paused: the prompt is not read");
  turns.resume();
  await turns.prompt("s", "m4", long("HARD"), async () => false);
  assert.equal(recorded.at(-1)?.outcome, "routed");

  // Off again: the next prompt goes back to your model.
  settings.routing.turns = false;
  await turns.prompt("s", "m5", long("HARD"), async () => false);
  assert.deepEqual(switched.at(-1), {
    providerID: "p",
    id: "medium",
    variant: "medium",
  });
  assert.deepEqual(turns.view("s"), { on: false });
});

test("a short reply is read with the prompt before it, and a command is not read", async () => {
  const settings = settingsWith(true);
  const fake = runtime();
  const turns = new Turns(settings, fake, () => "1", {
    modelOf: async () => ({ providerID: "p", id: "medium" }),
    switchModel: async () => {},
    prices: async () => new Map(),
  });
  const task = `HARD ${"refactor the parser ".repeat(5)}`;
  await turns.prompt("s", "m1", task, async () => false);
  await turns.prompt("s", "m2", "yes, do it", async () => false);
  assert.equal(fake.asked.at(-1), `${task}\n\nyes, do it`);
  await turns.prompt("s", "m3", "/compact", async () => false);
  assert.equal(fake.asked.length, 2);
});
