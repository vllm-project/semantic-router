import assert from "node:assert/strict";
import { test } from "node:test";
import { parseSettings } from "../src/settings.ts";

test("defaults are independent and Check remains off", () => {
  const one = parseSettings(undefined),
    two = parseSettings({});
  one.apps.guard = false;
  assert.equal(two.apps.guard, true);
  assert.equal(two.apps.check, false);
  assert.equal(two.runtimeUrl, "http://127.0.0.1:8790/v1/decisions");
  assert.equal(two.calibrateEvery, 200);
  assert.equal(two.routing.turns, false, "your turns stay yours by default");
  assert.equal(parseSettings({ calibrateEvery: 0 }).calibrateEvery, 0);
});

test("invalid controls, unbounded retention and embedded credentials are rejected", () => {
  for (const input of [
    null,
    [],
    { runtimeToken: "do-not-save-me" },
    { apps: { guard: "false" } },
    { apps: { gurad: true } },
    { waits: { guard: 0 } },
    { waits: { unknown: "done" } },
    { maxDecisions: Infinity },
    { maxSessions: 0 },
    { calibrateEvery: -1 },
    { calibrateEvery: 2.5 },
    { routing: { turns: "on" } },
    { runtimeTokenEnv: "BAD-NAME" },
    { runtimeUrl: "https://user:secret@example.test/v1/decisions" },
    { runtimeUrl: "https://example.test/v1/decisions?token=secret" },
    { runtimeUrl: "file:///v1/decisions" },
  ]) {
    assert.throws(() => parseSettings(input));
  }
  const valid = parseSettings({
    waits: { guard: "done" },
    runtimeTokenEnv: "MY_VSR_TOKEN",
    apps: { check: true },
  });
  assert.equal(valid.waits.guard, "done");
  assert.equal(valid.apps.check, true);
});

test("routing requires an ordered same-provider ladder and validates all thresholds", () => {
  const tiers = ["light", "middle", "strong"].map((id) => ({
    model: `fixture/${id}`,
    efforts: ["low", "high"],
  }));
  for (const routing of [
    null,
    { typo: 1 },
    { tiers: null },
    { tiers: tiers.slice(0, 2) },
    { tiers: [tiers[0], tiers[0], tiers[2]] },
    { tiers: [tiers[0], tiers[1], { model: "other/strong" }] },
    { tiers: [{ model: "fixture/light#low" }, tiers[1], tiers[2]] },
    {
      tiers: [
        { model: "fixture/light", efforts: ["low", "low"] },
        tiers[1],
        tiers[2],
      ],
    },
    { goal: null },
    { goal: "cheap" },
    { deadlineMs: 0 },
    { deadlineMs: 10001 },
    { minConfidence: -1 },
    { repeatThreshold: 1.1 },
    { easyBelow: 1, hardFrom: 1 },
    { downgradeConfidence: null },
  ]) {
    assert.throws(() => parseSettings({ routing }));
  }
  const config = parseSettings({ routing: { tiers } });
  tiers[0]!.efforts.push("changed");
  assert.deepEqual(config.routing.tiers[0]?.efforts, ["low", "high"]);
  assert.equal(config.routing.goal, "balanced");
  assert.equal(config.routing.deadlineMs, 1500);
});

test("the defaults are the Claude Code plugin's, as its plugin.json sets them", () => {
  const defaults = parseSettings({});
  assert.deepEqual(
    {
      easyBelow: defaults.routing.easyBelow,
      hardFrom: defaults.routing.hardFrom,
      minConfidence: defaults.routing.minConfidence,
      downgradeConfidence: defaults.routing.downgradeConfidence,
      stepsLow: defaults.routing.stepsLow,
      stepsHigh: defaults.routing.stepsHigh,
      stepsMinConfidence: defaults.routing.stepsMinConfidence,
      feedbackThreshold: defaults.routing.feedbackThreshold,
      repeatThreshold: defaults.routing.repeatThreshold,
      goal: defaults.routing.goal,
      turns: defaults.routing.turns,
      redact: defaults.redact.piiThreshold,
      shield: defaults.shield.injectionThreshold,
      guard: defaults.guard.reviewThreshold,
      check: defaults.check.unsupportedThreshold,
      checkWait: defaults.check.deadlineMs,
      waits: defaults.waits,
      calibrateEvery: defaults.calibrateEvery,
    },
    {
      easyBelow: 0.5,
      hardFrom: 1.5,
      minConfidence: 0.3,
      downgradeConfidence: 0.6,
      stepsLow: 0.5,
      stepsHigh: 1.2,
      stepsMinConfidence: 0.3,
      feedbackThreshold: 0.8,
      repeatThreshold: 0.75,
      goal: "balanced",
      turns: false,
      redact: 0.7,
      shield: 0.9,
      guard: 0.9,
      check: 0.5,
      checkWait: 3000,
      waits: { redact: 3000, shield: 2000, guard: 3000 },
      calibrateEvery: 200,
    },
  );
  assert.throws(
    () => parseSettings({ routing: { stepsLow: 1.2, stepsHigh: 1.2 } }),
    /steps boundaries/,
  );
});
