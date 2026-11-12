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
