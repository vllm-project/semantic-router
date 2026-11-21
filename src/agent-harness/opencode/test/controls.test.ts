import assert from "node:assert/strict";
import { test } from "node:test";
import { Controls } from "../src/controls.ts";
import { parseSettings } from "../src/settings.ts";
test("controls persist before becoming visible and serialize concurrent compare-and-set writes", async () => {
  const config = parseSettings({});
  let saved: unknown;
  let release!: () => void;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  const controls = new Controls(
    config,
    "instance",
    async (value) => {
      await gate;
      saved = value;
    },
    () => {},
  );
  const first = controls.update({
    instanceID: "instance",
    expectedRevision: 0,
    apps: { guard: false },
    waits: { redact: "done" },
  });
  const second = controls.update({
    instanceID: "instance",
    expectedRevision: 0,
    apps: { shield: false },
  });
  await Promise.resolve();
  assert.equal(config.apps.guard, true);
  release();
  assert.deepEqual(await first, { saved: true, revision: 1 });
  assert.deepEqual(await second, { saved: false, reason: "stale" });
  assert.equal(config.apps.guard, false);
  assert.equal(config.apps.shield, true);
  const reloaded = parseSettings({}),
    next = new Controls(
      reloaded,
      "new-instance",
      async () => {},
      () => {},
    );
  await next.load(async () => saved);
  assert.equal(reloaded.apps.guard, false);
  assert.equal(reloaded.waits.redact, "done");
  assert.deepEqual(
    await next.update({
      instanceID: "instance",
      expectedRevision: 1,
      apps: { guard: true },
    }),
    { saved: false, reason: "stale" },
  );
});
test("storage failures and invalid settings never optimistically change controls", async () => {
  const config = parseSettings({});
  const controls = new Controls(
    config,
    "instance",
    async () => {
      throw new Error("private storage failure");
    },
    () => {},
  );
  assert.deepEqual(
    await controls.update({
      instanceID: "instance",
      expectedRevision: 0,
      apps: { check: true },
    }),
    { saved: false, reason: "storage" },
  );
  assert.equal(config.apps.check, false);
  assert.equal(controls.state().persistence, "unavailable");
  for (const patch of [
    { apps: { guard: "false" } },
    { apps: { typo: true } },
    { waits: { redact: 0 } },
    { goal: "cheapest" },
  ])
    assert.deepEqual(
      await controls.update({
        instanceID: "instance",
        expectedRevision: 0,
        ...patch,
      }),
      { saved: false, reason: "invalid" },
    );
});
test("explicit configured-default changes supersede saved overrides", async () => {
  const original = parseSettings({});
  let saved: unknown;
  const first = new Controls(
    original,
    "one",
    async (value) => {
      saved = value;
    },
    () => {},
  );
  await first.update({
    instanceID: "one",
    expectedRevision: 0,
    apps: { check: true },
  });
  const edited = parseSettings({ apps: { shield: false } }),
    second = new Controls(
      edited,
      "two",
      async () => {},
      () => {},
    );
  await second.load(async () => saved);
  assert.equal(edited.apps.check, false);
  assert.equal(edited.apps.shield, false);
});
