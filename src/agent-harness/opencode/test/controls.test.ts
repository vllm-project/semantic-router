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
test("turns are saved beside the controls, kept the person's, and an option naming them wins", async () => {
  const config = parseSettings({});
  let saved: Record<string, unknown> = {};
  const pinned: unknown[] = [];
  const controls = new Controls(
    config,
    "instance",
    async (value) => {
      saved = value as never;
    },
    () => {},
    (values) => pinned.push(values),
  );
  assert.equal(controls.state().turns, false);
  assert.deepEqual(
    await controls.update({
      instanceID: "instance",
      expectedRevision: 0,
      turns: true,
      goal: "save",
    }),
    { saved: true, revision: 1 },
  );
  assert.equal(config.routing.turns, true);
  assert.equal(saved.turns, true);
  assert.deepEqual(saved.pinned, ["goal", "turns"]);
  assert.deepEqual(pinned.at(-1), { goal: "save", turns: true });
  assert.deepEqual(
    await controls.update({
      instanceID: "instance",
      expectedRevision: 1,
      turns: "yes",
    }),
    { saved: false, reason: "invalid" },
  );

  // Controls saved before turns existed still load, and turns stay off.
  const { turns: _, ...older } = saved;
  const before = parseSettings({});
  await new Controls(
    before,
    "i",
    async () => {},
    () => {},
  ).load(async () => older);
  assert.equal(before.routing.goal, "save");
  assert.equal(before.routing.turns, false);

  const reloaded = parseSettings({});
  await new Controls(
    reloaded,
    "i",
    async () => {},
    () => {},
  ).load(async () => saved);
  assert.equal(reloaded.routing.turns, true);
  const named = parseSettings({ routing: { turns: false } });
  await new Controls(
    named,
    "i",
    async () => {},
    () => {},
    () => {},
    async () => {},
    { turns: true },
  ).load(async () => saved);
  assert.equal(named.routing.turns, false, "the option the person named wins");

  // Following the listener hands goal and turns back too.
  assert.deepEqual(
    await controls.update({
      instanceID: "instance",
      expectedRevision: 1,
      follow: true,
    }),
    { saved: true, revision: 2 },
  );
  assert.equal(saved.pinned, undefined);
  assert.deepEqual(
    await controls.update({
      instanceID: "instance",
      expectedRevision: 2,
      follow: true,
      turns: true,
    }),
    { saved: false, reason: "invalid" },
  );
});

test("the router's line for a smaller tier is kept as calibrateEvery is, and an option naming it wins", async () => {
  const config = parseSettings({});
  let saved: unknown;
  const controls = new Controls(
    config,
    "i",
    async (value) => {
      saved = value;
    },
    () => {},
  );
  assert.equal(controls.state().downgradeConfidence, 0.6);
  assert.deepEqual(
    await controls.update({
      instanceID: "i",
      expectedRevision: 0,
      downgradeConfidence: 0.75,
    }),
    { saved: true, revision: 1 },
  );
  assert.equal(config.routing.downgradeConfidence, 0.75);
  assert.deepEqual(
    await controls.update({
      instanceID: "i",
      expectedRevision: 1,
      downgradeConfidence: 1.5,
    }),
    { saved: false, reason: "invalid" },
  );
  const reloaded = parseSettings({});
  await new Controls(
    reloaded,
    "j",
    async () => {},
    () => {},
  ).load(async () => saved);
  assert.equal(reloaded.routing.downgradeConfidence, 0.75);
  const named = parseSettings({ routing: { downgradeConfidence: 0.65 } });
  await new Controls(
    named,
    "k",
    async () => {},
    () => {},
    () => {},
    async () => {},
    { downgradeConfidence: true },
  ).load(async () => saved);
  assert.equal(named.routing.downgradeConfidence, 0.65);
});

test("the steps answer's lines are kept as the smaller-model line is, in their order", async () => {
  const config = parseSettings({});
  const controls = new Controls(
    config,
    "i",
    async () => {},
    () => {},
  );
  assert.deepEqual(
    await controls.update({
      instanceID: "i",
      expectedRevision: 0,
      stepsLow: 0.4,
      stepsHigh: 1.4,
      stepsMinConfidence: 0.5,
    }),
    { saved: true, revision: 1 },
  );
  assert.deepEqual(
    [
      config.routing.stepsLow,
      config.routing.stepsHigh,
      config.routing.stepsMinConfidence,
    ],
    [0.4, 1.4, 0.5],
  );
  assert.deepEqual(
    await controls.update({
      instanceID: "i",
      expectedRevision: 1,
      stepsLow: 1.5,
    }),
    { saved: false, reason: "invalid" },
    "the few-steps line stays under the many-steps line",
  );
  assert.deepEqual(
    await controls.update({
      instanceID: "i",
      expectedRevision: 1,
      stepsHigh: 2.5,
    }),
    { saved: false, reason: "invalid" },
  );
});
