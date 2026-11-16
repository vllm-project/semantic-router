import assert from "node:assert/strict";
import { test } from "node:test";
import { fixture } from "./output-fixture.ts";
import type { ViewSnapshot } from "../../src/view.ts";

test(
  "native durable controls survive a server restart and reject stale clients",
  { timeout: 60000 },
  async (t) => {
    const native = await fixture(t);
    const run = await native.run("OUTPUT_CASE READ");
    const snapshot = async () =>
      (await native.rpc("snapshot", {
        sessionID: run.sessionID,
      })) as ViewSnapshot;
    const before = await snapshot();
    const input = {
      instanceID: before.instanceID,
      expectedRevision: before.controls.revision,
      apps: { check: true },
      waits: { redact: "done" },
      goal: "quality",
    };
    assert.deepEqual(await native.rpc("controls", input), {
      saved: true,
      revision: 1,
    });
    assert.deepEqual(await native.rpc("controls", input), {
      saved: false,
      reason: "stale",
    });
    await native.restart();
    const after = await snapshot();
    assert.notEqual(after.instanceID, before.instanceID);
    assert.equal(after.apps.check, true);
    assert.equal(after.waits.redact, "done");
    assert.equal(after.controls.goal, "quality");
    assert.equal(after.controls.revision, 1);
    assert.equal(after.controls.persistence, "durable");
    assert.deepEqual(
      after.decisions,
      [],
      "ephemeral decisions must not pretend to survive a restart",
    );
    assert.deepEqual(
      await native.rpc("controls", { ...input, expectedRevision: 1 }),
      { saved: false, reason: "stale" },
    );
    assert.deepEqual(
      await native.rpc("controls", {
        ...input,
        instanceID: after.instanceID,
        expectedRevision: 1,
        apps: { check: false },
      }),
      { saved: true, revision: 2 },
    );
    assert.equal((await snapshot()).apps.check, false);
    assert.deepEqual(
      await native.rpc("controls", {
        instanceID: after.instanceID,
        expectedRevision: 2,
        apps: { check: !before.apps.check },
      }),
      { saved: true, revision: 3 },
    );
    // No listener answers here (404), so following hands the app back to the plugin's own default.
    assert.deepEqual(
      await native.rpc("controls", {
        instanceID: after.instanceID,
        expectedRevision: 3,
        follow: true,
      }),
      { saved: true, revision: 4 },
    );
    assert.equal((await snapshot()).apps.check, before.apps.check);
  },
);
