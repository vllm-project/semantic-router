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
    assert.ok(before.decisions.length);
    assert.deepEqual(
      after.decisions,
      before.decisions,
      "decisions are kept in plugin storage across a restart",
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

test(
  "a Router listener's profile sets the router's goal and turns, says so once, and /vsr-config changes and follows them",
  { timeout: 120000 },
  async (t) => {
    const native = await fixture(t, false, {
      redact: false,
      router_goal: "save",
      router_turns: true,
    });
    const run = await native.run("OUTPUT_CASE READ");
    const snapshot = async () =>
      (await native.rpc("snapshot", {
        sessionID: run.sessionID,
      })) as ViewSnapshot;
    const before = await snapshot();
    assert.equal(before.apps.redact, false);
    assert.equal(before.controls.goal, "save");
    assert.equal(before.controls.turns, true);
    assert.equal(
      before.controls.note,
      "vSR follows the listener: redact off, goal save, turns on",
    );
    assert.equal(before.controls.listener, true);
    const screen = await native.terminal(
      run.sessionID,
      [
        { at: 2500, keys: "/vsr-config" },
        { at: 3000, keys: "\r" },
        { at: 4000, keys: "\u001b[B".repeat(10) + "\r" }, // your turns: routed → yours
        { at: 4800, keys: "\u001b" },
        { at: 5800, keys: "/vsr" },
        { at: 6300, keys: "\r" },
      ],
      8.5,
    );
    assert.ok(
      // The toast wraps; the RPC above holds its whole text.
      screen.text.includes("vSR follows the listener: redact off,") &&
        /your turns +routed/.test(screen.text) &&
        /│ +turns +yours +routed/.test(screen.text),
      screen.text.slice(-6000),
    );
    const after = await snapshot();
    assert.equal(after.controls.turns, false, "the person's change wins");
    // /vsr-config's follow row, the Claude Code plugin's `/vsr follow`, hands the turns back to the listener.
    const followed = await native.terminal(
      run.sessionID,
      [
        { at: 2500, keys: "/vsr-config" },
        { at: 3000, keys: "\r" },
        { at: 4000, keys: "\u001b[A\u001b[A\r" }, // the row before the last
      ],
      6,
    );
    assert.ok(
      followed.text.includes("vSR: follows the listener's profile."),
      followed.text.slice(-4000),
    );
    assert.equal((await snapshot()).controls.turns, true);
  },
);

test(
  "the setup skill's tools write the settings the user chose and check each app, as the Claude Code plugin's setup scripts",
  { timeout: 120000 },
  async (t) => {
    const native = await fixture(t);
    const run = await native.run("OUTPUT_CASE SETTINGS");
    const snapshot = (await native.rpc("snapshot", {
      sessionID: run.sessionID,
    })) as ViewSnapshot;
    assert.equal(snapshot.apps.check, true);
    const guard = (
      (await native.rpc("thresholds", {})) as {
        app: string;
        values: number[];
        source: string;
      }[]
    ).find((line) => line.app === "guard");
    assert.deepEqual(guard?.values, [0.8]);
    assert.equal(guard?.source, "you");
    const said = (index: number) =>
      JSON.stringify(
        native.requests
          .slice(index)
          .flatMap((request) => request.messages)
          .filter((message) => message.role === "tool"),
      );
    assert.match(said(0), /check on/);
    assert.match(said(0), /guard review line 0\.80 \(set by the user\)/);
    const from = native.requests.length;
    await native.run("OUTPUT_CASE CHECK", run.sessionID);
    assert.match(said(from), /Runtime: from the plugin options/);
  },
);
