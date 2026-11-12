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
