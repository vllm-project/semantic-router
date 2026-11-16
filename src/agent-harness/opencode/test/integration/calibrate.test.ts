import assert from "node:assert/strict";
import { test } from "node:test";
import { fixture } from "./output-fixture.ts";
import type {
  CalibrationQuestion,
  CalibrationReport,
  ViewSnapshot,
} from "../../src/view.ts";

test(
  "native decisions are logged for calibration across a restart, and /vsr-calibrate asks about them",
  { timeout: 90000 },
  async (t) => {
    const native = await fixture(t);
    const run = await native.run("OUTPUT_CASE READ");
    const snapshot = (await native.rpc("snapshot", {
      sessionID: run.sessionID,
    })) as ViewSnapshot;
    const shield = snapshot.decisions.find(
      (decision) => decision.app === "shield",
    );
    assert.match(shield?.subject ?? "", /^read .*fixture\.txt$/);
    assert.ok(
      !JSON.stringify(snapshot).includes("ignore previous"),
      "the subject names the source, never its content",
    );

    await native.restart();
    const report = (await native.rpc(
      "calibrationReport",
      {},
    )) as CalibrationReport;
    assert.equal(report.persistence, "durable");
    const counts = Object.fromEntries(
      report.apps.map((item) => [item.app, item.decisions]),
    );
    assert.ok(
      counts.shield! >= 1 && counts.redact! >= 1,
      JSON.stringify(counts),
    );
    const [question] = (await native.rpc("calibrationSample", {
      app: "shield",
    })) as CalibrationQuestion[];
    assert.equal(question?.acted, true);
    assert.deepEqual(question?.thresholds, [0.9]);

    const second = await native.run("OUTPUT_CASE READ");
    const screen = await native.terminal(
      second.sessionID,
      [
        { at: 2000, keys: "/vsr-calibrate" },
        { at: 2800, keys: "\r" },
        { at: 4000, keys: "Shield" },
        { at: 4600, keys: "\r" },
        { at: 6000, keys: "\r" }, // Yes, it should have warned.
        { at: 7500, keys: "\r" }, // And the next one, if any.
        { at: 9000, keys: "\r" },
        { at: 10500, keys: "\r" }, // Balanced.
      ],
      13,
    );
    assert.ok(
      screen.text.includes("vSR calibrate: which app?") &&
        screen.text.includes("Should vSR have warned the agent about") &&
        screen.text.includes("needs 8 to move a line"),
      screen.text.slice(-6000),
    );
    const after = (await native.rpc(
      "calibrationReport",
      {},
    )) as CalibrationReport;
    assert.ok(
      after.apps.find((item) => item.app === "shield")!.answered >= 1,
      "the answer is kept",
    );
    assert.equal(
      after.apps.find((item) => item.app === "shield")!.calibrated,
      false,
      "too few answers change nothing",
    );
  },
);
