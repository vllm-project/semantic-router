import assert from "node:assert/strict";
import { test } from "node:test";
import { fixture } from "./output-fixture.ts";
import type {
  CalibrationQuestion,
  CalibrationReport,
  ProbeResult,
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
    const again = (await native.rpc("calibrationSample", {
      app: "shield",
      skip: [question!.id],
    })) as CalibrationQuestion[];
    assert.ok(
      !again.some((item) => item.id === question!.id),
      "a question asked this run is not asked again",
    );

    const second = await native.run("OUTPUT_CASE READ");
    const screen = await native.terminal(
      second.sessionID,
      // Opened well after the TUI started, as a person would, not in its first seconds.
      [
        { at: 12000, keys: "/vsr-calibrate" },
        { at: 12800, keys: "\r" },
        { at: 14000, keys: "Injection" },
        { at: 14600, keys: "\r" },
        { at: 16000, keys: "\r" }, // The report: continue.
        { at: 17500, keys: "\r" }, // Yes, it should have warned.
        { at: 19000, keys: "\r" }, // And the next one, if any.
        { at: 20500, keys: "\r" },
        { at: 22000, keys: "\r" }, // Balanced.
      ],
      25,
    );
    assert.ok(
      screen.text.includes("vSR calibrate: which app?") &&
        // Claude Code's step 2: what the log holds, before any question.
        screen.text.includes("vSR's log holds") &&
        /files? or pages? read: the agent was warned about/.test(screen.text) &&
        screen.text.includes("Should vSR have warned the agent about") &&
        // The decision is shown in full under the question, not cut to one option line.
        screen.text.includes("vSR warned · score 0.99 (line 0.90)") &&
        screen.text.includes("file or page") &&
        screen.text.includes("Not enough answers yet"),
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
    assert.equal(
      after.model,
      "fixture",
      "the decision model the runtime lists",
    );

    // The probe: the known cases of the apps asked for, judged at the thresholds they were measured at.
    const probed = (await native.rpc("calibrationProbe", {
      apps: ["shield"],
    })) as ProbeResult;
    assert.equal(probed.model, "fixture");
    assert.deepEqual(
      probed.rows.map((row) => [row.app, row.status === "error"]),
      [
        ["shield", false],
        ["shield", false],
        ["shield", false],
        ["shield", false],
      ],
    );
    const shown = await native.terminal(
      second.sessionID,
      [
        { at: 12000, keys: "/vsr-calibrate" },
        { at: 12800, keys: "\r" },
        { at: 14000, keys: "Probe" }, // The probe is a choice of /vsr-calibrate's.
        { at: 14600, keys: "\r" },
      ],
      20,
    );
    assert.ok(
      shown.text.includes("Decision model: fixture.") &&
        /Shield, 4 known cases: /.test(shown.text),
      shown.text.slice(-6000),
    );
  },
);

test(
  "/vsr-calibrate offers another round while an app is short of 8 answers, as Claude Code's does",
  { timeout: 300000 },
  async (t) => {
    const native = await fixture(t);
    let last = await native.run("OUTPUT_CASE READ");
    for (let i = 0; i < 5; i++) last = await native.run("OUTPUT_CASE READ");
    const screen = await native.terminal(
      last.sessionID,
      [
        { at: 12000, keys: "/vsr-calibrate" },
        { at: 12800, keys: "\r" },
        { at: 14000, keys: "Injection" },
        { at: 14600, keys: "\r" },
        { at: 16000, keys: "\r" }, // The report: continue.
        { at: 17500, keys: "\r" }, // Both questions of the first round.
        { at: 19000, keys: "\r" },
        { at: 20500, keys: "\r" }, // Answer more questions.
        { at: 22000, keys: "\r" },
        { at: 23500, keys: "\r" },
      ],
      27,
    );
    assert.ok(
      // The line shows where each question is in its round; the header counts the answers.
      screen.text.includes("question 1 of 2") &&
        screen.text.includes("question 2 of 2") &&
        screen.text.includes("Injection warnings · 1 of 8 answers") &&
        screen.text.includes("Injection warnings · 3 of 8 answers") &&
        screen.text.includes("vSR calibrate: 2 of 8 answers") &&
        screen.text.includes("Answer more questions") &&
        screen.text.includes("vSR calibrate: 4 of 8 answers"),
      screen.text.slice(-6000),
    );
    const after = (await native.rpc(
      "calibrationReport",
      {},
    )) as CalibrationReport;
    assert.equal(after.apps.find((item) => item.app === "shield")!.answered, 4);
  },
);

test(
  "/vsr-calibrate tunes several apps at once, as Claude Code's does: pick them, one goal, one suggestion",
  { timeout: 300000 },
  async (t) => {
    const native = await fixture(t);
    const run = await native.run("OUTPUT_CASE READ");
    const screen = await native.terminal(
      run.sessionID,
      [
        { at: 12000, keys: "/vsr-calibrate" },
        { at: 12800, keys: "\r" },
        { at: 14000, keys: "Several" },
        { at: 14600, keys: "\r" },
        { at: 15800, keys: "\r" }, // Enter with nothing ticked stays.
        { at: 16400, keys: " " }, // Space ticks the app under the cursor.
        { at: 17000, keys: "\x1b[B" },
        { at: 17600, keys: " " },
        { at: 18200, keys: " " }, // And unticks it.
        { at: 18800, keys: " " },
        { at: 20000, keys: "\r" },
        { at: 21500, keys: "\r" }, // The report: continue.
        { at: 23000, keys: "\x1b" }, // No answers: straight to the goal.
        { at: 24500, keys: "\r" }, // Balanced.
      ],
      28,
    );
    assert.ok(
      screen.text.includes("vSR calibrate: which apps?") &&
        screen.text.includes("Tick at least one app to go on.") &&
        screen.text.includes("[✓] Injection warnings") &&
        screen.text.includes(
          "Enter  Continue with Personal data and Injection warnings",
        ) &&
        screen.text.includes(
          "vSR calibrate: Personal data and Injection warnings",
        ) &&
        screen.text.includes("Overall: keep the current lines."),
      screen.text.slice(-6000),
    );
    assert.equal(
      (screen.last.match(/Aim: balanced/g) ?? []).length,
      1,
      "one goal for both",
    );
  },
);

test(
  "a calibration answer can be changed: ← goes back to it, and not sure takes the saved answer back",
  { timeout: 300000 },
  async (t) => {
    const native = await fixture(t);
    let last = await native.run("OUTPUT_CASE READ");
    for (let i = 0; i < 3; i++) last = await native.run("OUTPUT_CASE READ");
    const screen = await native.terminal(
      last.sessionID,
      [
        { at: 12000, keys: "/vsr-calibrate" },
        { at: 12800, keys: "\r" },
        { at: 14000, keys: "Injection" },
        { at: 14600, keys: "\r" },
        { at: 16000, keys: "\r" }, // The report: continue.
        { at: 17500, keys: "\r" }, // First question: yes.
        { at: 18700, keys: "\x1b[D" }, // ← back to it.
        { at: 19900, keys: "3" }, // Changed to not sure: the yes is taken back.
        { at: 21100, keys: "\x1b[D" }, // ← back again,
        { at: 22300, keys: "\x1b[C" }, // → on, keeping it,
        { at: 23500, keys: "\x1b[D" }, // and ← back, where the run ends.
      ],
      26,
    );
    assert.ok(
      screen.text.includes("← back") &&
        screen.text.includes("→ next") &&
        screen.text.includes("Injection warnings · 1 of 8 answers"),
      screen.text.slice(-6000),
    );
    // The line moved back with ←: the run ends on the first question, its not sure marked.
    assert.match(screen.last, /question 1 of 2/, screen.last);
    assert.match(screen.last, /Not sure \(skip\) ✓ your answer/, screen.last);
    assert.match(
      screen.last,
      /Injection warnings · 0 of 8 answers/,
      "the yes was taken back",
    );
    const after = (await native.rpc(
      "calibrationReport",
      {},
    )) as CalibrationReport;
    assert.equal(after.apps.find((item) => item.app === "shield")!.answered, 0);
  },
);
