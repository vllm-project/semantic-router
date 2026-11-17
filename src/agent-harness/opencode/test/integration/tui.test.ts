import { appendFile } from "node:fs/promises";
import { join } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import assert from "node:assert/strict";
import { test } from "node:test";
import { fixture } from "./output-fixture.ts";
import type { ViewSnapshot } from "../../src/view.ts";

test(
  "packed native TUI renders decisions and keyboard app/wait/feedback controls change authoritative state",
  { timeout: 60000 },
  async (t) => {
    const native = await fixture(t);
    const first = await native.run("OUTPUT_CASE READ");
    await native.api(
      `/api/session/${first.sessionID}`,
      { title: "First fixture session" },
      "PATCH",
    );
    const second = (await native.api("/api/session", {
      title: "Second fixture session",
      model: { providerID: "fixture", id: "middle" },
    })) as { data: { id: string } };
    const generations = native.requests.length,
      classifications = native.states.length;
    const screen = await native.terminal(
      first.sessionID,
      [
        { at: 2500, keys: "/vsr" },
        { at: 3000, keys: "\r" }, // The panel.
        { at: 3500, keys: "\u0018" }, // ctrl+x ←: back from the panel to the prompt.
        { at: 3700, keys: "\u001b[D" },
        { at: 4000, keys: "/vsr-apps" },
        { at: 4500, keys: "\r" },
        { at: 5000, keys: "\u001b[B".repeat(5) + "\r" }, // Check is the final option.
        { at: 6000, keys: "/vsr-waits" },
        { at: 6500, keys: "\r" },
        { at: 7000, keys: "\r" },
        { at: 7500, keys: "\u001b[B\r" }, // Strict is the final option.
        { at: 8500, keys: "/vsr-feedback" },
        { at: 9000, keys: "\r" },
        { at: 9500, keys: "\r" },
        { at: 10000, keys: "\u001b[B\r" }, // Wrong decision.
        { at: 11000, keys: "/vsr-check" },
        { at: 11500, keys: "\r" },
        { at: 13000, keys: "\r" }, // Close its report.
        { at: 14000, keys: "/vsr-sessions" },
        { at: 14500, keys: "\r" },
        { at: 15000, keys: "\u001b[B\r" },
        { at: 16000, keys: "/vsr" },
        { at: 16500, keys: "\r" },
        { at: 17500, resize: [46, 161] },
      ],
      18.5,
    );
    assert.ok(
      screen.text.includes("vSR decision runtime") &&
        screen.text.includes("Runtime: from the plugin options"),
      "/vsr-check reports the runtime check: " + screen.text.slice(-8000),
    );
    assert.ok(screen.text.includes("vSR decisions"), screen.text.slice(-8000));
    assert.ok(
      screen.text.includes("Observed child tokens:"),
      screen.text.slice(-8000),
    );
    assert.ok(
      screen.text.includes("Redact") && screen.text.includes("Shield"),
      screen.text.slice(-8000),
    );
    assert.ok(
      /read \S*fixture\.txt/.test(screen.text),
      "a decision shows its subject, not its call ID: " +
        screen.text.slice(-8000),
    );
    assert.ok(
      screen.text.includes("Runtime: HTTP reachable") &&
        screen.text.includes("Apps: Router"),
      "summary lines must not overlap: " + screen.text.slice(-8000),
    );
    const snapshot = (await native.rpc("snapshot", {
      sessionID: first.sessionID,
    })) as ViewSnapshot;
    assert.equal(snapshot.apps.check, true, screen.text.slice(-8000));
    assert.equal(snapshot.waits.redact, "done", screen.text.slice(-8000));
    assert.ok(
      Object.values(snapshot.feedback).includes("wrong"),
      screen.text.slice(-8000),
    );
    assert.equal(snapshot.controls.persistence, "durable");
    assert.equal(snapshot.runtime.models[0]?.id, "fixture");
    assert.ok(screen.text.includes("Second fixture session"));
    assert.ok(screen.text.includes(second.data.id), screen.text.slice(-8000));
    assert.equal(
      native.requests.length,
      generations,
      "UI rendering and controls must not generate model requests",
    );
    const asked = native.states.slice(classifications);
    assert.ok(asked.length > 0, "/vsr-check asks the runtime");
    assert.ok(
      asked.every((state) =>
        ["Ticket #1042", "Acme Payments SDK", "No, task 2 is wrong"].some(
          (sample) => state.includes(sample),
        ),
      ),
      "only /vsr-check's cases are classified, never the rendering: " +
        JSON.stringify(asked),
    );
    assert.notEqual(second.data.id, first.sessionID);
  },
);

test(
  "open native TUI survives plugin reload and server restart without duplicate controls",
  { timeout: 60000 },
  async (t) => {
    const native = await fixture(t);
    const run = await native.run("OUTPUT_CASE READ");
    const before = (await native.rpc("snapshot", {
      sessionID: run.sessionID,
    })) as ViewSnapshot;
    const generations = native.requests.length,
      classifications = native.states.length;
    const terminal = native.terminal(
      run.sessionID,
      [
        { at: 2500, keys: "/vsr" },
        { at: 3000, keys: "\r" },
        { at: 3500, keys: "\u0018" },
        { at: 3700, keys: "\u001b[D" },
        { at: 12000, keys: "/vsr-apps" },
        { at: 12500, keys: "\r" },
        { at: 13000, keys: "\u001b[B".repeat(5) + "\r" },
        { at: 14500, keys: "/vsr" },
        { at: 15000, keys: "\r" },
        { at: 15500, resize: [46, 161] },
      ],
      16.5,
    );
    await delay(6000);
    await appendFile(
      join(native.root, "package/src/tui.tsx"),
      "\n// Fixture hot reload\n",
    );
    await delay(2000);
    await native.restart();
    const screen = await terminal;
    const after = (await native.rpc("snapshot", {
      sessionID: run.sessionID,
    })) as ViewSnapshot;
    assert.notEqual(after.instanceID, before.instanceID);
    assert.equal(after.apps.check, true, screen.text.slice(-8000));
    assert.equal(
      after.controls.revision,
      1,
      "one command commits exactly once",
    );
    assert.deepEqual(after.decisions, []);
    assert.ok(
      screen.text.includes("No decisions in this session yet."),
      screen.text.slice(-8000),
    );
    assert.equal(native.requests.length, generations);
    assert.equal(native.states.length, classifications);
  },
);

test(
  "/vsr toggles the decisions panel and Esc closes it only from the panel",
  { timeout: 90000 },
  async (t) => {
    const native = await fixture(t);
    const run = await native.run("OUTPUT_CASE READ");
    const open = await native.terminal(
      run.sessionID,
      [
        { at: 2500, keys: "/vsr" },
        { at: 3000, keys: "\r" },
      ],
      5,
    );
    assert.ok(open.last.includes("vSR decisions ·"), open.last);
    const toggled = await native.terminal(
      run.sessionID,
      [
        { at: 2500, keys: "/vsr" },
        { at: 3000, keys: "\r" },
        { at: 3500, keys: "\u0018" }, // ctrl+x ←: back to the prompt.
        { at: 3700, keys: "\u001b[D" },
        { at: 4500, keys: "/vsr" },
        { at: 5000, keys: "\r" },
      ],
      7,
    );
    assert.ok(!toggled.last.includes("vSR decisions ·"), toggled.last);
    const escaped = await native.terminal(
      run.sessionID,
      [
        { at: 2500, keys: "/vsr" },
        { at: 3000, keys: "\r" },
        { at: 4000, keys: "\u001b" },
      ],
      6,
    );
    assert.ok(!escaped.last.includes("vSR decisions ·"), escaped.last);
    const prompt = await native.terminal(
      run.sessionID,
      [
        { at: 2500, keys: "/vsr" },
        { at: 3000, keys: "\r" },
        { at: 3500, keys: "\u0018" },
        { at: 3700, keys: "\u001b[D" },
        { at: 4500, keys: "\u001b" },
      ],
      6.5,
    );
    assert.ok(
      prompt.last.includes("vSR decisions ·"),
      "Esc at the prompt is OpenCode's, not the panel's: " + prompt.last,
    );
  },
);
