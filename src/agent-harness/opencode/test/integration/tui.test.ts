import { appendFile } from "node:fs/promises";
import { join } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import assert from "node:assert/strict";
import { test } from "node:test";
import { fixture } from "./output-fixture.ts";
import type { ViewSnapshot } from "../../src/view.ts";

test(
  "packed native TUI renders decisions, slash app/wait controls and panel answers change authoritative state",
  { timeout: 60000 },
  async (t) => {
    const native = await fixture(t);
    const first = await native.run("OUTPUT_CASE READ");
    const generations = native.requests.length,
      classifications = native.states.length;
    const screen = await native.terminal(
      first.sessionID,
      [
        { at: 2500, keys: "/vsr" },
        { at: 3000, keys: "\r" }, // The panel.
        { at: 3500, keys: "\u0018" }, // ctrl+x ←: back from the panel to the prompt.
        { at: 3700, keys: "\u001b[D" },
        { at: 4000, keys: "/vsr-config" },
        { at: 4500, keys: "\r" },
        { at: 5000, keys: "\u001b[B\r" }, // Check is the second row,
        { at: 5600, keys: "\u001b[B".repeat(5) + "\r" }, // redact waits: 3s → until done,
        { at: 6200, keys: "\u001b" }, // and Esc closes the settings.
        { at: 8500, keys: "\u0018" }, // ctrl+x →: into the panel.
        { at: 8700, keys: "\u001b[C" },
        { at: 9200, keys: "\u001b[B" }, // Picks the newest decision.
        { at: 9600, keys: "2" }, // Its second answer: wrong call.
        { at: 10200, keys: "\u0018" },
        { at: 10400, keys: "\u001b[D" },
        { at: 11000, keys: "/vsr-config" },
        { at: 11500, keys: "\r" },
        { at: 12000, keys: "\u001b[A\r" }, // Its last row checks the runtime.
        { at: 13500, keys: "\r" }, // Close the report,
        { at: 14200, keys: "\u001b" }, // and the settings it reopens.
        { at: 16000, keys: "/vsr" },
        { at: 16500, keys: "\r" },
        { at: 17500, resize: [46, 161] },
      ],
      18.5,
    );
    assert.ok(
      screen.text.includes("vSR decision runtime") &&
        screen.text.includes("Runtime: from the plugin options"),
      "/vsr-config checks the runtime: " + screen.text.slice(-8000),
    );
    assert.ok(screen.text.includes("vSR decisions"), screen.text.slice(-8000));
    assert.ok(screen.text.includes("subagents "), screen.text.slice(-8000));
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
      screen.text.includes("runtime reachable") &&
        /── apps ─+ click to change ──/.test(screen.text) &&
        /[●○] redact +wait +1s +2s +3s +until done/.test(screen.text) &&
        /[●○] router +goal +balanced +quality +save/.test(screen.text) &&
        /── this session ─+/.test(screen.text) &&
        /── decisions ─+ newest first ──/.test(screen.text),
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
    // An answer not given is a bracketed button; the given one is filled, without brackets.
    assert.ok(
      /\[1 right call\] {1,2}2 wrong call /.test(screen.text),
      "the answer given is filled in the panel: " + screen.text.slice(-8000),
    );
    assert.equal(
      native.requests.length,
      generations,
      "UI rendering and controls must not generate model requests",
    );
    const asked = native.states.slice(classifications);
    assert.ok(asked.length > 0, "the runtime check asks the runtime");
    assert.ok(
      asked.every((state) =>
        ["Ticket #1042", "Acme Payments SDK", "No, task 2 is wrong"].some(
          (sample) => state.includes(sample),
        ),
      ),
      "only the runtime check's cases are classified, never the rendering: " +
        JSON.stringify(asked),
    );
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
        { at: 12000, keys: "/vsr-config" },
        { at: 12500, keys: "\r" },
        { at: 13000, keys: "\u001b[B\r" },
        { at: 13400, keys: "\u001b" },
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
    assert.deepEqual(
      after.decisions,
      before.decisions,
      "the panel shows the decisions made before the reload and restart",
    );
    assert.ok(
      /REDACTED .*Redact/.test(screen.last),
      "the reopened panel draws them: " + screen.last,
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
    // The theme's tokens resolve: a badge is drawn on its own color, not on the panel's.
    const badge =
      /48;2;([\d;]+)m REDACTED \x1b\[0;38;2;[\d;]+;48;2;([\d;]+)m/.exec(
        open.ansi,
      );
    assert.ok(badge && badge[1] !== badge[2], JSON.stringify(badge?.[0]));
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

test(
  "the newest decision shows above the prompt and the session's decisions in the sidebar",
  { timeout: 60000 },
  async (t) => {
    const native = await fixture(t);
    const run = await native.run("OUTPUT_CASE READ");
    const screen = await native.terminal(
      run.sessionID,
      [
        { at: 2500, keys: "\u0010" }, // ctrl+p: the command palette.
        { at: 3200, keys: "Show sidebar" },
        { at: 4000, keys: "\r" },
        { at: 4500, keys: "\u001b" },
      ],
      6,
    );
    const composer = screen.last
      .split("\n")
      .find((line) => /^\s*vSR +[A-Z]+ /.test(line));
    assert.match(
      composer ?? "",
      /vSR +(REDACTED|SHIELDED) +[▒◆] read …\/fixture\.txt .*· 2 decisions · \/vsr/,
      screen.last,
    );
    assert.match(screen.last, /▝██▍▐██▏/, screen.last);
    assert.match(
      screen.last,
      /vLLM Semantic Router[^\n]*\n(?:[^\n]*\n){0,3}[^\n]*2 decisions · \/vsr/,
      screen.last,
    );
    assert.match(screen.last, /▒ redacted +read …\/fixture\.txt/, screen.last);
    assert.match(screen.last, /◆ shielded +read …\/fixture\.txt/, screen.last);
  },
);

test(
  "a subagent's decisions are listed in its parent's panel, and a click on an answer saves it",
  { timeout: 90000 },
  async (t) => {
    const native = await fixture(t);
    const run = await native.run("OUTPUT_CASE SUBAGENT");
    const open = [
      { at: 2000, keys: "/vsr" },
      { at: 2500, keys: "\r" },
    ];
    const shown = await native.terminal(run.sessionID, open, 5);
    const lines = shown.last.split("\n");
    const row = lines.findIndex((line) => line.includes("[2 wrong call]"));
    assert.ok(row >= 0, shown.last);
    assert.equal(
      lines.filter((line) => line.includes("· in a subagent")).length,
      2,
      shown.last,
    );
    // An SGR mouse press and release on the first decision's "wrong call".
    const at = `${lines[row]!.indexOf("[2 wrong call]") + 4};${row + 1}`;
    const answered = await native.terminal(
      run.sessionID,
      [
        ...open,
        { at: 3500, keys: `\u001b[<0;${at}M` },
        { at: 3800, keys: `\u001b[<0;${at}m` },
      ],
      5,
    );
    assert.doesNotMatch(
      answered.last,
      /answer {3,}\S/,
      "an answer does not open the decision's card: " + answered.last,
    );
    const snapshot = (await native.rpc("snapshot", {
      sessionID: run.sessionID,
    })) as ViewSnapshot;
    const newest = snapshot.decisions.at(-1)!;
    assert.notEqual(newest.sessionID, run.sessionID);
    assert.deepEqual(snapshot.feedback, { [newest.id]: "wrong" });
  },
);

test(
  "a click or Enter opens a decision's card, Esc closes only the card, and the sidebar says why a decision needs a look",
  { timeout: 90000 },
  async (t) => {
    const native = await fixture(t);
    const run = await native.run("OUTPUT_CASE SUBAGENT");
    const open = [
      { at: 2000, keys: "/vsr" },
      { at: 2500, keys: "\r" },
    ];
    const plain = await native.terminal(run.sessionID, open, 4);
    const lines = plain.last.split("\n");
    const row = lines.findIndex((line) => line.includes("REDACTED"));
    assert.ok(row >= 0, plain.last);
    // The end of the decision's subject, right of where the card opens, so the release lands outside the card.
    const at = `${lines[row + 1]!.indexOf("fixture.txt") + 8};${row + 2}`;
    // A press, then a release a moment later, as a hand clicks: the card must still be open after the release.
    const clicked = await native.terminal(
      run.sessionID,
      [
        ...open,
        { at: 3500, keys: `\u001b[<0;${at}M` },
        { at: 3800, keys: `\u001b[<0;${at}m` },
      ],
      5,
    );
    assert.match(clicked.last, /hid +secret ×1 · email ×1/, clicked.last);
    assert.match(clicked.last, /session +a subagent's, ses_/, clicked.last);
    // A hand's click drags a little and selects text; one Esc must still close the card.
    const [column, line] = at.split(";").map(Number);
    const dragged = await native.terminal(
      run.sessionID,
      [
        ...open,
        { at: 3500, keys: `\u001b[<0;${at}M` },
        { at: 3600, keys: `\u001b[<32;${column! + 3};${line}M` },
        { at: 3800, keys: `\u001b[<0;${column! + 3};${line}m` },
        { at: 4600, keys: "\u001b" },
      ],
      6,
    );
    assert.match(dragged.text, /hid {3,}secret/, dragged.text.slice(-8000));
    assert.doesNotMatch(dragged.last, /hid {3,}secret/, dragged.last);
    assert.ok(dragged.last.includes("↑↓ pick"), "the panel stays open");
    const closed = await native.terminal(
      run.sessionID,
      [
        ...open,
        { at: 3200, keys: "\u001b[B" }, // The newest, shield's.
        { at: 3700, keys: "\r" },
        { at: 4500, keys: "\u001b" },
      ],
      6,
    );
    assert.match(closed.text, /answer +none yet/, closed.text.slice(-8000));
    assert.doesNotMatch(closed.last, /answer +none yet/, closed.last);
    assert.ok(closed.last.includes("↑↓ pick"), "the panel stays open");

    const image = await native.run("OUTPUT_CASE IMAGE");
    const sidebar = await native.terminal(
      image.sessionID,
      [
        { at: 2500, keys: "\u0010" }, // ctrl+p: the command palette.
        { at: 3200, keys: "Show sidebar" },
        { at: 4000, keys: "\r" },
        { at: 4500, keys: "\u001b" },
      ],
      6,
    );
    assert.match(
      sidebar.last,
      /◆ partial +read …\/fixture\.png\s+↳ unsupported · 0\.01 < 0\.90/,
      sidebar.last,
    );
  },
);

test(
  "the panel's controls are clicked as the Claude Code plugin's: an app turns on or off, a wait and a goal are picked",
  { timeout: 90000 },
  async (t) => {
    const native = await fixture(t);
    const run = await native.run("OUTPUT_CASE READ");
    const open = [
      { at: 2000, keys: "/vsr" },
      { at: 2500, keys: "\r" },
    ];
    const shown = await native.terminal(run.sessionID, open, 5);
    const lines = shown.last.split("\n");
    // The panel fills the column OpenCode gives it, down beside the prompt and status, leaving no gap under it.
    const bottom = lines.findLastIndex((line) => line.trim() !== "");
    assert.match(lines[bottom]!, /╰─+/, shown.last);
    // An SGR mouse press and release on a control's text.
    const click = (
      text: string,
      line: (row: string) => boolean,
      at: number,
    ) => {
      const row = lines.findIndex(line);
      assert.ok(row >= 0, shown.last);
      const where = `${lines[row]!.indexOf(text) + 2};${row + 1}`;
      return [
        { at, keys: `\u001b[<0;${where}M` },
        { at: at + 300, keys: `\u001b[<0;${where}m` },
      ];
    };
    const clicked = await native.terminal(
      run.sessionID,
      [
        ...open,
        ...click("○ check", (row) => /○ check/.test(row), 3500),
        ...click("1s", (row) => /[●○] shield +wait/.test(row), 4500),
        ...click("save", (row) => /[●○] router +goal/.test(row), 5500),
        // The runtime line checks the runtime, as /vsr-config's last row does.
        ...click("runtime", (row) => /runtime reachable/.test(row), 6500),
      ],
      9,
    );
    const snapshot = (await native.rpc("snapshot", {
      sessionID: run.sessionID,
    })) as ViewSnapshot;
    assert.equal(snapshot.apps.check, true, clicked.last);
    assert.equal(snapshot.waits.shield, 1000, clicked.last);
    assert.equal(snapshot.controls.goal, "save", clicked.last);
    // Before the runtime check's dialog, which covers the check row at the foot of the apps.
    assert.match(
      clicked.text,
      /● check/,
      "the switch shows it on: " + clicked.text.slice(-8000),
    );
    assert.ok(
      clicked.text.includes("vSR decision runtime") &&
        clicked.text.includes("Runtime: from the plugin options"),
      "a click on the runtime line checks it: " + clicked.text.slice(-4000),
    );
  },
);

test(
  "/vsr-config changes every setting in place, as the Claude Code plugin's /config, and r brings the options' thresholds back",
  { timeout: 120000 },
  async (t) => {
    const native = await fixture(t);
    const run = await native.run("OUTPUT_CASE READ");
    const down = (n: number) => "\u001b[B".repeat(n);
    const screen = await native.terminal(
      run.sessionID,
      [
        { at: 2500, keys: "/vsr-config" },
        { at: 3000, keys: "\r" },
        { at: 4000, keys: down(1) + "\r" }, // check: on
        { at: 4600, keys: down(3) + "\r" }, // router: on
        { at: 5200, keys: down(3) + "\r" }, // shield waits: 2s → 3s
        { at: 6000, keys: down(4) + "\r" }, // guard · review line: asked for
        { at: 7000, keys: "\u007f".repeat(6) + "0.8" },
        { at: 7600, keys: "\r" },
        { at: 8600, keys: down(7) + "\r" }, // router · smaller-model line
        { at: 9400, keys: "\u007f".repeat(6) + "0.7" },
        { at: 10000, keys: "\r" },
        { at: 11000, keys: "\u001b[A".repeat(9) + "\r" }, // router goal: balanced → quality
        // A line that does nothing now is grayed and still changed, as /config changes any row.
        { at: 12000, keys: down(9) + "\r" },
        { at: 12800, keys: "\u007f".repeat(6) + "0.65" },
        { at: 13400, keys: "\r" },
      ],
      15,
    );
    assert.ok(
      screen.text.includes("vSR settings") &&
        /check +on/.test(screen.text) &&
        // Guard is off in the fixture: its line says so, beside whose value it is.
        /guard · review line +0\.80 +not used: guard off · yours · options 0\.90/.test(
          screen.text,
        ) &&
        // The smaller-model line says it is not used once the goal is not balanced.
        /router · smaller-model line +0\.65 +not used: quality never goes smaller/.test(
          screen.last,
        ),
      screen.text.slice(-6000),
    );
    const snapshot = (await native.rpc("snapshot", {
      sessionID: run.sessionID,
    })) as ViewSnapshot;
    assert.equal(snapshot.apps.check, true);
    assert.equal(snapshot.waits.shield, 3000);
    assert.equal(snapshot.apps.router, true);
    assert.equal(snapshot.controls.downgradeConfidence, 0.65);
    assert.equal(snapshot.controls.goal, "quality");
    type Lines = { app: string; values: number[]; source: string }[];
    const guard = async () =>
      ((await native.rpc("thresholds", {})) as Lines).find(
        (line) => line.app === "guard",
      )!;
    assert.deepEqual(await guard(), {
      app: "guard",
      values: [0.8],
      options: [0.9],
      source: "you",
    });
    // r on the line brings the options' threshold back, and it stays after a restart.
    await native.terminal(
      run.sessionID,
      [
        { at: 2500, keys: "/vsr-config" },
        { at: 3000, keys: "\r" },
        { at: 4000, keys: down(11) + "r" },
      ],
      6,
    );
    await native.restart();
    assert.equal((await guard()).source, "options");
    assert.deepEqual((await guard()).values, [0.9]);
  },
);
