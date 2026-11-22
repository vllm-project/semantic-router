import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import {
  impliedLabels,
  MAX_STEP,
  sample,
  suggest,
} from "../src/calibration.ts";
import type { Goal } from "../src/calibration.ts";
import { Calibrated, thresholdsOf } from "../src/calibrated.ts";
import { DecisionStore, subjectOf } from "../src/decisions.ts";
import { DecisionLog, KEPT_LINES, labelOf, lineOf } from "../src/log.ts";
import type { LogLine, LogStore } from "../src/log.ts";
import { parseSettings } from "../src/settings.ts";
import type { App } from "../src/settings.ts";
import {
  calibrationQuestion,
  calibrationReportText,
  listed,
  suggestionText,
  suggestionsText,
} from "../src/view.ts";
import type { CalibrationReport } from "../src/view.ts";

const settings = () =>
  parseSettings({
    routing: {
      tiers: [
        { model: "p/light" },
        { model: "p/middle" },
        { model: "p/strong" },
      ],
    },
  });
function memory() {
  const values = new Map<string, unknown>();
  const store: LogStore = {
    get: async (key) => structuredClone(values.get(key)),
    set: async (key, value) => {
      values.set(key, structuredClone(value));
    },
    remove: async (key) => {
      values.delete(key);
    },
    scan: async ({ prefix, after, limit = 1000 }) => {
      const keys = [...values.keys()]
        .filter((key) => key.startsWith(prefix) && (!after || key > after))
        .sort();
      const page = keys.slice(0, limit);
      return {
        entries: page.map((key) => ({ key, value: values.get(key) })),
        ...(keys.length > limit ? { next: page.at(-1)! } : {}),
      };
    },
  };
  return { values, store };
}
let clock = 1_000;
const line = (app: App, values: Partial<LogLine>): LogLine => ({
  id: crypto.randomUUID(),
  at: clock++,
  session: "s",
  app,
  outcome: "allowed",
  subject: "x",
  ...values,
});

test("a subject is one line, hides fixed secrets and keeps at most 200 characters", () => {
  assert.equal(
    subjectOf(
      "export KEY=sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n  && run",
    ),
    "export KEY=[vsr:secret] && run",
  );
  const long = subjectOf("x".repeat(198) + "😀 tail");
  assert.equal(long.length, 199);
  assert.ok(long.endsWith("x…"), "a cut never splits a surrogate pair");
});

test("only the model's scored calls are learnable, each with the thresholds in use", () => {
  const config = settings();
  const store = new DecisionStore(4, 50);
  const of = (input: Parameters<DecisionStore["record"]>[0]) =>
    lineOf(store.record(input), config);
  const base = { sessionID: "s", subject: "git push --force" };
  assert.deepEqual(
    of({
      ...base,
      app: "guard",
      outcome: "held",
      reason: "model-risk",
      score: 0.95,
    })?.thresholds,
    [0.9],
  );
  assert.equal(
    of({
      ...base,
      app: "guard",
      outcome: "held",
      reason: "fixed-rule",
      score: 0.2,
    }),
    undefined,
    "a fixed rule held it, not the score",
  );
  assert.equal(
    of({
      ...base,
      app: "shield",
      outcome: "skipped",
      reason: "timeout",
      score: 0.5,
    }),
    undefined,
  );
  assert.equal(
    of({ sessionID: "s", app: "check", outcome: "verified", score: 0.1 }),
    undefined,
    "no subject",
  );
  const escalated = of({
    ...base,
    app: "escalate",
    outcome: "escalated",
    score: 0.9,
    features: [0.9, 0.2],
  });
  assert.deepEqual(escalated?.scores, [0.9, 0.2]);
  assert.deepEqual(escalated?.thresholds, [0.8, 0.75]);
  const routed = of({
    ...base,
    app: "router",
    outcome: "routed",
    score: 0.3,
    confidence: 0.8,
    model: "p/light#low",
  });
  assert.equal(routed?.tier, 0);
  assert.deepEqual(routed?.thresholds, [0.5, 1.5, 0.3]);
  assert.equal(
    of({
      ...base,
      app: "router",
      outcome: "routed",
      score: 0.3,
      confidence: 0.8,
      model: "q/other",
    }),
    undefined,
    "a model outside the tiers has no tier to learn",
  );
  assert.equal(labelOf({ app: "guard", outcome: "held" }, "right"), "yes");
  assert.equal(labelOf({ app: "guard", outcome: "allowed" }, "right"), "no");
  assert.equal(labelOf({ app: "shield", outcome: "shielded" }, "wrong"), "no");
  assert.equal(
    labelOf({ app: "router", outcome: "routed" }, "needed-stronger"),
    "too-low",
  );
  assert.equal(
    labelOf({ app: "router", outcome: "routed" }, "wrong"),
    undefined,
  );
});

test("the log outlives a restart with its labels, keeps the newest lines and follows settled holds", async () => {
  const { values, store } = memory();
  const config = settings();
  const decisions = new DecisionStore(4, 50);
  const log = new DecisionLog(store, config);
  decisions.sink = log;
  const held = decisions.record({
    sessionID: "s",
    app: "guard",
    outcome: "held",
    reason: "model-risk",
    score: 0.93,
    subject: "git push --force origin main",
  });
  decisions.label("s", held.id, "right");
  decisions.settleGuard("s", held.id, "declined");
  await log.flushed();
  assert.equal(values.size, 1);

  const reloaded = new DecisionLog(store, config);
  await reloaded.load();
  const [kept] = reloaded.all();
  assert.equal(kept?.subject, "git push --force origin main");
  assert.equal(kept?.label, "yes");
  assert.equal(kept?.outcome, "declined");
  // An answer taken back leaves the line unanswered, also after a reload.
  assert.equal(reloaded.unlabel(kept!.id), true);
  assert.equal(reloaded.unlabel(kept!.id), false, "nothing left to take back");
  await reloaded.flushed();
  const again = new DecisionLog(store, config);
  await again.load();
  assert.equal(again.all()[0]?.label, undefined);
  assert.equal(again.label(kept!.id, "yes"), true);
  await again.flushed();

  for (let i = 0; i < KEPT_LINES + 2; i++)
    reloaded.recorded({
      id: `d${i}`,
      sessionID: "s",
      app: "shield",
      outcome: "allowed",
      at: 2_000 + i,
      score: 0.1,
      subject: "read a.md",
    });
  await reloaded.flushed();
  assert.equal(reloaded.all().length, KEPT_LINES);
  assert.equal(values.size, KEPT_LINES);
  assert.ok(
    !reloaded.all().some((item) => item.id === held.id),
    "oldest goes first",
  );
  assert.equal(reloaded.countSince(2_000 + KEPT_LINES), 1);
});

test("a declined hold and a light route escalated within 15 minutes imply labels", () => {
  const route = line("router", {
    outcome: "routed",
    level: 0.4,
    tier: 0,
    at: 0,
  });
  const late = line("router", {
    outcome: "routed",
    level: 0.4,
    tier: 1,
    session: "t",
    at: 0,
  });
  const strong = line("router", {
    outcome: "routed",
    level: 1.8,
    tier: 2,
    session: "u",
    at: 0,
  });
  const lines = [
    line("guard", { outcome: "declined", scores: [0.5] }),
    route,
    late,
    strong,
    line("escalate", { outcome: "escalated", at: 10 * 60_000 }),
    line("escalate", { outcome: "escalated", session: "t", at: 16 * 60_000 }),
    line("escalate", { outcome: "escalated", session: "u", at: 60_000 }),
  ];
  const implied = impliedLabels(lines);
  assert.equal(implied.get(lines[0]!.id), "yes");
  assert.equal(implied.get(route.id), "too-low");
  assert.equal(implied.has(late.id), false, "past the window");
  assert.equal(implied.has(strong.id), false, "nothing stronger to go to");
});

test("questions go to the unanswered decisions nearest the line, on both sides", () => {
  const lines = [0.1, 0.5, 0.85, 0.88, 0.92, 0.99].map((score) =>
    line("guard", {
      scores: [score],
      outcome: score >= 0.9 ? "held" : "allowed",
    }),
  );
  lines[3]!.label = "no";
  const picked = sample(lines, "guard", [0.9], 2).map(
    (item) => item.scores![0],
  );
  assert.deepEqual(picked, [0.92, 0.99, 0.85, 0.5]);
  const next = sample(
    lines,
    "guard",
    [0.9],
    2,
    new Set([lines[2]!.id, lines[4]!.id]),
  ).map((item) => item.scores![0]);
  assert.deepEqual(
    next,
    [0.99, 0.5, 0.1],
    "a skipped question is not asked again this run",
  );
});

test("a fit needs 8 answers, moves each line at most 0.15 and escalate only rises", () => {
  const guard = (score: number, label: "yes" | "no") =>
    line("guard", { scores: [score], label });
  const few = Array.from({ length: 7 }, () => guard(0.6, "yes"));
  assert.equal(suggest(few, "guard", [0.9]).changed, false);
  const many = [
    ...[0.76, 0.8, 0.82, 0.85].map((score) => guard(score, "yes")),
    ...[0.1, 0.2, 0.3, 0.35].map((score) => guard(score, "no")),
  ];
  const fit = suggest(many, "guard", [0.9]);
  assert.ok(fit.changed);
  assert.equal(
    fit.after[0],
    0.76,
    "the separating line nearest the current one",
  );
  assert.ok(fit.agreeAfter.agree > fit.agreeBefore.agree);
  const escalations = [
    ...[0.71, 0.72, 0.73, 0.74].map((score) =>
      line("escalate", {
        outcome: "escalated",
        scores: [score, 0],
        label: "no",
      }),
    ),
    ...[0.9, 0.92, 0.95, 0.97].map((score) =>
      line("escalate", {
        outcome: "escalated",
        scores: [score, 0],
        label: "yes",
      }),
    ),
  ];
  const raised = suggest(escalations, "escalate", [0.7, 0.8]);
  assert.ok(
    raised.after[0]! > 0.7 && raised.after[0]! - 0.7 <= MAX_STEP + 1e-9,
  );
  // Pushback under the line that should have escalated: a lower line would catch it, but escalate only rises.
  const missed = [
    ...[0.55, 0.58, 0.6, 0.62].map((score) =>
      line("escalate", {
        outcome: "escalated",
        scores: [score, 0],
        label: "yes",
      }),
    ),
    ...[0.1, 0.15, 0.2, 0.25].map((score) =>
      line("escalate", {
        outcome: "escalated",
        scores: [score, 0],
        label: "no",
      }),
    ),
  ];
  assert.deepEqual(suggest(missed, "escalate", [0.7, 0.8]).after, [0.7, 0.8]);
  const asGuard = missed.map((item) => ({
    ...item,
    app: "guard" as App,
    scores: [item.scores![0]!],
  }));
  assert.ok(
    suggest(asGuard, "guard", [0.7]).after[0]! < 0.7,
    "the same answers lower a line that may fall",
  );
});

test("fits match the Claude Code plugin's calibrate scripts on the same answers", () => {
  const golden = JSON.parse(
    readFileSync(
      new URL("./fixtures/calibration-golden.json", import.meta.url),
      "utf8",
    ),
  ) as {
    kind: App;
    goal: Goal;
    current: number[];
    items: {
      scores?: number[];
      level?: number;
      confidence?: number;
      tier?: number;
      label: string;
    }[];
    python: number[];
  }[];
  for (const item of golden) {
    const lines = item.items.map((value) =>
      line(item.kind, { ...value, label: value.label as LogLine["label"] }),
    );
    assert.deepEqual(
      suggest(lines, item.kind, item.current, item.goal).after.map(
        (value) => Math.round(value * 100) / 100,
      ),
      item.python,
      `${item.kind} ${item.goal}`,
    );
  }
});

test("calibrated thresholds apply over the options, outlive a restart and give way to an edited option", async () => {
  const values = new Map<string, unknown>();
  const store = {
    get: async (key: string) => structuredClone(values.get(key)),
    set: async (key: string, value: unknown) => {
      values.set(key, structuredClone(value));
    },
  };
  const config = settings();
  const calibrated = new Calibrated(config, store);
  await calibrated.load();
  await calibrated.apply("guard", [0.8], "vela3-lite", 5);
  await calibrated.apply("router", [0.7, 1.4, 0.45], "vela3-fast", 6);
  assert.equal(config.guard.reviewThreshold, 0.8);
  assert.deepEqual(thresholdsOf(config).router, [0.7, 1.4, 0.45]);
  await assert.rejects(calibrated.apply("router", [1.5, 1.4, 0.5]));

  const restarted = settings();
  const again = new Calibrated(restarted, store);
  await again.load();
  assert.equal(restarted.guard.reviewThreshold, 0.8);
  assert.deepEqual(
    again.state(),
    { at: 6, model: "vela3-fast", apps: ["router", "guard"], manual: [] },
    "the decision model of the last calibration is kept, for the probe",
  );

  // A threshold set in /vsr-config is kept the same way, named as the person's, and a reset brings the option back.
  await again.set("shield", [0.8]);
  assert.equal(restarted.shield.injectionThreshold, 0.8);
  assert.deepEqual(again.state().manual, ["shield"]);
  assert.deepEqual(
    again.state().apps,
    ["router", "guard"],
    "not a calibration",
  );
  const third = parseSettings({});
  await new Calibrated(third, store).load();
  assert.equal(third.shield.injectionThreshold, 0.8);
  await again.reset("shield");
  assert.equal(restarted.shield.injectionThreshold, 0.9);
  assert.deepEqual(again.state().manual, []);
  await again.set("guard", [0.85]);
  await again.apply("guard", [0.82], "vela3-fast", 7);
  assert.deepEqual(
    again.state().manual,
    [],
    "a calibration after it fits that app again",
  );

  const edited = parseSettings({ guard: { reviewThreshold: 0.95 } });
  await new Calibrated(edited, store).load();
  assert.equal(
    edited.guard.reviewThreshold,
    0.95,
    "the option the person set wins",
  );
});

test("questions and suggestions read in plain words", () => {
  const asked = calibrationQuestion({
    id: "1",
    app: "guard",
    subject: "git push origin main",
    acted: false,
    scores: [0.76],
    thresholds: [0.9],
  });
  assert.equal(
    asked.title,
    "Should vSR have asked you before this command ran?",
  );
  assert.equal(
    asked.description,
    "git push origin main · it let it run; score 0.76 (line 0.90)",
  );
  assert.deepEqual(
    asked.options.map((option) => option.value),
    ["yes", "no", "skip", "stop"],
  );
  const routed = calibrationQuestion({
    id: "2",
    app: "router",
    subject: "List the files",
    acted: false,
    level: 0.03,
    tier: 0,
    thresholds: [0.65, 1.5, 0.5],
  });
  assert.match(routed.title, /the light model/);
});

test("a suggestion says what each line does, what it would do on the log, and what to do, as Claude Code's does", () => {
  const guard = (score: number, label?: "yes" | "no") =>
    line("guard", { scores: [score], ...(label ? { label } : {}) });
  const lines = [
    ...[0.76, 0.8, 0.82, 0.85].map((score) => guard(score, "yes")),
    ...[0.1, 0.2, 0.3, 0.35].map((score) => guard(score, "no")),
    guard(0.95),
    guard(0.05),
  ];
  const found = suggest(lines, "guard", [0.9]);
  assert.equal(found.decisions, 10);
  assert.deepEqual(found.acted, { before: 1, after: 5 });
  assert.equal(found.recommend, "apply");
  const text = suggestionText(found, "balanced");
  assert.match(text, /^Aim: balanced: as few wrong calls/);
  assert.match(
    text,
    /Review line 0\.90 → 0\.76: vSR asks you before a command it rates at least this risky\./,
  );
  assert.match(
    text,
    /On your log: vSR would hold 5 of these 10 commands for you instead of 1\./,
  );
  assert.match(
    text,
    /Your answers: 4 of 8 agree with vSR now, 8 of 8 would after\./,
  );
  assert.match(text, /Recommended: apply$/);

  const few = suggest(lines.slice(0, 7), "guard", [0.9]);
  assert.equal(few.recommend, "more");
  assert.match(
    suggestionText(few),
    /Review line stays 0\.90: [^\n]+\n\nNot enough answers yet: 7 of 8 \(answer more in \/vsr or here\)\.$/,
  );

  // The router says where its logged subagents would go.
  const route = (level: number, tier: number, label: "too-low" | "right") =>
    line("router", { outcome: "routed", level, tier, confidence: 0.9, label });
  const routes = [
    ...[0.55, 0.58, 0.6, 0.62].map((level) => route(level, 0, "too-low")),
    ...[0.1, 0.2, 0.3, 0.4].map((level) => route(level, 0, "right")),
  ];
  const routed = suggest(routes, "router", [0.65, 1.5, 0.5]);
  assert.equal(routed.recommend, "apply");
  assert.deepEqual(routed.moves, [{ from: "light", to: "medium", n: 4 }]);
  assert.match(
    suggestionText(routed, "balanced"),
    /^Light line 0\.65 → 0\.\d\d: [^\n]+\n[^\n]+\n[^\n]+\n\nOn your log: 4 tasks that went to the light model would go to the medium model\./,
  );
});

test("several apps' suggestions read as Claude Code's: the aim once, each app under its name, then what to apply", () => {
  const guard = (score: number, label: "yes" | "no") =>
    line("guard", { scores: [score], label });
  const lines = [
    ...[0.76, 0.8, 0.82, 0.85].map((score) => guard(score, "yes")),
    ...[0.1, 0.2, 0.3, 0.35].map((score) => guard(score, "no")),
    line("shield", { scores: [0.5], label: "yes" }),
  ];
  const both = [
    suggest(lines, "guard", [0.9]),
    suggest(lines, "shield", [0.9]),
  ];
  const text = suggestionsText(both, "balanced");
  assert.match(
    text,
    /^Aim: balanced: [^\n]+\.\n\nRisky commands \(asks you before a risky command runs\)\n  Review line 0\.90 → 0\.76: /,
  );
  assert.equal(text.match(/Aim:/g)?.length, 1, "the aim is said once");
  assert.match(
    text,
    /\n\nInjection warnings \(warns the agent when a file or page tries to instruct it\)\n  Warning line stays 0\.90: /,
  );
  assert.match(text, /  Not enough answers yet: 1 of 8 /);
  assert.match(
    text,
    /\nOverall: apply the changes for Risky commands; keep the rest\.$/,
  );
  assert.match(
    suggestionsText([both[1]!]),
    /^Warning line stays/,
    "one app reads as before",
  );
  assert.match(
    suggestionsText([both[1]!, both[1]!]),
    /Overall: keep the current lines\.$/,
  );
  assert.equal(
    listed(["Guard", "Shield", "Redact"]),
    "Guard, Shield and Redact",
  );
});

test("the report says what the log holds for the chosen apps, as Claude Code's step 2, and which answers it counts", () => {
  const app = (
    name: CalibrationReport["apps"][number]["app"],
    extra: Partial<CalibrationReport["apps"][number]>,
  ): CalibrationReport["apps"][number] => ({
    app: name,
    enabled: true,
    decisions: 0,
    answered: 0,
    acted: 0,
    explicit: 0,
    implied: 0,
    thresholds: [0.9],
    calibrated: false,
    ...extra,
  });
  const found: CalibrationReport = {
    persistence: "durable",
    since: 0,
    remind: false,
    model: "vela3-lite",
    first: new Date(2026, 10, 17, 9).getTime(),
    last: new Date(2026, 10, 19, 18).getTime(),
    apps: [
      app("guard", {
        decisions: 25,
        acted: 1,
        answered: 5,
        explicit: 4,
        implied: 1,
      }),
      app("router", {
        decisions: 6,
        routes: [3, 2, 0, 1],
        thresholds: [0.65, 1.5, 0.5],
      }),
      app("shield", { decisions: 4, acted: 2, enabled: false }),
      app("check", {}),
    ],
  };
  const text = calibrationReportText(found, [
    "guard",
    "router",
    "shield",
    "check",
  ]);
  assert.match(
    text,
    /^vSR's log holds 35 decisions from 2026-11-17 to 2026-11-19\.\nDecision model now: vela3-lite; not calibrated before\./,
  );
  assert.match(
    text,
    /\n\nRisky commands \(asks you before a risky command runs\)\n  25 commands: 1 held for you, 24 allowed\. You answered 4 \(\+1 your declines imply\); 8 are needed to tune\./,
  );
  assert.match(
    text,
    /\n  6 subagents: 3 went to the light model, 2 to medium, 0 to strong, 1 kept its own model\. You answered 0; /,
  );
  assert.match(
    text,
    /Injection warnings \([^)]+\) — off in your settings\n  4 files or pages read: the agent was warned about 2, 2 passed\./,
  );
  assert.match(text, /Answer checks \([^)]+\)\n  No decisions logged yet\./);
  assert.match(
    text,
    /The log already counts some answers for you: a held command you declined counts as "should ask", and pushback within 15 minutes of a light or medium route counts as "needed stronger"\.$/,
  );
  assert.doesNotMatch(
    calibrationReportText({ ...found, at: 1, calibratedModel: "vela3-fast" }, [
      "shield",
    ]),
    /already counts/,
    "said only when router or guard is chosen",
  );
  assert.match(
    calibrationReportText(
      {
        ...found,
        at: new Date(2026, 10, 18).getTime(),
        calibratedModel: "vela3-fast",
      },
      ["shield"],
    ),
    /last calibrated 2026-11-18 with vela3-fast\./,
  );
});
