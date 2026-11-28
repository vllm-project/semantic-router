import type { Label, LogLine } from "./log.ts";
import type { App } from "./settings.ts";

// Ported from the Claude Code plugin's skills/calibrate/scripts (fitting.py, decisions.py, questions.py): the same
// grid, step limit, label minimum and goals, so both plugins move a threshold the same way on the same answers.
export const MIN_LABELS = 8;
export const MAX_STEP = 0.15;
const GRID = 0.01;
const IMPLIED_WINDOW_MS = 15 * 60_000;
export const GOALS = {
  balanced: 1,
  "fewer-false-alarms": 0.5,
  "catch-more": 2,
} as const;
export type Goal = keyof typeof GOALS;
// Escalate is logged only when it escalated: its log cannot show what a lower line would add, so it only rises.
const RAISE_ONLY: readonly App[] = ["escalate"];

const round = (value: number) => Math.round(value * 100) / 100;
const axis = (low: number, high: number): number[] =>
  Array.from({ length: Math.round((high - low) / GRID) + 1 }, (_, i) =>
    round(low + i * GRID),
  );
function grid(dimensions: number): number[][] {
  let combos: number[][] = [[]];
  for (let i = 0; i < dimensions; i++)
    combos = combos.flatMap((combo) =>
      axis(0.05, 0.99).map((value) => [...combo, value]),
    );
  return combos;
}
const gap = (a: readonly number[], b: readonly number[]) =>
  a.reduce((sum, value, i) => sum + Math.abs(value - b[i]!), 0);
const compare = (a: readonly number[], b: readonly number[]) => {
  for (let i = 0; i < a.length; i++)
    if (Math.abs(a[i]! - b[i]!) > 1e-9) return a[i]! - b[i]!;
  return 0;
};

// The best candidate within MAX_STEP of `currents` on every axis; ties go to the one nearest the best anywhere, so a
// run that cannot reach the best still moves toward it.
function bestToward(
  candidates: number[][],
  currents: number[],
  objective: (candidate: number[]) => number[],
): number[] {
  const all = [...candidates, currents];
  const scored = all.map((candidate) => ({
    candidate,
    score: objective(candidate),
  }));
  const anywhere = scored.reduce((best, item) => {
    const order = compare(item.score, best.score);
    return order < 0 ||
      (order === 0 &&
        gap(item.candidate, currents) < gap(best.candidate, currents))
      ? item
      : best;
  }).candidate;
  return scored
    .filter(({ candidate }) =>
      candidate.every(
        (value, i) => Math.abs(value - currents[i]!) <= MAX_STEP + 1e-9,
      ),
    )
    .reduce((best, item) => {
      const order = compare(item.score, best.score);
      return order < 0 ||
        (order === 0 &&
          gap(item.candidate, anywhere) < gap(best.candidate, anywhere))
        ? item
        : best;
    }).candidate;
}

export const acts = (
  scores: readonly number[],
  thresholds: readonly number[],
) => scores.some((score, i) => score >= thresholds[i]!);
export interface Agreement {
  agree: number;
  total: number;
}
type Case = { scores: number[]; truth: boolean };
function confusion(cases: readonly Case[], thresholds: readonly number[]) {
  let tp = 0,
    fp = 0,
    fn = 0,
    tn = 0;
  for (const { scores, truth } of cases) {
    const predicted = acts(scores, thresholds);
    if (predicted && truth) tp++;
    else if (predicted) fp++;
    else if (truth) fn++;
    else tn++;
  }
  return { tp, fp, fn, tn };
}
function fbeta(tp: number, fp: number, fn: number, beta: number): number {
  const b2 = beta * beta;
  const denominator = (1 + b2) * tp + b2 * fn + fp;
  return denominator === 0 ? 0 : ((1 + b2) * tp) / denominator;
}

export const routeTier = (level: number, easy: number, hard: number) =>
  level < easy ? 0 : level < hard ? 1 : 2;
type RouterCase = { level: number; confidence: number; target: number };

// Labels the log implies: a hold the person declined should have asked, and a route to the lighter two tiers followed
// within 15 minutes by an escalation in the same session was too low.
export function impliedLabels(lines: readonly LogLine[]): Map<string, Label> {
  const found = new Map<string, Label>();
  for (const line of lines)
    if (line.app === "guard" && line.outcome === "declined")
      found.set(line.id, "yes");
  const routes = lines.filter((line) => line.app === "router");
  for (const escalation of lines) {
    if (escalation.app !== "escalate" || escalation.outcome !== "escalated")
      continue;
    const before = routes.filter(
      (route) =>
        route.session === escalation.session &&
        escalation.at - route.at >= 0 &&
        escalation.at - route.at <= IMPLIED_WINDOW_MS,
    );
    const last = before.sort((a, b) => b.at - a.at)[0];
    if (last && (last.tier ?? 2) < 2 && !found.has(last.id))
      found.set(last.id, "too-low");
  }
  return found;
}
// The person's own answer wins over an implied one.
export function labelsOf(lines: readonly LogLine[]): Map<string, Label> {
  const labels = impliedLabels(lines);
  for (const line of lines) if (line.label) labels.set(line.id, line.label);
  return labels;
}

// The unlabeled decisions nearest each line, `perSide` on each side: the answers that move a line most. `skip` holds
// the ones already asked this run, so a skipped question does not come back in the next round.
export function sample(
  lines: readonly LogLine[],
  app: App,
  thresholds: readonly number[],
  perSide = 2,
  skip: ReadonlySet<string> = new Set(),
): LogLine[] {
  const labels = labelsOf(lines);
  const scored = lines
    .filter(
      (line) => line.app === app && !labels.has(line.id) && !skip.has(line.id),
    )
    .flatMap((line) => {
      if (app === "router") {
        if (line.level === undefined) return [];
        const nearest = [thresholds[0]!, thresholds[1]!].reduce((a, b) =>
          Math.abs(line.level! - a) <= Math.abs(line.level! - b) ? a : b,
        );
        return [
          {
            line,
            distance: Math.abs(line.level - nearest),
            side: line.level >= nearest,
          },
        ];
      }
      if (!line.scores) return [];
      return [
        {
          line,
          distance: Math.min(
            ...line.scores.map((score, i) => Math.abs(score - thresholds[i]!)),
          ),
          side: acts(line.scores, thresholds),
        },
      ];
    });
  return [true, false].flatMap((side) =>
    scored
      .filter((item) => item.side === side)
      .sort((a, b) => a.distance - b.distance)
      .slice(0, perSide)
      .map((item) => item.line),
  );
}

export interface Suggestion {
  app: App;
  labels: number;
  before: number[];
  after: number[];
  changed: boolean;
  agreeBefore: Agreement;
  agreeAfter: Agreement;
  // Router: labeled tasks under the confidence floor, which keep their model whatever the cuts.
  unrouted?: number;
  why: string;
  // The app's logged decisions, and what the new lines would do to them: how many a binary app would act on, or
  // where the router would now send its subagents.
  decisions: number;
  acted?: { before: number; after: number };
  moves?: { from: Route; to: Route; n: number }[];
  // Apply when the change agrees with more answers; more when there are too few to fit.
  recommend: "apply" | "keep" | "more";
}
export type Route = "light" | "medium" | "strong" | "kept";
const ROUTES: readonly Route[] = ["light", "medium", "strong", "kept"];
// Where router lines send a logged subagent, or undefined when it has no level.
function routeOf(line: LogLine, [easy, hard, least]: readonly number[]) {
  if (line.level === undefined) return undefined;
  if ((line.confidence ?? 1) < least!) return "kept" as const;
  return ROUTES[routeTier(line.level, easy!, hard!)]!;
}

function binaryCases(
  lines: readonly LogLine[],
  app: App,
  labels: Map<string, Label>,
): Case[] {
  return lines.flatMap((line) => {
    const label = labels.get(line.id);
    return line.app === app &&
      line.scores &&
      (label === "yes" || label === "no")
      ? [{ scores: line.scores, truth: label === "yes" }]
      : [];
  });
}
function routerCases(
  lines: readonly LogLine[],
  labels: Map<string, Label>,
): RouterCase[] {
  const shifts: Partial<Record<Label, number>> = {
    "too-low": 1,
    right: 0,
    "too-high": -1,
  };
  return lines.flatMap((line) => {
    const shift = shifts[labels.get(line.id)!];
    return line.app === "router" &&
      line.level !== undefined &&
      line.tier !== undefined &&
      shift !== undefined
      ? [
          {
            level: line.level,
            confidence: line.confidence ?? 1,
            target: Math.min(2, Math.max(0, line.tier + shift)),
          },
        ]
      : [];
  });
}
function routerAgreement(
  cases: readonly RouterCase[],
  [easy, hard, least]: readonly number[],
) {
  const routed = cases.filter((item) => item.confidence >= least!);
  return {
    agree: routed.filter(
      (item) => routeTier(item.level, easy!, hard!) === item.target,
    ).length,
    total: routed.length,
    unrouted: cases.length - routed.length,
  };
}

// New thresholds for one app from the log's labels, each moved at most MAX_STEP; `currents` are the ones in use. Says
// what they would do to the app's logged decisions and whether to apply them, as the Claude Code plugin's plan does.
export function suggest(
  lines: readonly LogLine[],
  app: App,
  currents: readonly number[],
  goal: Goal = "balanced",
): Suggestion {
  const found = fit(lines, app, currents, goal);
  const mine = lines.filter((line) => line.app === app);
  const recommend =
    found.labels < MIN_LABELS
      ? ("more" as const)
      : found.changed && found.agreeAfter.agree > found.agreeBefore.agree
        ? ("apply" as const)
        : ("keep" as const);
  if (app === "router") {
    const moves = new Map<string, { from: Route; to: Route; n: number }>();
    for (const line of mine) {
      const from = routeOf(line, found.before),
        to = routeOf(line, found.after);
      if (!from || !to || from === to) continue;
      const move = moves.get(`${from}>${to}`) ?? { from, to, n: 0 };
      move.n++;
      moves.set(`${from}>${to}`, move);
    }
    return {
      ...found,
      decisions: mine.length,
      moves: [...moves.values()].sort(
        (a, b) =>
          ROUTES.indexOf(a.from) - ROUTES.indexOf(b.from) ||
          ROUTES.indexOf(a.to) - ROUTES.indexOf(b.to),
      ),
      recommend,
    };
  }
  const scored = mine.filter((line) => line.scores);
  const actedOn = (thresholds: readonly number[]) =>
    scored.filter((line) => acts(line.scores!, thresholds)).length;
  return {
    ...found,
    decisions: scored.length,
    acted: { before: actedOn(found.before), after: actedOn(found.after) },
    recommend,
  };
}
type Fit = Omit<Suggestion, "decisions" | "acted" | "moves" | "recommend">;
function fit(
  lines: readonly LogLine[],
  app: App,
  currents: readonly number[],
  goal: Goal,
): Fit {
  const labels = labelsOf(lines);
  const before = [...currents];
  if (app === "router") {
    const cases = routerCases(lines, labels);
    const agreeBefore = routerAgreement(cases, before);
    if (cases.length < MIN_LABELS)
      return {
        app,
        labels: cases.length,
        before,
        after: before,
        changed: false,
        agreeBefore,
        agreeAfter: agreeBefore,
        unrouted: agreeBefore.unrouted,
        why: `needs ${MIN_LABELS} to move a line`,
      };
    const levels = axis(0, 2);
    const cuts = levels.flatMap((easy) =>
      levels.filter((hard) => easy < hard).map((hard) => [easy, hard]),
    );
    const [easy, hard] = bestToward(cuts, before.slice(0, 2), (cut) => [
      -cases.filter(
        (item) => routeTier(item.level, cut[0]!, cut[1]!) === item.target,
      ).length,
    ]);
    const [least] = bestToward(
      axis(0, 1).map((value) => [value]),
      [before[2]!],
      ([floor]) => [
        -cases
          .filter((item) => item.confidence >= floor!)
          .reduce(
            (sum, item) =>
              sum +
              (routeTier(item.level, easy!, hard!) === item.target ? 1 : -1),
            0,
          ),
      ],
    );
    const after = [easy!, hard!, least!];
    const agreeAfter = routerAgreement(cases, after);
    const changed = compare(after, before) !== 0;
    return {
      app,
      labels: cases.length,
      before,
      after,
      changed,
      agreeBefore,
      agreeAfter,
      unrouted: agreeAfter.unrouted,
      why: changed
        ? "more labeled tasks in their right tier"
        : "the current lines already fit the answers best",
    };
  }
  const cases = binaryCases(lines, app, labels);
  const agreement = (thresholds: readonly number[]): Agreement => {
    const m = confusion(cases, thresholds);
    return { agree: m.tp + m.tn, total: cases.length };
  };
  if (cases.length < MIN_LABELS)
    return {
      app,
      labels: cases.length,
      before,
      after: before,
      changed: false,
      agreeBefore: agreement(before),
      agreeAfter: agreement(before),
      why: `needs ${MIN_LABELS} to move a line`,
    };
  const beta = GOALS[goal];
  let candidates = grid(before.length);
  if (RAISE_ONLY.includes(app))
    candidates = candidates.filter((candidate) =>
      candidate.every((value, i) => value >= before[i]!),
    );
  const after = bestToward(candidates, before, (candidate) => {
    const m = confusion(cases, candidate);
    return [-Math.round(fbeta(m.tp, m.fp, m.fn, beta) * 1e9), m.fp + m.fn];
  });
  const changed = compare(after, before) !== 0;
  return {
    app,
    labels: cases.length,
    before,
    after,
    changed,
    agreeBefore: agreement(before),
    agreeAfter: agreement(after),
    why: changed
      ? "the best separation of your answers"
      : "the current line already separates your answers best",
  };
}
