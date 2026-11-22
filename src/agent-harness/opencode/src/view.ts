import { homedir } from "node:os";
import type { Snapshot, Feedback, Decision } from "./decisions.ts";
import { isRecord } from "./settings.ts";
import { MIN_LABELS } from "./calibration.ts";
import type { Settings } from "./settings.ts";
import type { RuntimeStatus } from "./runtime.ts";
import type { ControlResult } from "./controls.ts";
import type { Verification } from "./verify.ts";
import type { ProbeRow } from "./probes.ts";
import type { Spend, TurnsView } from "./turns.ts";
export interface ViewSnapshot extends Snapshot {
  answerable: string[];
  apps: Settings["apps"];
  waits: Settings["waits"];
  controls: {
    revision: number;
    persistence: "durable" | "unavailable";
    goal: Settings["routing"]["goal"];
    turns: boolean;
    calibrateEvery: number;
    downgradeConfidence: number;
    stepsLow: number;
    stepsHigh: number;
    stepsMinConfidence: number;
    note: string;
    listener: boolean;
  };
  calibration: { at?: number; since: number; remind: boolean };
  runtime: RuntimeStatus & { guardNote?: string; cleartext?: boolean };
  turns: TurnsView;
}
export type ControlPatch = {
  apps?: Partial<Settings["apps"]>;
  waits?: Partial<Settings["waits"]>;
  goal?: Settings["routing"]["goal"];
  turns?: boolean;
  calibrateEvery?: number;
  downgradeConfidence?: number;
  stepsLow?: number;
  stepsHigh?: number;
  stepsMinConfidence?: number;
  follow?: true;
};
export interface ViewService {
  snapshot(sessionID: string, signal: AbortSignal): Promise<ViewSnapshot>;
  controls(
    input: ControlPatch & { instanceID: string; expectedRevision: number },
  ): Promise<ControlResult>;
  feedback(
    sessionID: string,
    decisionID: string,
    feedback: Feedback,
  ): Promise<unknown>;
  diagnose(): Promise<Verification>;
}
export interface ViewState {
  sessionID: string | undefined;
  snapshot: ViewSnapshot | undefined;
  loading: boolean;
  busy: boolean;
  error: string | undefined;
}
export function emptyView(): ViewState {
  return {
    sessionID: undefined,
    snapshot: undefined,
    loading: false,
    busy: false,
    error: undefined,
  };
}
export class DecisionView {
  private readonly service: ViewService;
  private readonly publish: (state: ViewState) => void;
  private state = emptyView();
  private request = 0;
  private operation = 0;
  private abort: AbortController | undefined;
  private closed = false;
  constructor(service: ViewService, publish: (state: ViewState) => void) {
    this.service = service;
    this.publish = publish;
  }
  private emit(): void {
    if (!this.closed) this.publish({ ...this.state });
  }
  select(sessionID: string | undefined): void {
    if (this.closed || this.state.sessionID === sessionID) return;
    this.abort?.abort();
    this.request++;
    this.operation++;
    this.state = { ...emptyView(), sessionID };
    this.emit();
    if (sessionID) void this.refresh();
  }
  async refresh(): Promise<void> {
    const id = this.state.sessionID;
    if (!id || this.closed) return;
    const request = ++this.request;
    this.abort?.abort();
    this.abort = new AbortController();
    const signal = this.abort.signal,
      timer = setTimeout(
        () => this.abort?.signal === signal && this.abort.abort(),
        3000,
      );
    this.state.loading = true;
    this.emit();
    try {
      const snapshot = await this.service.snapshot(id, signal);
      if (
        this.closed ||
        request !== this.request ||
        id !== this.state.sessionID
      )
        return;
      if (
        snapshot.sessionID !== id ||
        !snapshot.instanceID ||
        !Number.isSafeInteger(snapshot.revision) ||
        !snapshot.controls
      )
        throw new Error("Wrong session snapshot");
      const previous = this.state.snapshot;
      if (
        !previous ||
        snapshot.instanceID !== previous.instanceID ||
        snapshot.revision >= previous.revision
      )
        this.state.snapshot = snapshot;
      this.state.error = undefined;
    } catch {
      if (!this.closed && request === this.request)
        this.state.error = "Snapshot unavailable; displayed data may be stale.";
    } finally {
      clearTimeout(timer);
      if (!this.closed && request === this.request) {
        this.state.loading = false;
        this.emit();
      }
    }
  }
  hint(event: unknown): void {
    if (
      !isRecord(event) ||
      typeof event.instanceID !== "string" ||
      typeof event.revision !== "number" ||
      !Number.isSafeInteger(event.revision) ||
      event.revision < 0 ||
      (event.sessionID !== undefined && typeof event.sessionID !== "string")
    )
      return;
    if (event.sessionID && event.sessionID !== this.state.sessionID) return;
    const snapshot = this.state.snapshot;
    if (
      snapshot &&
      event.instanceID === snapshot.instanceID &&
      event.revision <= snapshot.revision
    )
      return;
    void this.refresh();
  }
  async change(patch: ControlPatch): Promise<boolean> {
    const snapshot = this.state.snapshot;
    if (this.closed || this.state.busy || !snapshot) return false;
    const operation = ++this.operation;
    this.state.busy = true;
    this.emit();
    let saved = false,
      failure: string | undefined;
    try {
      const result = await this.service.controls({
        ...patch,
        instanceID: snapshot.instanceID,
        expectedRevision: snapshot.controls.revision,
      });
      saved = result.saved;
      if (!result.saved)
        failure =
          result.reason === "stale"
            ? "Settings changed elsewhere; refresh and retry."
            : "Settings were not saved.";
    } catch {
      failure = "Could not confirm the settings change; refresh to reconcile.";
    }
    if (this.closed || operation !== this.operation) return saved;
    await this.refresh();
    if (this.closed || operation !== this.operation) return saved;
    if (!saved) this.state.error = failure;
    this.state.busy = false;
    this.emit();
    return saved;
  }
  async feedback(decision: Decision, feedback: Feedback): Promise<boolean> {
    // A subagent's decision is listed in its parent's view and answered in its own session.
    if (
      !this.state.snapshot?.decisions.some(
        (item) =>
          item.id === decision.id && item.sessionID === decision.sessionID,
      ) ||
      this.state.busy ||
      this.closed
    )
      return false;
    const operation = ++this.operation;
    this.state.busy = true;
    this.emit();
    let saved = false,
      failure: string | undefined;
    try {
      await this.service.feedback(decision.sessionID, decision.id, feedback);
      saved = true;
    } catch {
      failure = "Could not confirm feedback; refresh to reconcile.";
    }
    if (this.closed || operation !== this.operation) return saved;
    await this.refresh();
    if (this.closed || operation !== this.operation) return saved;
    if (!saved) this.state.error = failure;
    this.state.busy = false;
    this.emit();
    return saved;
  }
  async diagnose(): Promise<Verification | undefined> {
    if (this.state.busy || this.closed) return undefined;
    const operation = ++this.operation;
    this.state.busy = true;
    this.emit();
    let found: Verification | undefined;
    try {
      found = await this.service.diagnose();
    } catch {
      /* Transport status is shown by refresh. */
    }
    if (this.closed || operation !== this.operation) return found;
    await this.refresh();
    if (this.closed || operation !== this.operation) return found;
    this.state.busy = false;
    this.emit();
    return found;
  }
  close(): void {
    this.closed = true;
    this.request++;
    this.operation++;
    this.abort?.abort();
  }
}
export function summary(decisions: readonly Decision[]) {
  const seen = new Set<string>(),
    children = new Set<string>();
  const outcomes: Record<string, number> = {};
  let masked = 0,
    input = 0,
    output = 0,
    cached = 0,
    reasoning = 0,
    cost = 0,
    priced = 0,
    unpriced = 0;
  // Routed subagents priced on the model that ran them and on the session's model.
  const routed = { count: 0, actual: 0, baseline: 0, on: new Set<string>() };
  for (const decision of decisions) {
    if (seen.has(decision.id)) continue;
    seen.add(decision.id);
    outcomes[decision.outcome] = (outcomes[decision.outcome] ?? 0) + 1;
    masked += Object.values(decision.counts ?? {}).reduce<number>(
      (a, b) => a + (b ?? 0),
      0,
    );
    if (
      !decision.usage ||
      !decision.observedModel ||
      !decision.childSessionID ||
      children.has(decision.childSessionID)
    )
      continue;
    children.add(decision.childSessionID);
    input += decision.usage.input;
    output += decision.usage.output;
    cached += decision.usage.cachedInput;
    reasoning += decision.usage.reasoning;
    if (decision.costKnown) {
      cost += decision.usage.cost;
      priced++;
    } else unpriced++;
    if (decision.estimate) {
      routed.count++;
      routed.actual += decision.estimate.actual;
      routed.baseline += decision.estimate.baseline;
      routed.on.add(decision.estimate.on);
    }
  }
  return {
    total: seen.size,
    outcomes,
    masked,
    children: children.size,
    input,
    output,
    cached,
    reasoning,
    cost,
    priced,
    unpriced,
    routed,
  };
}
// Cents and below keep four places, as the subagents row does, so two small amounts still differ.
const dollars = (value: number) =>
  value >= 1
    ? `~$${value.toFixed(2)}`
    : value < 0.0001
      ? "<$0.0001"
      : `~$${value.toFixed(4)}`;
// What routing came to, as the Claude Code plugin's routing row: routed subagents and your routed turns at list price,
// against the same tokens on the session's model. Undefined until something routed was priced.
export function savingWords(
  routed: ReturnType<typeof summary>["routed"],
  turns: Spend | undefined,
): string | undefined {
  const count = routed.count + (turns?.turns ?? 0);
  const baseline = routed.baseline + (turns?.baseline ?? 0);
  if (count === 0 || baseline === 0) return undefined;
  const actual = routed.actual + (turns?.actual ?? 0);
  const on = new Set([...routed.on, ...(turns ? [turns.on] : [])]);
  const what = [
    routed.count
      ? `${routed.count} ${routed.count === 1 ? "subagent" : "subagents"}`
      : "",
    turns?.turns
      ? `${turns.turns} ${turns.turns === 1 ? "turn" : "turns"}`
      : "",
  ]
    .filter(Boolean)
    .join(" + ");
  const ratio = actual / baseline;
  // Twice the cost or more reads better as a multiple than as a percentage.
  const verdict =
    ratio <= 1
      ? `saved ${Math.round((1 - ratio) * 100)}%`
      : ratio >= 2
        ? `${Math.round(ratio)}x more`
        : `${Math.round((ratio - 1) * 100)}% more`;
  return `routing ${what} ${dollars(actual)} vs ${dollars(baseline)} on ${on.size === 1 ? modelName([...on][0]!) : "your models"} · ${verdict}`;
}
// The answers the panel offers for a decision, as the Claude Code plugin's pane offers them: a route's model fit,
// or whether another app's call was right.
export function answersOf(
  decision: Pick<Decision, "app">,
): { value: Feedback; title: string }[] {
  return decision.app === "router"
    ? [
        { value: "right-model", title: "right model" },
        { value: "needed-stronger", title: "needed stronger" },
        { value: "could-be-lighter", title: "could be lighter" },
      ]
    : [
        { value: "right", title: "right call" },
        { value: "wrong", title: "wrong call" },
      ];
}
export const APP_NAMES = {
  router: "Router",
  escalate: "Escalate",
  redact: "Redact",
  shield: "Shield",
  guard: "Guard",
  check: "Check",
} as const;
// A path under the home directory reads as `~/…`, so the panel and a screenshot of it do not show the user's name.
const HOME = homedir();
export const homeShort = (text: string): string =>
  HOME && HOME !== "/" ? text.replaceAll(`${HOME}/`, "~/") : text;
export const subjectLine = (decision: Decision): string =>
  homeShort(
    decision.subject ?? decision.toolCallID ?? decision.turnID ?? decision.id,
  );

// Colors by role. The panel resolves each against OpenCode's theme tokens, so they follow the theme in use.
export type Tone =
  | "text"
  | "muted"
  | "border"
  | "focus"
  | "background"
  | "info"
  | "success"
  | "warning"
  | "error"
  | "blue"
  | "purple"
  | "orange";
export const THEME_TOKENS: Record<Tone, readonly string[]> = {
  text: ["text", "base"],
  muted: ["text", "muted"],
  border: ["border", "base"],
  focus: ["background", "action", "primary", "focused"],
  background: ["background", "base"],
  info: ["text", "feedback", "info", "base"],
  success: ["text", "feedback", "success", "base"],
  warning: ["text", "feedback", "warning", "base"],
  error: ["text", "feedback", "error", "base"],
  blue: ["categorical", "0", "200"],
  purple: ["categorical", "1", "200"],
  orange: ["categorical", "3", "200"],
};
// The Claude Code plugin's glyphs and colors per app (hooks/lib/apps.ts): guard and shield warning, check success,
// redact merged (purple), router suggestion (blue) and escalate claude (orange).
export const APP_LOOK: Record<
  keyof typeof APP_NAMES,
  { glyph: string; tone: Tone }
> = {
  router: { glyph: "⇢", tone: "info" },
  escalate: { glyph: "↑", tone: "orange" },
  redact: { glyph: "▒", tone: "purple" },
  shield: { glyph: "◆", tone: "warning" },
  guard: { glyph: "‖", tone: "warning" },
  check: { glyph: "✓", tone: "success" },
};
// A decision that needs a look is red, one that went through only in part amber, one that changed something in its
// app's color, and one that let things be is quiet.
const NEEDS_LOOK = new Set(["held", "declined", "withheld", "unverified"]);
const IN_PART = new Set(["partial", "skipped"]);
const LET_BE = new Set(["allowed", "verified", "kept"]);
export function toneOf(decision: Pick<Decision, "app" | "outcome">): Tone {
  if (NEEDS_LOOK.has(decision.outcome)) return "error";
  if (IN_PART.has(decision.outcome)) return "warning";
  if (LET_BE.has(decision.outcome)) return "muted";
  return APP_LOOK[decision.app].tone;
}
export type GaugePart = { text: string; kind: "fill" | "rest" | "line" };
// A score as a bar with the decision's thresholds marked on it, and the words that read it. The router's score is a
// difficulty level from 0 to 2 against two lines; escalate's is the larger of its two scores, against that one's line.
export function gauge(
  decision: Pick<Decision, "app" | "score" | "features" | "thresholds">,
  cells = 12,
): { parts: GaugePart[]; words: string } | undefined {
  const { score, thresholds = [] } = decision;
  if (score === undefined) return undefined;
  const max = decision.app === "router" ? 2 : 1;
  const lines =
    decision.app === "router"
      ? thresholds.slice(0, 2)
      : decision.app === "escalate"
        ? thresholds
            .slice(0, 2)
            .filter((_, i) => {
              const features = decision.features ?? [];
              return (
                features.length < 2 || features[i] === Math.max(...features)
              );
            })
            .slice(0, 1)
        : thresholds.slice(0, 1);
  const at = (value: number) =>
    Math.round((Math.min(max, Math.max(0, value)) / max) * cells);
  const filled = at(score);
  const marks = new Set(lines.map((line) => Math.min(cells - 1, at(line))));
  const parts: GaugePart[] = [];
  for (let cell = 0; cell < cells; cell++) {
    const kind = marks.has(cell) ? "line" : cell < filled ? "fill" : "rest";
    const text = kind === "line" ? "┃" : kind === "fill" ? "█" : "░";
    const last = parts[parts.length - 1];
    if (last?.kind === kind) last.text += text;
    else parts.push({ kind, text });
  }
  const fixed = (value: number) => value.toFixed(2);
  const words =
    decision.app === "router"
      ? `level ${fixed(score)}` +
        (lines.length === 2
          ? ` · light below ${fixed(lines[0]!)}, strong from ${fixed(lines[1]!)}`
          : "")
      : lines.length
        ? `${fixed(score)} ${score >= lines[0]! ? "≥" : "<"} ${fixed(lines[0]!)}`
        : fixed(score);
  return { parts, words };
}
// A model as the panel names it: its last path part, without OpenCode's `#default` variant.
export const modelName = (model: string): string =>
  model
    .replace(/#default$/, "")
    .split("/")
    .pop()!;
export const tokenCount = (value: number): string =>
  value < 1000 ? String(value) : `${(value / 1000).toFixed(1)}k`;
export function median(values: readonly number[]): number | undefined {
  if (!values.length) return undefined;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = sorted.length >> 1;
  return sorted.length % 2
    ? sorted[middle]
    : (sorted[middle - 1]! + sorted[middle]!) / 2;
}
// The model every app asks, not the whole listing: a hosted API lists many and answers with the one named.
function modelWords(found: Verification): string {
  const listed = found.models.length;
  if (!listed) return "no model listed";
  if (found.model)
    return `model ${found.model}${listed > 1 ? ` (${listed} listed)` : ""}`;
  if (listed === 1) return `model ${found.models[0]}`;
  return `model not set: the runtime lists ${listed}, name one ${found.source === "connect" ? "in /connect" : "in the runtimeModel option"}`;
}
// One line per app: ✓ when it decides the case as it does on vela3-lite, ✗ when it does not or cannot.
export function report(found: Verification): string {
  const asked = found.probes.filter((probe) => probe.verdict !== "off");
  const lines = [
    `Runtime: ${found.source === "connect" ? "from /connect" : "from the plugin options"} · ${modelWords(found)}`,
    ...found.probes.map((probe) =>
      probe.verdict === "off"
        ? `– ${APP_NAMES[probe.app]} off`
        : `${probe.verdict === probe.expected ? "✓" : "✗"} ${APP_NAMES[probe.app]} ${probe.verdict} · ${probe.detail} · ${Math.round(probe.elapsedMs)} ms`,
    ),
  ];
  const middle = median(asked.map((probe) => probe.elapsedMs));
  if (middle !== undefined) lines.push(`Median ${Math.round(middle)} ms`);
  if (found.cleartext)
    lines.push(
      "Warning: plain http to another machine, so what the apps check, and the token, travel unencrypted; https is recommended.",
    );
  if (asked.some((probe) => probe.verdict === "no answer"))
    lines.push(
      "No answer: offline, timeout or HTTP 401/403 mean the runtime is down, slower than the app's wait (/vsr-config) or refuses the token; HTTP 503 that its model did not load.",
    );
  return lines.join("\n");
}

export interface CalibrationQuestion {
  id: string;
  app: keyof typeof APP_NAMES;
  subject: string;
  acted: boolean;
  scores?: number[];
  level?: number;
  tier?: number;
  thresholds: number[];
}
export type CalibrationLabel = "yes" | "no" | "too-low" | "right" | "too-high";
const TIER_NAMES = ["the light model", "the middle model", "the strong model"];
const ASK: Record<Exclude<keyof typeof APP_NAMES, "router">, string> = {
  guard: "Should vSR have asked you before this command ran?",
  redact: "Should vSR have hidden personal data in this output?",
  shield: "Should vSR have warned the agent about instructions in this?",
  check: "Should vSR have marked the answer to this as unsupported?",
  escalate: "Was this prompt pushback that should raise the next model?",
};
const DID: Record<
  Exclude<keyof typeof APP_NAMES, "router">,
  [string, string]
> = {
  guard: ["it asked", "it let it run"],
  redact: ["it hid some", "it hid none"],
  shield: ["it warned", "it did not warn"],
  check: ["it marked it", "it did not mark it"],
  escalate: ["it escalated", "it did not escalate"],
};
const two = (value: number) => value.toFixed(2);
// What a question is about, named on the box that shows it.
const ABOUT: Record<keyof typeof APP_NAMES, string> = {
  router: "task",
  guard: "command",
  redact: "output",
  shield: "file or page",
  check: "answer",
  escalate: "prompt",
};
// One question about a logged decision, in plain words, with what vSR did and how near the line it was: as a title
// and description, and in parts for a dialog that sets them apart.
export function calibrationQuestion(question: CalibrationQuestion): {
  title: string;
  description: string;
  about: string;
  subject: string;
  did: string;
  acted: boolean;
  score: string;
  options: { value: CalibrationLabel | "skip" | "stop"; title: string }[];
} {
  const tail = [
    { value: "skip" as const, title: "Not sure (skip)" },
    { value: "stop" as const, title: "Stop asking" },
  ];
  if (question.app === "router") {
    const tier = TIER_NAMES[question.tier ?? 1]!;
    const score = `difficulty ${two(question.level ?? 0)} (lines ${two(question.thresholds[0]!)} and ${two(question.thresholds[1]!)})`;
    return {
      title: `This task went to ${tier}. Was that right?`,
      description: `${question.subject} · ${score}`,
      about: ABOUT.router,
      subject: question.subject,
      did: `vSR sent it to ${tier}`,
      acted: true,
      score,
      options: [
        { value: "too-low", title: "It needed a stronger model" },
        { value: "right", title: "The model was right" },
        { value: "too-high", title: "A lighter model would do" },
        ...tail,
      ],
    };
  }
  const [yes, no] = DID[question.app];
  const scores = (question.scores ?? [])
    .map((score, i) => `${two(score)} (line ${two(question.thresholds[i]!)})`)
    .join(", ");
  const did = question.acted ? yes : no;
  return {
    title: ASK[question.app],
    description: `${question.subject} · ${did}; score ${scores}`,
    about: ABOUT[question.app],
    subject: question.subject,
    did: did.replace(/^it /, "vSR "),
    acted: Boolean(question.acted),
    score: `score ${scores}`,
    options: [
      { value: "yes", title: "Yes" },
      { value: "no", title: "No" },
      ...tail,
    ],
  };
}
export type Route = "light" | "medium" | "strong" | "kept";
export interface CalibrationSuggestion {
  app: keyof typeof APP_NAMES;
  labels: number;
  before: number[];
  after: number[];
  changed: boolean;
  agreeBefore: { agree: number; total: number };
  agreeAfter: { agree: number; total: number };
  unrouted?: number;
  why: string;
  decisions: number;
  acted?: { before: number; after: number };
  moves?: { from: Route; to: Route; n: number }[];
  recommend: "apply" | "keep" | "more";
}
// Each line's name and what it does, in the Claude Code plugin's words with the agent and OpenCode's tiers.
export const LINES: Record<keyof typeof APP_NAMES, [string, string][]> = {
  router: [
    ["Light line", "tasks judged easier than this go to the light model."],
    [
      "Strong line",
      "tasks judged at least this hard go to the strong model; those between the two lines go to the medium one.",
    ],
    [
      "Confidence line",
      "vSR changes a subagent's model only when it is at least this sure.",
    ],
  ],
  escalate: [
    [
      "Pushback line",
      "when your message reads at least this much like pushback on the last answer, the agent re-checks and the next subagent goes one step up.",
    ],
    [
      "Repeat line",
      "when you ask for the same thing again at least this clearly, vSR does the same.",
    ],
  ],
  redact: [
    [
      "Hiding line",
      "vSR hides text it is at least this sure is personal data (secrets are always hidden).",
    ],
  ],
  shield: [
    [
      "Warning line",
      "vSR warns the agent about a file or page at least this likely to be trying to instruct it.",
    ],
  ],
  guard: [
    [
      "Review line",
      "vSR asks you before a command it rates at least this risky.",
    ],
  ],
  check: [
    [
      "Check line",
      "vSR marks an answer unverified when it is at least this sure the answer claims things the agent did not read.",
    ],
  ],
};
// What the new lines would do on the person's log, for an app that acts or not.
const EFFECT: Record<Exclude<keyof typeof APP_NAMES, "router">, string> = {
  guard: "vSR would hold {after} of these {n} commands for you{was}.",
  escalate: "vSR would escalate {after} of these {n} messages{was}.",
  redact: "vSR would hide data in {after} of these {n} outputs{was}.",
  shield: "vSR would warn the agent about {after} of these {n} outputs{was}.",
  check: "vSR would mark {after} of these {n} answers unverified{was}.",
};
export const GOAL_WORDS = {
  balanced: "balanced: as few wrong calls as possible either way",
  "fewer-false-alarms":
    "fewer false alarms: act only when sure, even if some cases slip by",
  "catch-more":
    "catch more: act on more borderline cases, even with some false alarms",
} as const;
const KEPT = "its own model";
const routeName = (route: Route) =>
  route === "kept" ? KEPT : `the ${route} model`;
const plural = (n: number, one: string, many = `${one}s`) =>
  `${n} ${n === 1 ? one : many}`;
// What new router lines would do to the logged subagents, in words.
function routerMoves(moves: NonNullable<CalibrationSuggestion["moves"]>) {
  const phrases = moves
    .filter((move) => move.from !== "kept")
    .map(
      (move) =>
        `${plural(move.n, "task")} that went to ${routeName(move.from)} would ${move.to === "kept" ? `keep ${KEPT}` : `go to ${routeName(move.to)}`}`,
    );
  const routed = moves.filter((move) => move.from === "kept");
  if (routed.length)
    phrases.push(
      `${plural(
        routed.reduce((sum, move) => sum + move.n, 0),
        "subagent",
      )} kept on ${KEPT} would be routed (${routed.map((move) => `${move.n} to ${routeName(move.to)}`).join(", ")})`,
    );
  if (!phrases.length)
    return "No subagent in your log would get a different model.";
  const text = phrases.join("; ") + ".";
  return text[0]!.toUpperCase() + text.slice(1);
}
// What a calibration would change, in plain words as the Claude Code plugin prints it: each line and what it does,
// what the change would do on the person's own log, how many answers agree, and a recommendation.
export function suggestionText(
  found: CalibrationSuggestion,
  goal?: keyof typeof GOAL_WORDS,
): string {
  const out: string[] = [];
  if (goal && found.app !== "router") out.push(`Aim: ${GOAL_WORDS[goal]}.`, "");
  LINES[found.app].forEach(([name, does], i) => {
    const was = found.before[i]!,
      now = found.after[i]!;
    out.push(
      `${name} ${Math.abs(now - was) > 1e-9 ? `${two(was)} → ${two(now)}` : `stays ${two(was)}`}: ${does}`,
    );
  });
  if (found.recommend !== "more") {
    const effect =
      found.app === "router"
        ? routerMoves(found.moves ?? [])
        : EFFECT[found.app]
            .replace("{after}", String(found.acted?.after ?? 0))
            .replace("{n}", String(found.decisions))
            .replace(
              "{was}",
              found.acted && found.acted.before !== found.acted.after
                ? ` instead of ${found.acted.before}`
                : " (the same as now)",
            );
    out.push(
      "",
      `On your log: ${effect}`,
      `Your answers: ${found.agreeBefore.agree} of ${found.agreeBefore.total} agree with vSR now, ${found.agreeAfter.agree} of ${found.agreeAfter.total} would after.`,
    );
    if (found.unrouted)
      out.push(
        `${plural(found.unrouted, "answered task")} under the confidence line keep ${KEPT}.`,
      );
  }
  out.push(
    "",
    found.recommend === "more"
      ? `Not enough answers yet: ${found.labels} of ${MIN_LABELS} (answer more in /vsr or here).`
      : found.recommend === "apply"
        ? "Recommended: apply"
        : "Recommended: keep the current lines",
  );
  return out.join("\n");
}
// Several apps' suggestions at once, as plain.py's `suggestion`: the aim once, each app under its name, then which to
// apply overall.
export function suggestionsText(
  found: readonly CalibrationSuggestion[],
  goal?: keyof typeof GOAL_WORDS,
): string {
  if (found.length === 1) return suggestionText(found[0]!, goal);
  const out: string[] = [];
  if (goal && found.some((item) => item.app !== "router"))
    out.push(`Aim: ${GOAL_WORDS[goal]}.`, "");
  for (const item of found)
    out.push(
      appTitle(item.app),
      ...suggestionText(item)
        .split("\n")
        .filter(Boolean)
        .map((line) => `  ${line}`),
      "",
    );
  const apply = found
    .filter((item) => item.recommend === "apply")
    .map((item) => APP_WORDS[item.app][0]);
  out.push(
    apply.length
      ? `Overall: apply the changes for ${listed(apply)}; keep the rest.`
      : "Overall: keep the current lines.",
  );
  return out.join("\n");
}
export const listed = (names: readonly string[]) =>
  names.length < 2
    ? names.join("")
    : `${names.slice(0, -1).join(", ")} and ${names.at(-1)}`;
export interface CalibrationReport {
  persistence: "durable" | "unavailable";
  at?: number;
  since: number;
  remind: boolean;
  model?: string;
  calibratedModel?: string;
  first?: number;
  last?: number;
  apps: {
    app: keyof typeof APP_NAMES;
    enabled: boolean;
    decisions: number;
    answered: number;
    acted: number;
    explicit: number;
    implied: number;
    routes?: number[];
    thresholds: number[];
    calibrated: boolean;
  }[];
}
// Each app's plain name and what it does, as the Claude Code plugin's plain.py APP_WORDS, with the agent for Claude.
export const APP_WORDS: Record<keyof typeof APP_NAMES, [string, string]> = {
  router: ["Routing", "which model each subagent gets"],
  guard: ["Risky commands", "asks you before a risky command runs"],
  redact: ["Personal data", "hides personal data from the agent"],
  shield: [
    "Injection warnings",
    "warns the agent when a file or page tries to instruct it",
  ],
  escalate: [
    "Pushback",
    "sends the next subagent a tier up when you push back",
  ],
  check: ["Answer checks", "marks answers that claim more than the agent read"],
};
export const appTitle = (app: keyof typeof APP_NAMES) =>
  `${APP_WORDS[app][0]} (${APP_WORDS[app][1]})`;
// What the log shows for an app: {n} decisions, {acted} acted, {other} not. Unlike the Claude Code plugin's, this log
// keeps what redact and shield let through as well.
const LOGGED: Record<
  Exclude<keyof typeof APP_NAMES, "router">,
  [string, string, string]
> = {
  guard: ["command", "commands", ": {acted} held for you, {other} allowed."],
  escalate: [
    "message",
    "messages",
    " taken as pushback on the last answer (vSR logs only these).",
  ],
  redact: [
    "text read",
    "texts read",
    ": {acted} with personal data hidden, {other} left as they were.",
  ],
  shield: [
    "file or page read",
    "files or pages read",
    ": the agent was warned about {acted}, {other} passed.",
  ],
  check: [
    "answer checked",
    "answers checked",
    ": {acted} marked unverified, {other} passed.",
  ],
};
// Which answers the log implies for an app, as the person gave them.
const IMPLIED_BY: Partial<Record<keyof typeof APP_NAMES, string>> = {
  router: "your pushback implies",
  guard: "your declines imply",
};
const day = (at: number) => {
  const date = new Date(at);
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")}`;
};
// The log in plain words for the chosen apps, as the Claude Code plugin's `report`: what it holds, the decision model
// now and at the last calibration, and per app what vSR did and how many answers there are.
export function calibrationReportText(
  found: CalibrationReport,
  apps: readonly (keyof typeof APP_NAMES)[],
): string {
  const total = found.apps.reduce((sum, item) => sum + item.decisions, 0);
  const out = [
    `vSR's log holds ${plural(total, "decision")}${found.first === undefined || found.last === undefined ? "" : ` from ${day(found.first)} to ${day(found.last)}`}.`,
    `Decision model now: ${found.model ?? "unreachable"}; ` +
      (found.at === undefined
        ? "not calibrated before."
        : `last calibrated ${day(found.at)}${found.calibratedModel ? ` with ${found.calibratedModel}` : ""}.`),
  ];
  for (const app of apps) {
    const item = found.apps.find((entry) => entry.app === app)!;
    const head =
      appTitle(app) + (item.enabled ? "" : " — off in your settings");
    if (!item.decisions) {
      out.push("", head, "  No decisions logged yet.");
      continue;
    }
    const routes = item.routes ?? [0, 0, 0, 0];
    const logged =
      app === "router"
        ? `${plural(item.decisions, "subagent")}: ${routes[0]} went to the light model, ${routes[1]} to medium, ${routes[2]} to strong, ${routes[3]} kept ${KEPT}.`
        : plural(item.decisions, LOGGED[app][0], LOGGED[app][1]) +
          LOGGED[app][2]
            .replace("{acted}", String(item.acted))
            .replace("{other}", String(item.decisions - item.acted));
    const implied = item.implied
      ? ` (+${item.implied} ${IMPLIED_BY[app] ?? "the log implies"})`
      : "";
    out.push(
      "",
      head,
      `  ${logged} You answered ${item.explicit}${implied}; ${MIN_LABELS} are needed to tune.`,
    );
  }
  // Claude Code's step 3 says this once, when router or guard is chosen.
  const counted = [
    apps.includes("guard") &&
      'a held command you declined counts as "should ask"',
    apps.includes("router") &&
      'pushback within 15 minutes of a light or medium route counts as "needed stronger"',
  ].filter(Boolean);
  if (counted.length)
    out.push(
      "",
      `The log already counts some answers for you: ${counted.join(", and ")}.`,
    );
  return out.join("\n");
}
export interface ProbeResult {
  model?: string;
  calibratedModel?: string;
  rows: ProbeRow[];
}
// What the decision model changed from, when it did since the last calibration.
export const modelChanged = (found: {
  model?: string;
  calibratedModel?: string;
}) =>
  Boolean(
    found.model &&
      found.calibratedModel &&
      found.model !== found.calibratedModel,
  );
const DOES: Record<string, [string, string]> = {
  guard: ["vSR holds it", "it runs"],
  shield: ["vSR warns", "no warning"],
  redact: ["vSR hides personal data", "nothing hidden"],
  check: ["vSR marks it", "no mark"],
  escalate: ["vSR escalates", "no escalation"],
  repeat: ["a repeat", "not a repeat"],
};
function does(row: ProbeRow, verdict: ProbeRow["expect"]): string {
  if (verdict === "kept") return "the model is kept";
  if (verdict === "light" || verdict === "medium" || verdict === "strong")
    return `routed ${verdict}`;
  const [acts, passes] = DOES[row.repeat ? "repeat" : row.app]!;
  return verdict === "acts" ? acts : passes;
}
const score2 = (value: number | undefined) =>
  value === undefined ? "-" : value.toFixed(2);
// The probe in plain words, as the Claude Code plugin's calibrate tells it: per app how many known cases the model
// still decides as measured, and each one that flipped, drifted or got no answer.
export function probeText(found: ProbeResult): string {
  const out = [
    `Decision model: ${found.model ?? "unknown"}` +
      (modelChanged(found)
        ? `; you last calibrated with ${found.calibratedModel}.`
        : "."),
  ];
  const apps = [...new Set(found.rows.map((row) => row.app))];
  for (const app of apps) {
    const rows = found.rows.filter((row) => row.app === app);
    const count = (status: ProbeRow["status"]) =>
      rows.filter((row) => row.status === status).length;
    out.push(
      "",
      `${APP_NAMES[app]}, ${rows.length} known cases: ` +
        [
          `${count("ok")} as measured`,
          count("flip") && `${count("flip")} flipped`,
          count("drift") && `${count("drift")} drifted`,
          count("error") && `${count("error")} no answer`,
        ]
          .filter(Boolean)
          .join(", ") +
        ".",
    );
    for (const row of rows) {
      const name = clip(row.name, 48);
      if (row.status === "flip")
        out.push(
          `  Flipped: ${name}: ${does(row, row.now!)} now (${score2(row.score)}), measured ${does(row, row.expect)} (${score2(row.measured)}).`,
        );
      else if (row.status === "drift")
        out.push(
          `  Drifted: ${name}: ${score2(row.measured)} → ${score2(row.score)}.`,
        );
      else if (row.status === "error")
        out.push(`  No answer: ${name} (${row.error ?? "error"}).`);
    }
  }
  if (!found.rows.length) out.push("", "No known cases for these apps.");
  else if (
    modelChanged(found) &&
    found.rows.some((row) => row.status === "flip" || row.status === "drift")
  )
    out.push(
      "",
      "Your answers were given on the old model's scores, so lines fitted to them may not suit this one: answer some newer decisions before you apply.",
    );
  return out.join("\n");
}
// At most `width` characters, an ellipsis taking the last when it is cut.
export function clip(text: string, width: number): string {
  const chars = Array.from(text);
  return chars.length <= width
    ? text
    : chars.slice(0, Math.max(0, width - 1)).join("") + "…";
}
// What one line says of a decision beside its subject: what redact hid, the model the router picked, or the score
// against its threshold.
export function brief(
  decision: Pick<
    Decision,
    "app" | "score" | "features" | "thresholds" | "counts" | "model"
  >,
): string {
  const counts = Object.entries(decision.counts ?? {});
  if (counts.length)
    return counts.map(([label, count]) => `${label} ×${count}`).join(" · ");
  if (decision.model) return `→ ${modelName(decision.model)}`;
  return gauge(decision)?.words ?? "";
}
// Why a decision went the way it did, in a few words: the reason when it was not the score's call, then what the
// score or the change was. The sidebar shows it under a decision that needs a look or went through in part.
export function why(decision: Decision): string {
  return [
    decision.reason && decision.reason !== "selected"
      ? decision.reason.replaceAll("-", " ")
      : "",
    brief(decision),
  ]
    .filter(Boolean)
    .join(" · ");
}
// What prompted a route of the user's own turn, using the lines recorded with that decision.
export function turnRouteReason(decision: Decision): string | undefined {
  if (
    decision.app !== "router" ||
    decision.scope !== "turn" ||
    decision.outcome !== "routed"
  )
    return undefined;
  if (decision.reason === "pushback") return "Routed after pushback";
  const { score, thresholds } = decision;
  if (decision.reason !== "selected" || score === undefined) return undefined;
  const [light, strong] = thresholds ?? [];
  const level = score.toFixed(2);
  if (strong !== undefined && score >= strong)
    return `High difficulty ${level} ≥ ${strong.toFixed(2)}`;
  if (light !== undefined && strong !== undefined)
    return `${score < light ? "Low" : "Medium"} difficulty ${level}`;
  return `Task difficulty ${level}`;
}
// Everything the view keeps of a decision, one labelled row each, for its detail card under its score bar. Rows with
// nothing to say are left out.
export function details(
  decision: Decision,
  context: { sessionID: string; feedback?: Feedback; answerable: boolean },
): { label: string; value: string }[] {
  const usage = decision.usage;
  const counts = Object.entries(decision.counts ?? {});
  const rows: [string, string | undefined][] = [
    ["about", subjectLine(decision)],
    [
      "scores",
      decision.app === "escalate" && decision.features?.length === 2
        ? `pushback ${decision.features[0]!.toFixed(2)} · repeat ${decision.features[1]!.toFixed(2)}`
        : undefined,
    ],
    ["confidence", decision.confidence?.toFixed(2)],
    [
      "reason",
      turnRouteReason(decision) ??
        (decision.reason && decision.reason !== "selected"
          ? decision.reason.replaceAll("-", " ")
          : undefined),
    ],
    [
      "model",
      decision.model &&
        modelName(decision.model) +
          (!decision.observedModel
            ? " · pending"
            : modelName(decision.observedModel) === modelName(decision.model)
              ? " · ran"
              : ` · ran ${modelName(decision.observedModel)}`),
    ],
    ["hid", counts.map(([label, count]) => `${label} ×${count}`).join(" · ")],
    ["found", decision.labels?.join(" · ")],
    // What check's model marked as backed by nothing the agent read, one row each.
    ...(decision.claims ?? []).map((claim, index): [string, string] => [
      index === 0 ? "unbacked" : "",
      `“${claim}”`,
    ]),
    [
      "cache",
      decision.reread &&
        `would re-read ${tokenCount(decision.reread.tokens)} tokens uncached: $${decision.reread.cost.toFixed(2)}, 3 turns save $${decision.reread.saving.toFixed(2)}`,
    ],
    [
      "checked",
      decision.coverage &&
        `${decision.coverage.checked} of ${decision.coverage.total} characters`,
    ],
    [
      "tokens",
      usage &&
        `${tokenCount(usage.input)} in · ${tokenCount(usage.output)} out · ${tokenCount(usage.cachedInput)} cached` +
          (decision.costKnown ? ` · ~$${usage.cost.toFixed(4)}` : ""),
    ],
    ["subagent", decision.childOutcome],
    [
      "session",
      decision.sessionID === context.sessionID
        ? "this session"
        : `a subagent's, ${decision.sessionID}`,
    ],
    [
      "when",
      new Date(decision.at).toLocaleTimeString() +
        (decision.elapsedMs === undefined
          ? ""
          : ` · took ${decision.elapsedMs} ms`),
    ],
    [
      "answer",
      context.feedback
        ? context.feedback.replaceAll("-", " ")
        : context.answerable
          ? "none yet"
          : "none: a rule or a skipped call decided this, so calibration does not learn from it",
    ],
  ];
  return rows
    .filter((row): row is [string, string] => Boolean(row[1]))
    .map(([label, value]) => ({ label, value }));
}
// A subject for a narrow line: each path down to its last part, so the file a step touched stays in view.
export const shortSubject = (decision: Decision): string =>
  subjectLine(decision).replace(
    /(?:~|\.{1,2})?(?:\/[^\s/]+)+\/([^\s/]+)/g,
    "…/$1",
  );
