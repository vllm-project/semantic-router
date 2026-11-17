import type { Snapshot, Feedback, Decision } from "./decisions.ts";
import { isRecord } from "./settings.ts";
import type { Settings } from "./settings.ts";
import type { RuntimeStatus } from "./runtime.ts";
import type { ControlResult } from "./controls.ts";
import type { Verification } from "./verify.ts";
export interface ViewSnapshot extends Snapshot {
  apps: Settings["apps"];
  waits: Settings["waits"];
  controls: {
    revision: number;
    persistence: "durable" | "unavailable";
    goal: Settings["routing"]["goal"];
  };
  calibration: { at?: number; since: number; remind: boolean };
  runtime: RuntimeStatus;
}
export type ControlPatch = {
  apps?: Partial<Settings["apps"]>;
  waits?: Partial<Settings["waits"]>;
  goal?: Settings["routing"]["goal"];
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
    const id = this.state.sessionID;
    if (!id || decision.sessionID !== id || this.state.busy || this.closed)
      return false;
    const operation = ++this.operation;
    this.state.busy = true;
    this.emit();
    let saved = false,
      failure: string | undefined;
    try {
      await this.service.feedback(id, decision.id, feedback);
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
  };
}
export const APP_NAMES = {
  router: "Router",
  escalate: "Escalate",
  redact: "Redact",
  shield: "Shield",
  guard: "Guard",
  check: "Check",
} as const;
export const subjectLine = (decision: Decision): string =>
  decision.subject ?? decision.toolCallID ?? decision.turnID ?? decision.id;

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
// The Claude Code plugin's glyphs, one color per app.
export const APP_LOOK: Record<
  keyof typeof APP_NAMES,
  { glyph: string; tone: Tone }
> = {
  router: { glyph: "⇢", tone: "info" },
  escalate: { glyph: "↑", tone: "orange" },
  redact: { glyph: "▒", tone: "purple" },
  shield: { glyph: "◆", tone: "blue" },
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
// One line per app: ✓ when it decides the case as it does on vela3-lite, ✗ when it does not or cannot.
export function report(found: Verification): string {
  const asked = found.probes.filter((probe) => probe.verdict !== "off");
  const lines = [
    `Runtime: ${found.source === "connect" ? "from /connect" : "from the plugin options"} · ${found.models.length ? `models ${found.models.join(", ")}` : "no model listed"}`,
    ...found.probes.map((probe) =>
      probe.verdict === "off"
        ? `– ${APP_NAMES[probe.app]} off`
        : `${probe.verdict === probe.expected ? "✓" : "✗"} ${APP_NAMES[probe.app]} ${probe.verdict} · ${probe.detail} · ${Math.round(probe.elapsedMs)} ms`,
    ),
  ];
  const middle = median(asked.map((probe) => probe.elapsedMs));
  if (middle !== undefined) lines.push(`Median ${Math.round(middle)} ms`);
  if (asked.some((probe) => probe.verdict === "no answer"))
    lines.push(
      "No answer: offline, timeout or HTTP 401/403 mean the runtime is down, slower than the app's wait (/vsr-waits) or refuses the token; HTTP 503 that its model did not load.",
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
  check: ["it marked it", "it did not"],
  escalate: ["it escalated", "it did not"],
};
const two = (value: number) => value.toFixed(2);
// One question about a logged decision, in plain words, with what vSR did and how near the line it was.
export function calibrationQuestion(question: CalibrationQuestion): {
  title: string;
  description: string;
  options: { value: CalibrationLabel | "skip" | "stop"; title: string }[];
} {
  const tail = [
    { value: "skip" as const, title: "Not sure (skip)" },
    { value: "stop" as const, title: "Stop asking" },
  ];
  if (question.app === "router")
    return {
      title: `This task went to ${TIER_NAMES[question.tier ?? 1]}. Was that right?`,
      description: `${question.subject} · difficulty ${two(question.level ?? 0)} (lines ${two(question.thresholds[0]!)} and ${two(question.thresholds[1]!)})`,
      options: [
        { value: "too-low", title: "It needed a stronger model" },
        { value: "right", title: "The model was right" },
        { value: "too-high", title: "A lighter model would do" },
        ...tail,
      ],
    };
  const [yes, no] = DID[question.app];
  const scores = (question.scores ?? [])
    .map((score, i) => `${two(score)} (line ${two(question.thresholds[i]!)})`)
    .join(", ");
  return {
    title: ASK[question.app],
    description: `${question.subject} · ${question.acted ? yes : no}; score ${scores}`,
    options: [
      { value: "yes", title: "Yes" },
      { value: "no", title: "No" },
      ...tail,
    ],
  };
}
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
}
const LINE_NAMES: Record<keyof typeof APP_NAMES, string[]> = {
  router: ["light below", "strong from", "confidence floor"],
  escalate: ["pushback line", "repeat line"],
  redact: ["personal-data line"],
  shield: ["instruction line"],
  guard: ["review line"],
  check: ["unsupported line"],
};
// What the fit would change, and how many of the person's answers each setting agrees with.
export function suggestionText(found: CalibrationSuggestion): string {
  const lines = LINE_NAMES[found.app].map(
    (name, i) =>
      `${name}: ${two(found.before[i]!)}${found.after[i] === found.before[i] ? " (kept)" : ` → ${two(found.after[i]!)}`}`,
  );
  return [
    ...lines,
    `Agrees with ${found.agreeBefore.agree} of ${found.agreeBefore.total} answers now${found.changed ? `, ${found.agreeAfter.agree} of ${found.agreeAfter.total} after` : ""}.`,
    ...(found.unrouted
      ? [
          `${found.unrouted} answered tasks are under the confidence floor and keep their model.`,
        ]
      : []),
    `${found.labels} answers: ${found.why}. Each line moves at most 0.15 per calibration.`,
  ].join("\n");
}
export interface CalibrationReport {
  persistence: "durable" | "unavailable";
  at?: number;
  since: number;
  remind: boolean;
  apps: {
    app: keyof typeof APP_NAMES;
    enabled: boolean;
    decisions: number;
    answered: number;
    thresholds: number[];
    calibrated: boolean;
  }[];
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
// A subject for a narrow line: each path down to its last part, so the file a step touched stays in view.
export const shortSubject = (decision: Decision): string =>
  subjectLine(decision).replace(
    /(?:~|\.{1,2})?(?:\/[^\s/]+)+\/([^\s/]+)/g,
    "…/$1",
  );
