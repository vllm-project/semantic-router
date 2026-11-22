import type { Decision, Feedback, Outcome } from "./decisions.ts";
import type { App, Settings } from "./settings.ts";
import { isRecord } from "./settings.ts";

export const KEPT_LINES = 5000;
const PREFIX = "log/";
export const BINARY_LABELS = ["yes", "no"] as const;
export const ROUTER_LABELS = ["too-low", "right", "too-high"] as const;
// `yes`/`no`: should the app have stepped in. Router: was the routed tier too low, right or too high.
export type Label =
  | (typeof BINARY_LABELS)[number]
  | (typeof ROUTER_LABELS)[number];

// One decision calibration can learn from: the scores it held against the thresholds in use then, and what it was
// about. Decisions with no score, or decided by a rule rather than the model, are not kept.
export interface LogLine {
  id: string;
  at: number;
  session: string;
  app: App;
  outcome: Outcome;
  subject: string;
  // Binary apps: the scores and the thresholds they were held against, in the same order.
  scores?: number[];
  thresholds?: number[];
  // Router: the difficulty level, its confidence, and the tier it routed to (0 lightest of the three).
  level?: number;
  confidence?: number;
  tier?: number;
  label?: Label;
}
export interface LogStore {
  get(key: string): Promise<unknown>;
  set(key: string, value: unknown): Promise<void>;
  remove(key: string): Promise<void>;
  scan(options: { prefix: string; after?: string; limit?: number }): Promise<{
    entries: readonly { key: string; value: unknown }[];
    next?: string;
  }>;
}

const ACTS: readonly Outcome[] = [
  "held",
  "approved",
  "declined",
  "redacted",
  "shielded",
  "unverified",
  "escalated",
];
export const acted = (line: Pick<LogLine, "outcome">) =>
  ACTS.includes(line.outcome);
// Not the model's call: a rule, the listener, a native permission or a limit decided these.
const RULED = new Set([
  "fixed-rule",
  "listener-rule",
  "listener-unchecked",
  "native-permission",
  "coverage-limit",
  "placeholder-write",
]);

export function lineOf(
  decision: Decision,
  settings: Settings,
): LogLine | undefined {
  if (
    decision.subject === undefined ||
    (decision.reason && RULED.has(decision.reason))
  )
    return undefined;
  const base = {
    id: decision.id,
    at: decision.at,
    session: decision.sessionID,
    app: decision.app,
    outcome: decision.outcome,
    subject: decision.subject,
  };
  if (decision.app === "router") {
    const tiers = settings.routing.tiers;
    const name = decision.model?.split("#")[0];
    const tier = tiers.findIndex((item) => item.model === name);
    if (
      decision.outcome !== "routed" ||
      decision.score === undefined ||
      decision.confidence === undefined ||
      tier < 0
    )
      return undefined;
    return {
      ...base,
      level: decision.score,
      confidence: decision.confidence,
      tier,
      thresholds: [
        settings.routing.easyBelow,
        settings.routing.hardFrom,
        settings.routing.minConfidence,
      ],
    };
  }
  const thresholds: Record<Exclude<App, "router">, number[]> = {
    guard: [settings.guard.reviewThreshold],
    redact: [settings.redact.piiThreshold],
    shield: [settings.shield.injectionThreshold],
    check: [settings.check.unsupportedThreshold],
    escalate: [
      settings.routing.feedbackThreshold,
      settings.routing.repeatThreshold,
    ],
  };
  const scores =
    decision.app === "escalate"
      ? decision.features
      : decision.score === undefined
        ? undefined
        : [decision.score];
  if (
    !scores ||
    ["skipped", "withheld"].includes(decision.outcome) ||
    scores.length !== thresholds[decision.app].length
  )
    return undefined;
  return { ...base, scores, thresholds: thresholds[decision.app] };
}

// What a decision-view answer says for calibration: right/wrong about whether the app stepped in, or the router's
// model labels.
export function labelOf(
  line: Pick<LogLine, "app" | "outcome">,
  feedback: Feedback,
): Label | undefined {
  if (line.app === "router")
    return feedback === "needed-stronger"
      ? "too-low"
      : feedback === "could-be-lighter"
        ? "too-high"
        : feedback === "right-model" || feedback === "right"
          ? "right"
          : undefined;
  if (feedback === "right") return acted(line) ? "yes" : "no";
  if (feedback === "wrong") return acted(line) ? "no" : "yes";
  return undefined;
}

const keyOf = (line: Pick<LogLine, "at" | "id">) =>
  `${PREFIX}${String(line.at).padStart(15, "0")}-${line.id}`;
function parseLine(value: unknown): LogLine | undefined {
  if (
    !isRecord(value) ||
    typeof value.id !== "string" ||
    typeof value.at !== "number" ||
    typeof value.session !== "string" ||
    typeof value.app !== "string" ||
    typeof value.outcome !== "string" ||
    typeof value.subject !== "string"
  )
    return undefined;
  return value as unknown as LogLine;
}

// The newest KEPT_LINES learnable decisions across sessions, kept in OpenCode's plugin storage one key per line so a
// decision writes one small value. It outlives the server; the decision view's per-session memory does not.
export class DecisionLog {
  private readonly lines = new Map<string, LogLine>();
  private readonly store: LogStore;
  private readonly settings: Settings;
  private tail: Promise<unknown> = Promise.resolve();
  persistence: "durable" | "unavailable" = "durable";
  constructor(store: LogStore, settings: Settings) {
    this.store = store;
    this.settings = settings;
  }
  async load(): Promise<void> {
    try {
      let after: string | undefined;
      const found: LogLine[] = [];
      do {
        const page = await this.store.scan({
          prefix: PREFIX,
          limit: 1000,
          ...(after ? { after } : {}),
        });
        for (const entry of page.entries) {
          const line = parseLine(entry.value);
          if (line) found.push(line);
        }
        after = page.next;
      } while (after);
      found.sort((a, b) => a.at - b.at);
      for (const line of found) this.lines.set(line.id, line);
      this.prune();
    } catch {
      this.persistence = "unavailable";
    }
  }
  private write(line: LogLine): void {
    this.tail = this.tail
      .then(() => this.store.set(keyOf(line), line))
      .catch(() => {
        this.persistence = "unavailable";
      });
  }
  private prune(): void {
    for (const line of this.lines.values()) {
      if (this.lines.size <= KEPT_LINES) break;
      this.lines.delete(line.id);
      this.tail = this.tail
        .then(() => this.store.remove(keyOf(line)))
        .catch(() => {});
    }
  }
  recorded(decision: Decision): void {
    const line = lineOf(decision, this.settings);
    if (!line) return;
    this.lines.set(line.id, line);
    this.write(line);
    this.prune();
  }
  settled(decision: Decision): void {
    const line = this.lines.get(decision.id);
    if (!line) return;
    line.outcome = decision.outcome;
    this.write(line);
  }
  labeled(decision: Decision, feedback: Feedback): void {
    const line = this.lines.get(decision.id);
    const label = line && labelOf(line, feedback);
    if (line && label) this.label(line.id, label);
  }
  label(id: string, label: Label): boolean {
    const line = this.lines.get(id);
    if (
      !line ||
      !(line.app === "router" ? ROUTER_LABELS : BINARY_LABELS).includes(
        label as never,
      )
    )
      return false;
    line.label = label;
    this.write(line);
    return true;
  }
  // Takes an answer back: the line counts as unanswered again.
  unlabel(id: string): boolean {
    const line = this.lines.get(id);
    if (!line?.label) return false;
    delete line.label;
    this.write(line);
    return true;
  }
  // How many lines came after `at`, without copying the log: the decision view asks on every refresh.
  countSince(at: number): number {
    let count = 0;
    for (const line of this.lines.values()) if (line.at > at) count++;
    return count;
  }
  has(id: string): boolean {
    return this.lines.has(id);
  }
  all(): LogLine[] {
    return structuredClone([...this.lines.values()]);
  }
  // Waits for the writes in flight, so a test or a shutdown sees them stored.
  async flushed(): Promise<void> {
    await this.tail;
  }
}
