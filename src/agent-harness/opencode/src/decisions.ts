import { hide, isTagName, secretsIn } from "./masking.ts";
import type { Counts } from "./masking.ts";
import { APPS, isRecord } from "./settings.ts";
import type { App } from "./settings.ts";

export const OUTCOMES = [
  "routed",
  "kept",
  "escalated",
  "allowed",
  "held",
  "approved",
  "declined",
  "redacted",
  "shielded",
  "verified",
  "unverified",
  "partial",
  "withheld",
  "skipped",
] as const;
export type Outcome = (typeof OUTCOMES)[number];
export const REASONS = [
  "selected",
  "unconfigured",
  "unknown-parent",
  "low-confidence",
  "quality",
  "invalid-tier",
  "pushback",
  "input-too-long",
  "cancelled",
  "timeout",
  "offline",
  "http",
  "invalid-response",
  "unsupported",
  "queue-full",
  "spawn-failed",
  "observation-unavailable",
  "fixed-rule",
  "model-risk",
  "coverage-limit",
  "native-permission",
  "placeholder-write",
  "listener-rule",
  "listener-unchecked",
  "agent-model",
] as const;
export type Reason = (typeof REASONS)[number];
export interface Usage {
  input: number;
  output: number;
  cachedInput: number;
  reasoning: number;
  cost: number;
}
export interface Decision {
  messageID?: string;
  costKnown?: boolean;
  childOutcome?: "succeeded" | "failed" | "interrupted";
  counts?: Counts;
  coverage?: { checked: number; total: number };
  reason?: Reason;
  observedModel?: string;
  childSessionID?: string;
  usage?: Usage;
  id: string;
  sessionID: string;
  turnID?: string;
  toolCallID?: string;
  app: App;
  outcome: Outcome;
  at: number;
  elapsedMs?: number;
  score?: number;
  confidence?: number;
  model?: string;
  labels?: string[];
  // What the decision was about, in at most 200 characters with fixed secret patterns hidden: the command, task,
  // prompt or output source. Calibration asks about it; nothing else of the text is kept.
  subject?: string;
  // Escalate's two scores, pushback and repeat, each held against its own threshold.
  features?: number[];
  // The app's thresholds when the decision was made, in calibration's order, so a later calibration does not redraw it.
  thresholds?: number[];
}
export const FEEDBACK = [
  "right",
  "wrong",
  "right-model",
  "needed-stronger",
  "could-be-lighter",
] as const;
export type Feedback = (typeof FEEDBACK)[number];
export const SUBJECT_CHARS = 200;
// A decision's subject: one line, fixed secret patterns hidden, at most SUBJECT_CHARS (without splitting a pair).
export function subjectOf(text: string): string {
  const line = hide(text, secretsIn(text)).text.replace(/\s+/g, " ").trim();
  if (line.length <= SUBJECT_CHARS) return line;
  // The ellipsis takes the last place; a cut between a surrogate pair drops its first half too.
  const end = /[\uD800-\uDBFF]/.test(line[SUBJECT_CHARS - 2]!)
    ? SUBJECT_CHARS - 2
    : SUBJECT_CHARS - 1;
  return line.slice(0, end) + "…";
}
export interface DecisionSink {
  recorded(decision: Decision): void;
  labeled(decision: Decision, feedback: Feedback): void;
  settled(decision: Decision): void;
}
export interface Snapshot {
  version: 1;
  instanceID: string;
  revision: number;
  sessionID: string;
  decisions: Decision[];
  feedback: Record<string, Feedback>;
}
interface Session {
  decisions: Decision[];
  feedback: Record<string, Feedback>;
}
// A finite, non-negative copy, or nothing.
function thresholdsOf(values: readonly number[] | undefined) {
  return values?.length &&
    values.length <= 3 &&
    values.every((value) => Number.isFinite(value) && value >= 0)
    ? { thresholds: [...values] }
    : {};
}
// Only a decision's clipped subject is kept of its text; prompts, outputs and model explanations are not.
export class DecisionStore {
  private readonly sessions = new Map<string, Session>();
  private readonly instanceID = crypto.randomUUID();
  private revision = 0;
  private readonly maxSessions: number;
  private readonly maxDecisions: number;
  sink: DecisionSink | undefined;
  // The thresholds in force for an app; each decision keeps a copy.
  thresholds: ((app: App) => readonly number[]) | undefined;
  // Told which session changed, or was dropped, so an archive can keep it across restarts.
  onChange: ((sessionID: string, removed: boolean) => void) | undefined;
  constructor(maxSessions: number, maxDecisions: number) {
    if (
      !Number.isInteger(maxSessions) ||
      maxSessions < 1 ||
      !Number.isInteger(maxDecisions) ||
      maxDecisions < 1
    )
      throw new Error("Invalid retention limits");
    this.maxSessions = maxSessions;
    this.maxDecisions = maxDecisions;
  }
  private touch(id: string): Session {
    let session = this.sessions.get(id);
    if (!session) session = { decisions: [], feedback: {} };
    this.sessions.delete(id);
    this.sessions.set(id, session);
    if (this.sessions.size > this.maxSessions) {
      const oldest = this.sessions.keys().next().value!;
      this.sessions.delete(oldest);
      this.onChange?.(oldest, true);
    }
    return session;
  }
  record(input: Omit<Decision, "id" | "at">): Decision {
    const decision = this.project(input, crypto.randomUUID(), Date.now());
    const session = this.touch(input.sessionID);
    session.decisions.push(decision);
    while (session.decisions.length > this.maxDecisions)
      delete session.feedback[session.decisions.shift()!.id];
    this.revision++;
    this.onChange?.(input.sessionID, false);
    this.sink?.recorded(structuredClone(decision));
    return structuredClone(decision);
  }
  // Validates a decision and copies only its known fields.
  private project(
    input: Omit<Decision, "id" | "at">,
    id: string,
    at: number,
  ): Decision {
    if (
      !input.sessionID ||
      !APPS.includes(input.app) ||
      !OUTCOMES.includes(input.outcome) ||
      (input.reason !== undefined && !REASONS.includes(input.reason))
    )
      throw new Error("Invalid decision");
    for (const [key, value] of [
      ["score", input.score],
      ["confidence", input.confidence],
      ["elapsedMs", input.elapsedMs],
    ] as const) {
      if (
        value !== undefined &&
        (!Number.isFinite(value) ||
          value < 0 ||
          (key === "confidence" && value > 1))
      )
        throw new Error("Invalid decision measurement");
    }
    if (
      input.coverage &&
      (!Number.isInteger(input.coverage.checked) ||
        !Number.isInteger(input.coverage.total) ||
        input.coverage.checked < 0 ||
        input.coverage.total < input.coverage.checked)
    )
      throw new Error("Invalid decision coverage");
    if (
      input.counts &&
      Object.entries(input.counts).some(
        ([key, value]) =>
          !isTagName(key) || !Number.isInteger(value) || value! < 0,
      )
    )
      throw new Error("Invalid masking counts");
    if (
      (input.subject !== undefined &&
        (typeof input.subject !== "string" ||
          input.subject.length > SUBJECT_CHARS)) ||
      (input.features !== undefined &&
        (!Array.isArray(input.features) ||
          input.features.length > 2 ||
          input.features.some(
            (value) => !Number.isFinite(value) || value < 0 || value > 1,
          )))
    )
      throw new Error("Invalid decision subject");
    // Explicit projection also discards unexpected properties supplied by JS callers.
    const decision: Decision = {
      ...(input.coverage
        ? {
            coverage: {
              checked: input.coverage.checked,
              total: input.coverage.total,
            },
          }
        : {}),
      ...(input.counts ? { counts: { ...input.counts } } : {}),
      ...(input.messageID ? { messageID: input.messageID } : {}),
      id,
      sessionID: input.sessionID,
      app: input.app,
      outcome: input.outcome,
      at,
      ...(input.reason ? { reason: input.reason } : {}),
      ...(input.turnID === undefined ? {} : { turnID: input.turnID }),
      ...(input.toolCallID === undefined
        ? {}
        : { toolCallID: input.toolCallID }),
      ...(input.elapsedMs === undefined ? {} : { elapsedMs: input.elapsedMs }),
      ...(input.score === undefined ? {} : { score: input.score }),
      ...(input.confidence === undefined
        ? {}
        : { confidence: input.confidence }),
      ...(input.model === undefined ? {} : { model: input.model }),
      ...(input.labels === undefined ? {} : { labels: [...input.labels] }),
      ...(input.subject === undefined ? {} : { subject: input.subject }),
      ...(input.features === undefined
        ? {}
        : { features: [...input.features] }),
      ...thresholdsOf(input.thresholds ?? this.thresholds?.(input.app)),
    };
    return decision;
  }
  // A session saved by an earlier server: each decision passes the same checks as a new one, and what fails is
  // dropped. Nothing is logged again.
  restore(sessionID: string, saved: unknown): void {
    if (!isRecord(saved) || !Array.isArray(saved.decisions)) return;
    const restored: Decision[] = [];
    for (const raw of saved.decisions.slice(-this.maxDecisions)) {
      if (
        !isRecord(raw) ||
        raw.sessionID !== sessionID ||
        typeof raw.id !== "string" ||
        !raw.id ||
        typeof raw.at !== "number" ||
        !Number.isFinite(raw.at)
      )
        continue;
      try {
        const decision = this.project(
          raw as unknown as Omit<Decision, "id" | "at">,
          raw.id,
          raw.at,
        );
        restored.push(decision);
        for (const key of ["observedModel", "childSessionID"] as const)
          if (typeof raw[key] === "string") decision[key] = raw[key];
        if (typeof raw.costKnown === "boolean")
          decision.costKnown = raw.costKnown;
        if (
          raw.childOutcome === "succeeded" ||
          raw.childOutcome === "failed" ||
          raw.childOutcome === "interrupted"
        )
          decision.childOutcome = raw.childOutcome;
        const usage = raw.usage;
        if (
          isRecord(usage) &&
          ["input", "output", "cachedInput", "reasoning", "cost"].every(
            (key) =>
              typeof usage[key] === "number" &&
              Number.isFinite(usage[key]) &&
              (usage[key] as number) >= 0,
          )
        )
          decision.usage = {
            input: usage.input as number,
            output: usage.output as number,
            cachedInput: usage.cachedInput as number,
            reasoning: usage.reasoning as number,
            cost: usage.cost as number,
          };
      } catch {
        /* A decision this version cannot read is left out. */
      }
    }
    if (!restored.length) return;
    const feedback: Record<string, Feedback> = {};
    if (isRecord(saved.feedback))
      for (const [id, value] of Object.entries(saved.feedback))
        if (
          FEEDBACK.includes(value as Feedback) &&
          restored.some((decision) => decision.id === id)
        )
          feedback[id] = value as Feedback;
    const session = this.touch(sessionID);
    session.decisions = restored;
    session.feedback = feedback;
    this.revision++;
  }
  // A session's decisions and feedback as an archive keeps them.
  saved(
    sessionID: string,
  ): { decisions: Decision[]; feedback: Record<string, Feedback> } | undefined {
    const session = this.sessions.get(sessionID);
    return session && structuredClone(session);
  }
  observe(
    sessionID: string,
    id: string,
    observation: {
      model?: string;
      childSessionID?: string;
      usage?: Usage;
      failed?: boolean;
      costKnown?: boolean;
      childOutcome?: "succeeded" | "failed" | "interrupted";
    },
  ): void {
    const decision = this.sessions
      .get(sessionID)
      ?.decisions.find((item) => item.id === id);
    if (!decision) return;
    const usage = observation.usage
      ? {
          input: observation.usage.input,
          output: observation.usage.output,
          cachedInput: observation.usage.cachedInput,
          reasoning: observation.usage.reasoning,
          cost: observation.usage.cost,
        }
      : undefined;
    if (
      usage &&
      Object.values(usage).some((value) => !Number.isFinite(value) || value < 0)
    )
      throw new Error("Invalid usage");
    if (observation.failed) {
      decision.outcome = "skipped";
      decision.reason = "spawn-failed";
    }
    if (observation.model) decision.observedModel = observation.model;
    if (observation.childSessionID)
      decision.childSessionID = observation.childSessionID;
    if (usage) decision.usage = usage;
    if (observation.childOutcome)
      decision.childOutcome = observation.childOutcome;
    if (typeof observation.costKnown === "boolean")
      decision.costKnown = observation.costKnown;
    this.revision++;
    this.onChange?.(sessionID, false);
  }
  settleGuard(
    sessionID: string,
    id: string,
    outcome: "approved" | "declined",
  ): void {
    const decision = this.sessions
      .get(sessionID)
      ?.decisions.find((item) => item.id === id);
    if (!decision || decision.app !== "guard" || decision.outcome !== "held")
      return;
    decision.outcome = outcome;
    this.revision++;
    this.onChange?.(sessionID, false);
    this.sink?.settled(structuredClone(decision));
  }
  label(sessionID: string, decisionID: string, feedback: Feedback): void {
    const session = this.sessions.get(sessionID);
    const decision = session?.decisions.find((item) => item.id === decisionID);
    if (
      !decision ||
      !FEEDBACK.includes(feedback) ||
      ((feedback.endsWith("model") ||
        feedback === "needed-stronger" ||
        feedback === "could-be-lighter") &&
        decision.app !== "router")
    )
      throw new Error("Invalid decision feedback");
    session!.feedback[decisionID] = feedback;
    this.touch(sessionID);
    this.revision++;
    this.onChange?.(sessionID, false);
    this.sink?.labeled(structuredClone(decision), feedback);
  }
  changed(): void {
    this.revision++;
  }
  snapshot(sessionID: string): Snapshot {
    const session = this.sessions.get(sessionID);
    if (session) this.touch(sessionID);
    return structuredClone({
      version: 1,
      instanceID: this.instanceID,
      revision: this.revision,
      sessionID,
      decisions: session?.decisions ?? [],
      feedback: session?.feedback ?? {},
    });
  }
  remove(sessionID: string): void {
    if (this.sessions.delete(sessionID)) this.revision++;
    this.onChange?.(sessionID, true);
  }
}
