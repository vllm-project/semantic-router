import { APPS } from "./settings.ts";
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
export interface Decision {
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
}
export const FEEDBACK = [
  "right",
  "wrong",
  "right-model",
  "needed-stronger",
  "could-be-lighter",
] as const;
export type Feedback = (typeof FEEDBACK)[number];
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
// Text, prompts, commands, file paths and model-provided explanations are deliberately absent.
export class DecisionStore {
  private readonly sessions = new Map<string, Session>();
  private readonly instanceID = crypto.randomUUID();
  private revision = 0;
  private readonly maxSessions: number;
  private readonly maxDecisions: number;
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
    if (this.sessions.size > this.maxSessions)
      this.sessions.delete(this.sessions.keys().next().value!);
    return session;
  }
  record(input: Omit<Decision, "id" | "at">): Decision {
    if (
      !input.sessionID ||
      !APPS.includes(input.app) ||
      !OUTCOMES.includes(input.outcome)
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
    // Explicit projection also discards unexpected properties supplied by JS callers.
    const decision: Decision = {
      id: crypto.randomUUID(),
      sessionID: input.sessionID,
      app: input.app,
      outcome: input.outcome,
      at: Date.now(),
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
    };
    const session = this.touch(input.sessionID);
    session.decisions.push(decision);
    while (session.decisions.length > this.maxDecisions)
      delete session.feedback[session.decisions.shift()!.id];
    this.revision++;
    return structuredClone(decision);
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
  }
}
