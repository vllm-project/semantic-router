import {
  FEEDBACK_QUESTION,
  REASK_QUESTION,
  ROUTE_QUESTIONS,
} from "./questions.ts";
import { choose, routeScores } from "./routing-policy.ts";
import type { Candidate, ModelRef } from "./routing-policy.ts";
import { subjectOf } from "./decisions.ts";
import type { Decision } from "./decisions.ts";
import { isRecord } from "./settings.ts";
import type { Settings } from "./settings.ts";
import type { RuntimeClient } from "./runtime.ts";

const MAX_TEXT = 32_000;
export const RECHECK =
  "vSR: the user pushed back on the previous answer. Re-check that answer and its evidence before building on it.";
interface Prompt {
  id: string;
  text: string;
  controller: AbortController;
  checked?: Promise<boolean>;
  claimed: boolean;
  escalated: boolean;
}
interface Session {
  prompts: Map<string, Prompt>;
  jobs: Set<AbortController>;
  active?: Prompt;
  prior?: string;
  parent?: ModelRef;
}
type RecordDecision = (decision: Omit<Decision, "id" | "at">) => string;
const probability = (value: unknown): value is number =>
  typeof value === "number" &&
  Number.isFinite(value) &&
  value >= 0 &&
  value <= 1;

export class Routing {
  private readonly sessions = new Map<string, Session>();
  private readonly settings: Settings;
  private readonly runtime: Pick<RuntimeClient, "decide">;
  private readonly record: RecordDecision;
  constructor(
    settings: Settings,
    runtime: Pick<RuntimeClient, "decide">,
    record: RecordDecision,
  ) {
    this.settings = settings;
    this.runtime = runtime;
    this.record = record;
  }
  private session(id: string): Session {
    const session = this.sessions.get(id) ?? {
      prompts: new Map(),
      jobs: new Set(),
    };
    this.sessions.delete(id);
    this.sessions.set(id, session);
    if (this.sessions.size > this.settings.maxSessions)
      this.remove(this.sessions.keys().next().value!);
    return session;
  }
  admit(sessionID: string, id: string, text: string): void {
    if (!text.trim() || text.startsWith("/") || text.length > MAX_TEXT) return;
    const session = this.session(sessionID);
    if (session.prompts.has(id)) return;
    session.prompts.set(id, {
      id,
      text,
      controller: new AbortController(),
      claimed: false,
      escalated: false,
    });
    while (session.prompts.size > 32) {
      const oldest = session.prompts.keys().next().value!;
      session.prompts.get(oldest)?.controller.abort();
      session.prompts.delete(oldest);
    }
  }
  async activate(
    sessionID: string,
    messageID: string | undefined,
    parent: ModelRef,
  ): Promise<boolean> {
    const session = this.session(sessionID);
    session.parent = { ...parent };
    const prompt = messageID ? session.prompts.get(messageID) : undefined;
    session.active = prompt;
    if (!prompt) return false; // Synthetic/background messages cannot inherit a human escalation.
    if (prompt.checked) return prompt.checked;
    const prior = session.prior;
    session.prior = prompt.text;
    prompt.checked = this.checkPrompt(sessionID, prompt, prior);
    return prompt.checked;
  }
  private async checkPrompt(
    sessionID: string,
    prompt: Prompt,
    prior: string | undefined,
  ): Promise<boolean> {
    if (!this.settings.apps.escalate || !this.settings.routing.tiers.length)
      return false;
    const budget = this.settings.routing.deadlineMs;
    const [feedback, repeat] = await Promise.all([
      this.runtime.decide(
        prompt.text,
        FEEDBACK_QUESTION,
        budget,
        prompt.controller.signal,
      ),
      prior === undefined
        ? undefined
        : this.runtime.decide(
            { current: prompt.text, prior },
            REASK_QUESTION,
            budget,
            prompt.controller.signal,
          ),
    ]);
    if (prompt.controller.signal.aborted) return false;
    const answer = feedback.ok
      ? feedback.value.answers.user_feedback
      : undefined;
    const probabilities =
      isRecord(answer) && isRecord(answer.probabilities)
        ? answer.probabilities
        : {};
    const pushed = [
      probabilities.WRONG_ANSWER,
      probabilities.WANT_DIFFERENT,
    ].filter(probability);
    const repeated =
      repeat?.ok && isRecord(repeat.value.answers.reask)
        ? repeat.value.answers.reask.noul
        : undefined;
    const feedbackScore = Math.max(0, ...pushed);
    const repeatScore = probability(repeated) ? repeated : 0;
    prompt.escalated =
      feedbackScore >= this.settings.routing.feedbackThreshold ||
      repeatScore >= this.settings.routing.repeatThreshold;
    if (prompt.escalated)
      this.record({
        sessionID,
        turnID: prompt.id,
        app: "escalate",
        outcome: "escalated",
        score: Math.max(feedbackScore, repeatScore),
        features: [feedbackScore, repeatScore],
        elapsedMs: Math.max(feedback.elapsedMs, repeat?.elapsedMs ?? 0),
        reason: "pushback",
        subject: subjectOf(prompt.text),
      });
    else if (!feedback.ok || (repeat && !repeat.ok))
      this.record({
        sessionID,
        turnID: prompt.id,
        app: "escalate",
        outcome: "skipped",
        reason: !feedback.ok
          ? feedback.code
          : repeat && !repeat.ok
            ? repeat.code
            : "invalid-response",
      });
    return prompt.escalated;
  }
  async route(
    sessionID: string,
    toolCallID: string,
    task: string,
    candidates: readonly Candidate[],
  ): Promise<{ model?: string; decisionID?: string }> {
    if (!this.settings.apps.router && !this.settings.apps.escalate) return {};
    const session = this.session(sessionID),
      prompt = session.active;
    const escalated =
      this.settings.apps.escalate && prompt?.escalated && !prompt.claimed;
    // Claim synchronously before the first await: simultaneous eligible children cannot both consume it.
    if (escalated) prompt.claimed = true;
    if (!this.settings.apps.router && !escalated) return {};
    const anchor = {
      sessionID,
      toolCallID,
      ...(prompt ? { turnID: prompt.id } : {}),
      app: "router" as const,
      subject: subjectOf(task),
    };
    if (!session.parent || !this.settings.routing.tiers.length) {
      this.record({ ...anchor, outcome: "kept", reason: "unconfigured" });
      return {};
    }
    if (task.length > MAX_TEXT) {
      this.record({ ...anchor, outcome: "kept", reason: "input-too-long" });
      return {};
    }
    const parent = { ...session.parent };
    const controller = new AbortController();
    const cancelled = () => controller.abort();
    prompt?.controller.signal.addEventListener("abort", cancelled, {
      once: true,
    });
    if (prompt?.controller.signal.aborted) controller.abort();
    session.jobs.add(controller);
    let asked;
    try {
      asked = this.settings.apps.router
        ? await this.runtime.decide(
            task,
            ROUTE_QUESTIONS,
            this.settings.routing.deadlineMs,
            controller.signal,
          )
        : undefined;
    } finally {
      session.jobs.delete(controller);
      prompt?.controller.signal.removeEventListener("abort", cancelled);
    }
    if (controller.signal.aborted) return {};
    const scores = asked?.ok ? routeScores(asked.value) : undefined;
    if (asked && !scores && !escalated) {
      this.record({
        ...anchor,
        outcome: "kept",
        reason: asked.ok ? "invalid-response" : asked.code,
        elapsedMs: asked.elapsedMs,
      });
      return {};
    }
    const choice = choose(
      this.settings.routing,
      parent,
      candidates,
      scores,
      !!escalated,
    );
    const decisionID = this.record({
      ...anchor,
      outcome: choice.model ? "routed" : "kept",
      reason: choice.reason,
      ...(choice.model ? { model: choice.model } : {}),
      ...(choice.score === undefined ? {} : { score: choice.score }),
      ...(choice.confidence === undefined
        ? {}
        : { confidence: choice.confidence }),
      ...(asked ? { elapsedMs: asked.elapsedMs } : {}),
    });
    return { ...(choice.model ? { model: choice.model } : {}), decisionID };
  }
  interrupt(sessionID: string): void {
    const session = this.sessions.get(sessionID);
    session?.active?.controller.abort();
    for (const job of session?.jobs ?? []) job.abort();
    if (session) session.active = undefined;
  }
  remove(sessionID: string): void {
    const session = this.sessions.get(sessionID);
    for (const prompt of session?.prompts.values() ?? [])
      prompt.controller.abort();
    for (const job of session?.jobs ?? []) job.abort();
    this.sessions.delete(sessionID);
  }
  close(): void {
    for (const id of this.sessions.keys()) this.remove(id);
  }
}
