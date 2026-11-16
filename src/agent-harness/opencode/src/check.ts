import type { Settings } from "./settings.ts";
import type { RuntimeClient } from "./runtime.ts";
import { subjectOf } from "./decisions.ts";
import type { Decision } from "./decisions.ts";
import { isRecord } from "./settings.ts";
import { probability } from "./masking.ts";

// Question framing retained from Claude revision 4021f944f.
export const HALLUCINATION = {
  hallucination: {
    type: "noul",
    instructions:
      "Does the answer contain any factual claim that is unsupported by or contradicts the supplied grounding context? Judge the answer against context, not outside knowledge.",
  },
};
const cap = (text: string, size = 8000) =>
  text.slice(0, /[\uDC00-\uDFFF]/.test(text[size] ?? "") ? size - 1 : size);
interface Turn {
  id: string;
  prompt: string;
  promptLength: number;
  eligible: Set<string>;
  attempted: Set<string>;
  overflow: boolean;
  delivered: Map<string, { tool: string; text: string; total: number }>;
  checked: Set<string>;
  controller: AbortController;
}
interface Session {
  prompts: Map<string, Turn>;
  owners: Map<string, Turn>;
  active?: Turn;
}
type RecordDecision = (input: Omit<Decision, "id" | "at">) => void;
export class AnswerCheck {
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
    const value = this.sessions.get(id) ?? {
      prompts: new Map(),
      owners: new Map(),
    };
    this.sessions.delete(id);
    this.sessions.set(id, value);
    while (this.sessions.size > this.settings.maxSessions)
      this.remove(this.sessions.keys().next().value!);
    return value;
  }
  admit(sessionID: string, id: string, text: string): void {
    if (!this.settings.apps.check || !text.trim() || text.startsWith("/"))
      return;
    const session = this.session(sessionID);
    if (session.prompts.has(id)) return;
    session.prompts.set(id, {
      id,
      prompt: cap(text),
      promptLength: text.length,
      eligible: new Set(),
      attempted: new Set(),
      overflow: false,
      delivered: new Map(),
      checked: new Set(),
      controller: new AbortController(),
    });
    while (session.prompts.size > 32) {
      const oldest = session.prompts.keys().next().value!;
      session.prompts.get(oldest)?.controller.abort();
      session.prompts.delete(oldest);
      for (const [key, turn] of session.owners)
        if (turn.id === oldest) session.owners.delete(key);
    }
  }
  activate(sessionID: string, id: string | undefined): void {
    const session = this.sessions.get(sessionID);
    if (session) session.active = id ? session.prompts.get(id) : undefined;
  }
  startedTool(sessionID: string, key: string, tool: string): void {
    if (!this.settings.apps.check || !["read", "shell"].includes(tool)) return;
    const session = this.sessions.get(sessionID),
      turn = session?.active;
    if (!turn || turn.controller.signal.aborted) return;
    if (turn.attempted.size >= 64) {
      turn.overflow = true;
      return;
    }
    turn.attempted.add(key);
    session!.owners.set(key, turn);
  }
  completedTool(sessionID: string, key: string, succeeded = true): void {
    const session = this.sessions.get(sessionID),
      turn = session?.owners.get(key);
    session?.owners.delete(key);
    if (turn && succeeded) turn.eligible.add(key);
  }
  delivered(sessionID: string, key: string, tool: string, text: string): void {
    const turn = this.sessions.get(sessionID)?.active;
    if (
      !this.settings.apps.check ||
      !turn?.eligible.has(key) ||
      !text.trim() ||
      /^vSR: (?:tool|non-text tool) content withheld/.test(text)
    )
      return;
    const retained = [...turn.delivered.entries()].reduce(
      (sum, [id, item]) => sum + (id === key ? 0 : item.text.length),
      0,
    );
    turn.delivered.set(key, {
      tool,
      text: cap(text, Math.max(0, 8000 - retained)),
      total: text.length,
    });
  }
  async complete(
    sessionID: string,
    turnID: string,
    messageID: string,
    answer: string,
  ): Promise<void> {
    const turn = this.sessions.get(sessionID)?.prompts.get(turnID);
    if (
      !this.settings.apps.check ||
      !turn ||
      turn.controller.signal.aborted ||
      turn.checked.has(messageID)
    )
      return;
    turn.checked.add(messageID);
    answer = answer.trim();
    if (!turn.delivered.size || answer.length < 40) return;
    const context = cap(
      [...turn.delivered.values()]
        .map((item) => `${item.tool}:\n${item.text}`)
        .join("\n\n"),
    );
    const contextTotal = [...turn.delivered.values()].reduce(
      (sum, item, index) =>
        sum + item.tool.length + 2 + item.total + (index ? 2 : 0),
      0,
    );
    const clippedAnswer = cap(answer);
    const checked = context.length + turn.prompt.length + clippedAnswer.length;
    const total = contextTotal + turn.promptLength + answer.length;
    const partial =
      turn.overflow ||
      checked < total ||
      turn.eligible.size > turn.delivered.size;
    const asked = await this.runtime.decide(
      { context, request: turn.prompt, answer: clippedAnswer },
      HALLUCINATION,
      this.settings.check.deadlineMs,
      turn.controller.signal,
    );
    if (turn.controller.signal.aborted || !this.settings.apps.check) return;
    const value = asked.ok ? asked.value.answers.hallucination : undefined;
    const score =
      isRecord(value) && probability(value.noul) ? value.noul : undefined;
    this.record({
      sessionID,
      turnID,
      messageID,
      app: "check",
      outcome:
        score === undefined
          ? "skipped"
          : score >= this.settings.check.unsupportedThreshold
            ? "unverified"
            : partial
              ? "partial"
              : "verified",
      reason:
        score === undefined
          ? asked.ok
            ? "invalid-response"
            : asked.code
          : partial
            ? "coverage-limit"
            : "selected",
      ...(score === undefined ? {} : { score }),
      subject: subjectOf(turn.prompt || "the answer"),
      coverage: { checked, total },
      elapsedMs: asked.elapsedMs,
    });
  }
  interrupt(sessionID: string): void {
    for (const turn of this.sessions.get(sessionID)?.prompts.values() ?? [])
      turn.controller.abort();
  }
  remove(sessionID: string): void {
    this.interrupt(sessionID);
    this.sessions.delete(sessionID);
  }
  close(): void {
    for (const id of this.sessions.keys()) this.remove(id);
  }
}
