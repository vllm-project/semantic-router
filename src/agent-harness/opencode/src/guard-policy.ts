import type { Reply, Result, RuntimeClient } from "./runtime.ts";
import { isRecord } from "./settings.ts";
import type { Settings } from "./settings.ts";
import { alwaysHeld, partsOf } from "./shell.ts";
import type { Risk } from "./shell.ts";
import type { Outcome, Reason } from "./decisions.ts";

// Question framing retained from the Claude plugin at d7ab7f587.
export const GUARD_QUESTIONS = {
  kind: {
    type: "choice",
    instructions: "What is the main risk of running this shell command?",
    require_full_input: true,
    choices: [
      {
        key: "safe",
        description:
          "a routine command that reads, builds, tests or makes ordinary reversible changes",
      },
      {
        key: "destructive",
        description:
          "deletes, overwrites or corrupts important data or system state in a way that is hard to undo",
      },
      {
        key: "exfiltration",
        description:
          "sends local files, secrets or credentials to an outside server",
      },
      {
        key: "remote_code",
        description: "downloads code from the internet and runs it",
      },
    ],
  },
  review: {
    type: "noul",
    instructions:
      "Should a person review this shell command before it runs, because it could cause serious harm that is hard to undo?",
  },
};
const RISKS: readonly Risk[] = [
  "safe",
  "destructive",
  "exfiltration",
  "remote_code",
];
const probability = (value: unknown): value is number =>
  typeof value === "number" &&
  Number.isFinite(value) &&
  value >= 0 &&
  value <= 1;
function parse(reply: Reply): { review: number; risk: Risk } | undefined {
  const kind = reply.answers.kind;
  const review = reply.answers.review;
  if (
    !isRecord(kind) ||
    !isRecord(kind.probabilities) ||
    !isRecord(review) ||
    !probability(review.noul)
  )
    return;
  const risk = RISKS.find((risk) => kind.choice === risk);
  if (!risk || !probability(kind.probabilities[risk])) return;
  return { review: review.noul, risk };
}
export interface GuardVerdict {
  effect: "allow" | "ask" | "deny";
  outcome: Outcome;
  reason: Reason;
  score?: number;
  labels?: string[];
  elapsedMs: number;
}
export async function guard(
  command: string,
  settings: Settings,
  runtime: Pick<RuntimeClient, "decide">,
  signal?: AbortSignal,
  now: () => number = Date.now,
): Promise<GuardVerdict> {
  const start = now();
  const elapsedMs = () => Math.max(0, now() - start);
  const unavailable = (reason: Reason): GuardVerdict => ({
    effect:
      reason === "cancelled"
        ? "deny"
        : settings.waits.guard === "done"
          ? "ask"
          : "allow",
    outcome:
      reason === "cancelled"
        ? "skipped"
        : settings.waits.guard === "done"
          ? "held"
          : "skipped",
    reason,
    elapsedMs: elapsedMs(),
  });
  if (signal?.aborted) return unavailable("cancelled");
  if (command.length > 32_000)
    return {
      effect: "ask",
      outcome: "held",
      reason: "coverage-limit",
      elapsedMs: elapsedMs(),
    };
  const parts = partsOf(command);
  for (const part of parts) {
    const rule = alwaysHeld(part);
    if (rule)
      return {
        effect: "ask",
        outcome: "held",
        reason: "fixed-rule",
        score: 1,
        labels: [rule.risk],
        elapsedMs: elapsedMs(),
      };
  }
  if (parts.length > 12)
    return {
      effect: "ask",
      outcome: "held",
      reason: "coverage-limit",
      elapsedMs: elapsedMs(),
    };
  const budget =
    settings.waits.guard === "done" ? 60_000 : settings.waits.guard;
  let top: ReturnType<typeof parse>;
  let failure: Reason | undefined;
  for (const part of parts) {
    const left = budget - elapsedMs();
    const result: Result<Reply> =
      left > 0
        ? await runtime.decide(
            `Tool: Bash\nCommand: ${part}`,
            GUARD_QUESTIONS,
            left,
            signal,
          )
        : { ok: false, code: "timeout", elapsedMs: 0 };
    if (signal?.aborted || (!result.ok && result.code === "cancelled"))
      return unavailable("cancelled");
    const checked = result.ok ? parse(result.value) : undefined;
    if (!checked) {
      failure ??= result.ok ? "invalid-response" : result.code;
      continue;
    }
    if (!top || checked.review > top.review) top = checked;
    if (checked.review >= settings.guard.reviewThreshold)
      return {
        effect: "ask",
        outcome: "held",
        reason: "model-risk",
        score: checked.review,
        labels: [checked.risk],
        elapsedMs: elapsedMs(),
      };
  }
  if (failure) return unavailable(failure);
  return {
    effect: "allow",
    outcome: "allowed",
    reason: "selected",
    ...(top ? { score: top.review, labels: [top.risk] } : {}),
    elapsedMs: elapsedMs(),
  };
}
