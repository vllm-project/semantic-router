import type { Plugin } from "@opencode/plugin";
import type { Settings } from "./settings.ts";
import { isRecord } from "./settings.ts";
import type { RuntimeClient } from "./runtime.ts";
import type { DecisionStore } from "./decisions.ts";
import { guard } from "./guard-policy.ts";
import type { GuardVerdict } from "./guard-policy.ts";
import type { Listener } from "./listener.ts";

interface Pending {
  sessionID: string;
  command: string;
  controller: AbortController;
  decisionID?: string;
  verdict?: Promise<GuardVerdict>;
}
export async function registerGuard(
  ctx: Plugin.Context,
  settings: Settings,
  runtime: RuntimeClient,
  decisions: DecisionStore,
  notify: (sessionID: string) => Promise<void>,
  listener?: Listener,
): Promise<() => void> {
  const pending = new Map<string, Pending>();
  const key = (sessionID: string, messageID: string, id: string) =>
    `${sessionID}:${messageID}:${id}`;
  const abort = new AbortController();
  const settle = (
    sessionID: string,
    messageID: string,
    id: string,
    executed: boolean,
  ) => {
    const entry = pending.get(key(sessionID, messageID, id));
    if (!entry) return;
    if (entry.decisionID)
      decisions.settleGuard(
        sessionID,
        entry.decisionID,
        executed ? "approved" : "declined",
      );
    entry.controller.abort();
    pending.delete(key(sessionID, messageID, id));
    void notify(sessionID);
  };
  void (async () => {
    for await (const event of ctx.event.subscribe({
      signal: abort.signal,
      types: [
        "session.tool.success",
        "session.tool.failed",
        "session.execution.interrupted",
        "session.deleted",
      ],
    })) {
      if (event.type === "session.tool.success")
        settle(
          event.data.sessionID,
          event.data.assistantMessageID,
          event.data.id,
          true,
        );
      else if (event.type === "session.tool.failed")
        settle(
          event.data.sessionID,
          event.data.assistantMessageID,
          event.data.id,
          event.data.executed,
        );
      else {
        for (const [id, entry] of pending)
          if (entry.sessionID === event.data.sessionID) {
            entry.controller.abort();
            if (event.type === "session.deleted") pending.delete(id);
          }
      }
    }
  })().catch(() => {});
  await ctx.tool.hook("execute.before", (event) => {
    if (
      !settings.apps.guard ||
      event.tool !== "shell" ||
      !isRecord(event.input) ||
      typeof event.input.command !== "string"
    )
      return;
    if (pending.size >= settings.maxSessions * 16)
      throw new Error(
        "vSR guard: too many pending shell checks; retry after a command finishes.",
      );
    pending.set(key(event.sessionID, event.messageID, event.id), {
      sessionID: event.sessionID,
      command: event.input.command,
      controller: new AbortController(),
    });
  });
  await ctx.permission.hook("evaluate", async (event) => {
    if (
      !settings.apps.guard ||
      event.action !== "shell" ||
      event.effect === "deny" ||
      !event.source
    )
      return;
    const entry = pending.get(
      key(event.sessionID, event.source.messageID, event.source.id),
    );
    if (!entry) return;
    const verdict = await (entry.verdict ??= guard(
      entry.command,
      settings,
      runtime,
      entry.controller.signal,
      Date.now,
      listener,
    ));
    // Only tighten. Native ask/deny rules retain their authority even for a low score.
    const nativeAsk = event.effect === "ask" && verdict.effect === "allow";
    if (verdict.effect !== "allow") {
      event.effect = verdict.effect;
      const words = verdict.rule?.description.replace(/[.\s]+$/, "");
      const message =
        verdict.reason === "listener-rule"
          ? `${words || `listener rule ${verdict.rule?.name}`}, a rule of the Router listener, holds it`
          : verdict.reason === "listener-unchecked"
            ? `the listener's required rule ${verdict.rule?.name} could not be checked (${verdict.ruleReason})`
            : verdict.reason === "fixed-rule"
              ? "a built-in rule flags a destructive or downloaded-code pattern"
              : verdict.reason === "model-risk"
                ? "the decision model flags a potentially harmful command"
                : verdict.reason === "coverage-limit"
                  ? "this command exceeds the guard's checking limit"
                  : "the guard could not finish checking this command";
      event.message =
        verdict.effect === "deny"
          ? "vSR guard: check cancelled."
          : `vSR guard: ${message}. Confirm before this command runs.`;
    }
    const decision = decisions.record({
      sessionID: event.sessionID,
      toolCallID: event.source.id,
      app: "guard",
      outcome: nativeAsk ? "held" : verdict.outcome,
      reason: nativeAsk ? "native-permission" : verdict.reason,
      elapsedMs: verdict.elapsedMs,
      ...(verdict.score === undefined ? {} : { score: verdict.score }),
      ...(verdict.labels ? { labels: verdict.labels } : {}),
    });
    entry.decisionID = decision.id;
    for (const { rule } of verdict.unchecked ?? [])
      decisions.record({
        sessionID: event.sessionID,
        toolCallID: event.source.id,
        app: "guard",
        outcome: "skipped",
        reason: "listener-unchecked",
        labels: [rule.name],
      });
    // Only execution state/metadata is needed while the native approval dialog is pending.
    entry.command = "";
    await notify(event.sessionID);
  });
  return () => {
    abort.abort();
    for (const entry of pending.values()) entry.controller.abort();
    pending.clear();
  };
}
