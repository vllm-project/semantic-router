import type { Plugin } from "@opencode/plugin";
import type { Settings } from "./settings.ts";
import type { RuntimeClient } from "./runtime.ts";
import type { DecisionStore } from "./decisions.ts";
import { AnswerCheck } from "./check.ts";

export async function registerCheck(
  ctx: Plugin.Context,
  settings: Settings,
  runtime: RuntimeClient,
  decisions: DecisionStore,
  notify: (sessionID: string) => Promise<void>,
): Promise<() => void> {
  const checks = new AnswerCheck(settings, runtime, (input) => {
    decisions.record(input);
    void notify(input.sessionID);
  });
  const stop = new AbortController();
  await ctx.session.hook("prompt", async (event) => {
    if (!settings.apps.check) return;
    if (!(await ctx.session.get({ sessionID: event.sessionID })).parentID)
      checks.admit(event.sessionID, event.messageID, event.prompt.text);
  });
  await ctx.tool.hook("execute.before", (event) =>
    checks.startedTool(
      event.sessionID,
      `${event.messageID}:${event.id}`,
      event.tool,
    ),
  );
  await ctx.tool.hook("execute.after", (event) =>
    checks.completedTool(
      event.sessionID,
      `${event.messageID}:${event.id}`,
      event.status === "completed",
    ),
  );
  await ctx.session.hook("context", (event) => {
    checks.activate(
      event.sessionID,
      event.messages.findLast((message) => message.role === "user")?.id,
    );
    const assistants = new Map<string, string>();
    for (const message of event.messages) {
      if (message.role === "assistant")
        for (const part of message.content) {
          if (part.type === "tool-call")
            assistants.set(part.id, message.id ?? "context");
        }
      if (message.role !== "tool") continue;
      for (const part of message.content) {
        if (
          part.type !== "tool-result" ||
          !["read", "shell"].includes(part.name) ||
          part.result.type === "error"
        )
          continue;
        const text =
          part.result.type === "content"
            ? part.result.value
                .filter((item) => item.type === "text")
                .map((item) => item.text)
                .join("\n")
            : typeof part.result.value === "string"
              ? part.result.value
              : (JSON.stringify(part.result.value) ?? "");
        checks.delivered(
          event.sessionID,
          `${assistants.get(part.id) ?? "context"}:${part.id}`,
          part.name,
          text,
        );
      }
    }
  });
  void (async () => {
    for await (const event of ctx.event.subscribe({
      signal: stop.signal,
      types: [
        "session.step.ended",
        "session.execution.interrupted",
        "session.deleted",
      ],
    })) {
      const sessionID = event.data.sessionID;
      if (event.type === "session.deleted") checks.remove(sessionID);
      else if (event.type === "session.execution.interrupted")
        checks.interrupt(sessionID);
      else if (settings.apps.check && event.data.finish === "stop") {
        const messageID = event.data.assistantMessageID;
        // Freeze the answer's identity, then locate its human boundary by ID.
        // Read only the final answer from history; grounding comes from the outgoing hook.
        void (async () => {
          const messages = await ctx.session.context({ sessionID });
          const index = messages.findIndex(
            (message) => message.id === messageID,
          );
          const answer = messages[index];
          if (
            !answer ||
            answer.type !== "assistant" ||
            answer.finish !== "stop"
          )
            return;
          const boundary = messages
            .slice(0, index)
            .findLast(
              (message) =>
                message.type === "user" || message.type === "synthetic",
            );
          if (!boundary || boundary.type !== "user") return;
          await checks.complete(
            sessionID,
            boundary.id,
            messageID,
            answer.content
              .filter((part) => part.type === "text")
              .map((part) => part.text)
              .join("\n"),
          );
        })().catch(() => {});
      }
    }
  })().catch(() => {});
  return () => {
    stop.abort();
    checks.close();
  };
}
