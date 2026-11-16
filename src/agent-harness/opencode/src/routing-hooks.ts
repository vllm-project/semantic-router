import { knownPrices } from "./prices.ts";
import type { Plugin } from "@opencode/plugin";
import type { Settings } from "./settings.ts";
import { isRecord } from "./settings.ts";
import { Routing, RECHECK } from "./routing.ts";
import { fullModel } from "./routing-policy.ts";
import type { DecisionStore } from "./decisions.ts";
import type { RuntimeClient } from "./runtime.ts";

export async function registerRouting(
  ctx: Plugin.Context,
  settings: Settings,
  runtime: RuntimeClient,
  decisions: DecisionStore,
  notify: (sessionID: string) => Promise<void>,
): Promise<() => void> {
  const routing = new Routing(settings, runtime, (input) => {
    const decision = decisions.record(input);
    void notify(input.sessionID);
    return decision.id;
  });
  const pending = new Map<string, { sessionID: string; decisionID: string }>();
  const children = new Map<
    string,
    { sessionID: string; decisionID: string; foreground: boolean }
  >();
  const abort = new AbortController();
  async function observe(childID: string, completed = false): Promise<void> {
    const selected = children.get(childID);
    if (!selected || abort.signal.aborted) return;
    try {
      const child = await ctx.session.get({ sessionID: childID });
      const terminal =
        completed ||
        selected.foreground ||
        child.outcome !== undefined ||
        child.time.idle !== undefined;
      let costKnown = false;
      if (terminal && child.model) {
        try {
          const models = (await ctx.model.list()).data;
          const model = models.find(
            (model) =>
              model.providerID === child.model!.providerID &&
              model.id === child.model!.id,
          );
          costKnown = knownPrices(model?.cost);
        } catch {
          /* Usage remains useful without a known price. */
        }
      }
      if (abort.signal.aborted || children.get(childID) !== selected) return;
      decisions.observe(selected.sessionID, selected.decisionID, {
        childSessionID: child.id,
        ...(child.model ? { model: fullModel(child.model) } : {}),
        ...(child.outcome ? { childOutcome: child.outcome } : {}),
        ...(terminal
          ? {
              costKnown,
              usage: {
                input: child.tokens.input,
                output: child.tokens.output,
                cachedInput: child.tokens.cache.read,
                reasoning: child.tokens.reasoning,
                cost: child.cost,
              },
            }
          : {}),
      });
      if (terminal) children.delete(childID);
      await notify(selected.sessionID);
    } catch {
      /* A deleted child cannot be confirmed. */
    }
  }

  void (async () => {
    for await (const event of ctx.event.subscribe({
      signal: abort.signal,
      types: [
        "session.execution.succeeded",
        "session.execution.failed",
        "session.execution.interrupted",
        "session.deleted",
      ],
    })) {
      if (event.type === "session.deleted") {
        routing.remove(event.data.sessionID);
        decisions.remove(event.data.sessionID);
        for (const [id, selected] of children)
          if (
            id === event.data.sessionID ||
            selected.sessionID === event.data.sessionID
          )
            children.delete(id);
      } else {
        if (event.type === "session.execution.interrupted")
          routing.interrupt(event.data.sessionID);
        void observe(event.data.sessionID, true);
      }
    }
  })().catch(() => {});
  await ctx.session.hook("prompt", async (event) => {
    if (
      (!settings.apps.router && !settings.apps.escalate) ||
      !settings.routing.tiers.length
    )
      return;
    const session = await ctx.session.get({ sessionID: event.sessionID });
    if (!session.parentID)
      routing.admit(event.sessionID, event.messageID, event.prompt.text);
  });
  await ctx.session.hook("context", async (event) => {
    const lastUser = event.messages.findLast(
      (message) => message.role === "user",
    );
    const pushed = await routing.activate(
      event.sessionID,
      lastUser?.id,
      event.model,
    );
    if (pushed) event.system.push({ type: "text", text: RECHECK });
  });
  await ctx.tool.hook("execute.before", async (event) => {
    const input = event.input;
    if (
      event.tool !== "subagent" ||
      !isRecord(input) ||
      input.agent !== "general" ||
      input.model !== undefined ||
      input.sessionID !== undefined ||
      typeof input.prompt !== "string"
    )
      return;
    // Agent definitions with their own model retain it, just like explicit calls.
    let candidates;
    try {
      const agent = await ctx.agent.get({ agentID: "general" });
      if (agent.data.model) return;
      candidates = (await ctx.model.list()).data;
    } catch {
      decisions.record({
        sessionID: event.sessionID,
        toolCallID: event.id,
        app: "router",
        outcome: "kept",
        reason: "invalid-tier",
      });
      await notify(event.sessionID);
      return;
    }
    const route = await routing.route(
      event.sessionID,
      event.id,
      input.prompt,
      candidates,
    );
    if (route.model) input.model = route.model;
    if (route.decisionID && route.model) {
      pending.set(`${event.sessionID}:${event.messageID}:${event.id}`, {
        sessionID: event.sessionID,
        decisionID: route.decisionID,
      });
      while (pending.size > settings.maxSessions * 4)
        pending.delete(pending.keys().next().value!);
    }
  });
  await ctx.tool.hook("execute.after", async (event) => {
    if (event.tool !== "subagent") return;
    const key = `${event.sessionID}:${event.messageID}:${event.id}`;
    const selected = pending.get(key);
    pending.delete(key);
    if (!selected) return;
    if (event.status === "error")
      decisions.observe(selected.sessionID, selected.decisionID, {
        failed: true,
      });
    else if (
      isRecord(event.result.output) &&
      typeof event.result.output.sessionID === "string"
    ) {
      const id = event.result.output.sessionID;
      children.set(id, {
        ...selected,
        foreground: !isRecord(event.input) || event.input.background !== true,
      });
      while (children.size > settings.maxSessions * 4)
        children.delete(children.keys().next().value!);
      await observe(id);
    }
    await notify(selected.sessionID);
  });
  return () => {
    abort.abort();
    routing.close();
    pending.clear();
    children.clear();
  };
}
