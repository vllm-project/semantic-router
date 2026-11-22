import { subjectOf } from "./decisions.ts";
import { knownPrices, pricesOf } from "./prices.ts";
import type { Plugin } from "@opencode/plugin";
import type { Settings } from "./settings.ts";
import { isRecord } from "./settings.ts";
import { Routing, RECHECK } from "./routing.ts";
import { fullModel } from "./routing-policy.ts";
import type { DecisionStore } from "./decisions.ts";
import type { RuntimeClient } from "./runtime.ts";
import { estimateOf } from "./turns.ts";
import type { Turns } from "./turns.ts";

export async function registerRouting(
  ctx: Plugin.Context,
  settings: Settings,
  runtime: RuntimeClient,
  decisions: DecisionStore,
  notify: (sessionID: string) => Promise<void>,
  turns?: Turns,
): Promise<() => void> {
  const routing = new Routing(settings, runtime, (input) => {
    const decision = decisions.record(input);
    void notify(input.sessionID);
    return decision.id;
  });
  // `from`: the session's model, the one a routed subagent's tokens are priced against.
  const pending = new Map<
    string,
    { sessionID: string; decisionID: string; from?: string }
  >();
  const children = new Map<
    string,
    {
      sessionID: string;
      decisionID: string;
      from?: string;
      foreground: boolean;
    }
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
      let estimate;
      if (terminal && child.model) {
        try {
          const models = (await ctx.model.list()).data;
          const model = models.find(
            (model) =>
              model.providerID === child.model!.providerID &&
              model.id === child.model!.id,
          );
          costKnown = knownPrices(model?.cost);
          if (selected.from)
            estimate = estimateOf(
              pricesOf(models),
              `${child.model.providerID}/${child.model.id}`,
              selected.from,
              {
                input: child.tokens.input,
                // Reasoning is billed as output.
                output: child.tokens.output + child.tokens.reasoning,
                cacheRead: child.tokens.cache.read,
                cacheWrite: child.tokens.cache.write,
              },
            );
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
              ...(estimate ? { estimate } : {}),
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
        "session.model.selected",
        "session.step.ended",
      ],
    })) {
      if (event.type === "session.model.selected") {
        turns?.picked(event.data.sessionID, event.data.model);
        continue;
      }
      if (event.type === "session.step.ended") {
        const tokens = event.data.tokens;
        turns?.stepped(
          event.data.sessionID,
          {
            input: tokens.input,
            output: tokens.output,
            cacheRead: tokens.cache.read,
            cacheWrite: tokens.cache.write,
          },
          tokens.reasoning,
        );
        continue;
      }
      if (event.type === "session.deleted") {
        turns?.remove(event.data.sessionID);
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
    if (!settings.routing.tiers.length) return;
    const session = await ctx.session.get({ sessionID: event.sessionID });
    if (session.parentID) return;
    if (settings.apps.router || settings.apps.escalate)
      routing.admit(event.sessionID, event.messageID, event.prompt.text);
    // Your own turn: the router moves it before it starts when your turns are routed, and back to your model when
    // they no longer are.
    await turns
      ?.prompt(event.sessionID, event.messageID, event.prompt.text, () =>
        routing.pushback(event.sessionID, event.messageID),
      )
      .catch(() => {});
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
      typeof input.agent !== "string" ||
      input.model !== undefined ||
      input.sessionID !== undefined ||
      typeof input.prompt !== "string"
    )
      return;
    // Any subagent whose definition names no model is routed: in OpenCode such an agent, Explore among them, runs on
    // the session's model. One with its own model keeps it, and says so.
    let candidates;
    try {
      const agent = await ctx.agent.get({ agentID: input.agent });
      if (agent.data.model) {
        decisions.record({
          sessionID: event.sessionID,
          toolCallID: event.id,
          app: "router",
          outcome: "kept",
          reason: "agent-model",
          subject: subjectOf(input.prompt),
          model: fullModel(agent.data.model),
        });
        await notify(event.sessionID);
        return;
      }
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
        ...(route.from ? { from: route.from } : {}),
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
