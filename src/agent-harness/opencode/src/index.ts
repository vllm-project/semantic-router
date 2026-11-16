import { registerRouting } from "./routing-hooks.ts";
import { registerGuard } from "./guard-hooks.ts";
import { registerCheck } from "./check-hooks.ts";
import { Controls } from "./controls.ts";
import { registerOutput } from "./output-hooks.ts";
import { Plugin } from "@opencode/plugin";
import { DecisionStore } from "./decisions.ts";
import type { Feedback } from "./decisions.ts";
import { Vsr } from "./rpc.ts";
import { RuntimeClient } from "./runtime.ts";
import { Listener } from "./listener.ts";
import { APPS, isRecord, parseSettings } from "./settings.ts";
import type { App } from "./settings.ts";

function sessionID(input: unknown): string {
  if (
    !isRecord(input) ||
    typeof input.sessionID !== "string" ||
    input.sessionID.length === 0 ||
    input.sessionID.length > 256
  )
    throw new Error("Invalid session ID");
  return input.sessionID;
}
export default Plugin.define({
  id: "vsr.opencode",
  async setup(ctx) {
    const settings = parseSettings(ctx.options);
    const runtime = new RuntimeClient(settings);
    const named = isRecord(ctx.options) ? ctx.options.apps : undefined;
    const listener = new Listener(
      settings,
      runtime,
      isRecord(named) ? APPS.filter((app) => named[app] !== undefined) : [],
      {
        get: (key) => ctx.storage.get(key),
        set: (key, value) => ctx.storage.set(key, value),
        remove: (key) => ctx.storage.remove(key),
      },
    );
    // The first prompt reads the listing: loading the plugin calls no runtime.
    const prompts = new AbortController();
    void (async () => {
      for await (const _ of ctx.event.subscribe({
        signal: prompts.signal,
        types: ["session.execution.started"],
      }))
        listener.followLater();
    })().catch(() => {});
    const decisions = new DecisionStore(
      settings.maxSessions,
      settings.maxDecisions,
    );
    const controls = new Controls(
      settings,
      decisions.snapshot("").instanceID,
      (value) => ctx.storage.set("controls.v1", value),
      () => decisions.changed(),
      (apps) => listener.pin(apps),
      () => listener.unpin(),
    );
    await controls.load(() => ctx.storage.get("controls.v1"));
    const registration = await ctx.rpc.register(Vsr, {
      snapshot: async (input) => ({
        ...decisions.snapshot(sessionID(input)),
        apps: { ...settings.apps },
        waits: { ...settings.waits },
        controls: controls.state(),
        runtime: runtime.status(),
      }),
      diagnose: async (_input, request) => {
        const result = await runtime.discover(3000, request.signal);
        decisions.changed();
        const { instanceID, revision } = decisions.snapshot("");
        await registration.events
          .emit("stateChanged", {
            instanceID,
            revision,
          })
          .catch(() => {});
        return result;
      },
      controls: async (input) => {
        const result = await controls.update(input);
        if (result.saved) {
          const { instanceID, revision } = decisions.snapshot("");
          await registration.events
            .emit("stateChanged", {
              instanceID,
              revision,
            })
            .catch(() => {});
        }
        return result;
      },
      feedback: async (input) => {
        const id = sessionID(input);
        if (
          !isRecord(input) ||
          typeof input.decisionID !== "string" ||
          typeof input.feedback !== "string"
        )
          throw new Error("Invalid feedback");
        decisions.label(id, input.decisionID, input.feedback as Feedback);
        const { instanceID, revision } = decisions.snapshot(id);
        await registration.events
          .emit("updated", {
            instanceID,
            sessionID: id,
            revision,
          })
          .catch(() => {});
        return { saved: true };
      },
    });
    const notify = async (id: string): Promise<void> => {
      const { instanceID, revision } = decisions.snapshot(id);
      await registration.events
        .emit("updated", { instanceID, sessionID: id, revision })
        .catch(() => {});
    };
    const closeRouting = await registerRouting(
      ctx,
      settings,
      runtime,
      decisions,
      notify,
    );
    const closeGuard = await registerGuard(
      ctx,
      settings,
      runtime,
      decisions,
      notify,
      listener,
    );
    const closeOutput = await registerOutput(
      ctx,
      settings,
      runtime,
      decisions,
      notify,
      listener,
    );
    const closeCheck = await registerCheck(
      ctx,
      settings,
      runtime,
      decisions,
      notify,
    );
    return async () => {
      await controls.close();
      closeRouting();
      closeGuard();
      closeCheck();
      closeOutput();
      prompts.abort();
      runtime.close();
      await registration.dispose();
    };
  },
});
