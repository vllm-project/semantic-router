import { registerRouting } from "./routing-hooks.ts";
import { registerGuard } from "./guard-hooks.ts";
import { Plugin } from "@opencode/plugin";
import { DecisionStore } from "./decisions.ts";
import type { Feedback } from "./decisions.ts";
import { Vsr } from "./rpc.ts";
import { RuntimeClient } from "./runtime.ts";
import { isRecord, parseSettings } from "./settings.ts";

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
    const decisions = new DecisionStore(
      settings.maxSessions,
      settings.maxDecisions,
    );
    const registration = await ctx.rpc.register(Vsr, {
      snapshot: async (input) => ({
        ...decisions.snapshot(sessionID(input)),
        apps: settings.apps,
        waits: settings.waits,
        runtime: runtime.status(),
      }),
      diagnose: async (_input, request) =>
        await runtime.discover(3000, request.signal),
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
        await registration.events.emit("updated", {
          instanceID,
          sessionID: id,
          revision,
        });
        return { saved: true };
      },
    });
    const notify = async (id: string): Promise<void> => {
      const { instanceID, revision } = decisions.snapshot(id);
      try {
        await registration.events.emit("updated", {
          instanceID,
          sessionID: id,
          revision,
        });
      } catch {
        /* UI transport must not block routing. */
      }
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
    );
    return async () => {
      closeRouting();
      closeGuard();
      runtime.close();
      await registration.dispose();
    };
  },
});
