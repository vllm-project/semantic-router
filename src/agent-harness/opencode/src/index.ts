import { registerRouting } from "./routing-hooks.ts";
import { registerGuard } from "./guard-hooks.ts";
import { registerCheck } from "./check-hooks.ts";
import { Controls } from "./controls.ts";
import { registerOutput } from "./output-hooks.ts";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { Plugin } from "@opencode/plugin";
import { DecisionStore } from "./decisions.ts";
import type { Feedback } from "./decisions.ts";
import { Vsr } from "./rpc.ts";
import { RuntimeClient } from "./runtime.ts";
import { Listener } from "./listener.ts";
import { verify } from "./verify.ts";
import {
  INTEGRATION,
  KEY_METHOD,
  OPEN_METHOD,
  targetOf,
} from "./connection.ts";
import type { Target } from "./connection.ts";
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
    // `/connect` → "vSR decision runtime" names the runtime, and OpenCode's credential store keeps its token.
    const integration = await ctx.integration.transform((editor) => {
      editor.update(INTEGRATION, (entry) => {
        entry.name = "vSR decision runtime";
      });
      editor.method.update({ integrationID: INTEGRATION, method: KEY_METHOD });
      editor.method.update({ integrationID: INTEGRATION, method: OPEN_METHOD });
    });
    const connect = async (): Promise<boolean> => {
      let target: Target | undefined;
      try {
        const active = await ctx.integration.connection.active(INTEGRATION);
        target = active
          ? targetOf(await ctx.integration.connection.resolve(active))
          : undefined;
      } catch {
        return false; // The credential store did not answer: keep the runtime in use.
      }
      const changed = runtime.connect(target);
      if (changed) listener.forget();
      return changed;
    };
    await connect();
    // `vsr-setup` installs and starts a decision runtime on this machine; OpenCode asks before each command it runs.
    const skillPath = fileURLToPath(
      new URL("../skills/vsr-setup/SKILL.md", import.meta.url),
    );
    const skillText = await readFile(skillPath, "utf8");
    const skill = await ctx.skill.transform((editor) => {
      editor.add({
        id: "vsr-setup",
        name: "vsr-setup",
        description:
          "Installs and starts vSR's decision runtime (vllm-srun with vela3-lite) on this machine's GPU or CPU, or points vSR at one that already answers, then hands over to /connect and F9. Use when vSR was just installed, when F9 says the runtime does not answer, or when the user wants to change where vSR's decision model runs.",
        path: skillPath,
        content: skillText,
        // The schema brands its IDs and paths; these are plain strings of the right form.
      } as Parameters<typeof editor.add>[0]);
    });
    // The first prompt reads the listing: loading the plugin calls no runtime.
    const prompts = new AbortController();
    void (async () => {
      for await (const event of ctx.event.subscribe({
        signal: prompts.signal,
        types: [
          "session.execution.started",
          "credential.updated",
          "credential.switched",
        ],
      })) {
        // One event's failure (the RPC is registered after this loop starts) must not end the subscription.
        try {
          const changed = await connect();
          if (event.type === "session.execution.started")
            listener.followLater();
          else if (changed) await stateChanged();
        } catch {
          /* The next event tries again. */
        }
      }
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
      // F9: the runtime in use, then one case per enabled app through the path its hook takes.
      verify: async (_input, request) => {
        await connect();
        const listed = await runtime.discover(3000, request.signal);
        const probes = await verify(settings, runtime, request.signal);
        await stateChanged();
        return {
          source: runtime.status().source,
          models: listed.ok ? listed.value.map((model) => model.id) : [],
          probes,
        };
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
    async function stateChanged(): Promise<void> {
      decisions.changed();
      const { instanceID, revision } = decisions.snapshot("");
      await registration.events
        .emit("stateChanged", { instanceID, revision })
        .catch(() => {});
    }
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
      await integration.dispose();
      await skill.dispose();
      await registration.dispose();
    };
  },
});
