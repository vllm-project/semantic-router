import { registerRouting } from "./routing-hooks.ts";
import { Turns } from "./turns.ts";
import { LINES, report } from "./view.ts";
import { pricesOf } from "./prices.ts";
import { SessionArchive } from "./archive.ts";
import { registerGuard } from "./guard-hooks.ts";
import { registerCheck } from "./check-hooks.ts";
import { Controls } from "./controls.ts";
import { registerOutput } from "./output-hooks.ts";
import { readFile } from "node:fs/promises";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { Plugin } from "@opencode/plugin";
import { DecisionStore } from "./decisions.ts";
import type { Feedback } from "./decisions.ts";
import { Vsr } from "./rpc.ts";
import { RuntimeClient } from "./runtime.ts";
import { Listener } from "./listener.ts";
import { cleartextRemote, guardCanary, verify } from "./verify.ts";
import { probe } from "./probes.ts";
import { DecisionLog, acted } from "./log.ts";
import type { Label } from "./log.ts";
import { labelsOf, sample, suggest } from "./calibration.ts";
import type { Goal } from "./calibration.ts";
import { Calibrated, thresholdsOf } from "./calibrated.ts";
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
    const routingNamed =
      isRecord(ctx.options) && isRecord(ctx.options.routing)
        ? ctx.options.routing
        : {};
    // Plugin storage, as every keeper of state across a restart reads and writes it.
    const store = {
      get: (key: string) => ctx.storage.get(key),
      set: (key: string, value: unknown) =>
        ctx.storage.set(key, value as never),
      remove: (key: string) => ctx.storage.remove(key),
      scan: (options: Parameters<typeof ctx.storage.scan>[0]) =>
        ctx.storage.scan(options),
    };
    // What the plugin options name stays the person's: a Router listener's profile does not change it.
    const listener = new Listener(
      settings,
      runtime,
      [
        ...(isRecord(named)
          ? APPS.filter((app) => named[app] !== undefined)
          : []),
        ...(routingNamed.goal !== undefined ? (["goal"] as const) : []),
        ...(routingNamed.turns !== undefined ? (["turns"] as const) : []),
      ],
      store,
    );
    // `/connect` → "vSR decision runtime" names the runtime, and OpenCode's credential store keeps its token.
    const integration = await ctx.integration.transform((editor) => {
      editor.update(INTEGRATION, (entry) => {
        entry.name = "vSR decision runtime";
      });
      editor.method.update({ integrationID: INTEGRATION, method: KEY_METHOD });
      editor.method.update({ integrationID: INTEGRATION, method: OPEN_METHOD });
    });
    // The guard case, asked once per runtime at the first prompt, as the Claude Code plugin asks it at session start.
    let canary = { asked: false, running: false, note: "" };
    const askCanary = async (signal: AbortSignal) => {
      if (!settings.apps.guard || canary.asked || canary.running) return;
      const asking = canary;
      asking.running = true;
      try {
        const note = await guardCanary(settings, runtime, signal);
        // A /connect to another runtime while it asked, or one that did not answer: the next prompt asks again. A
        // runtime that answered without a guard score is asked once, as the Claude Code plugin asks.
        if (asking !== canary) return;
        if (note === undefined && runtime.status().reachable !== true) return;
        canary = { asked: true, running: false, note: note ?? "" };
        if (note) await stateChanged();
      } finally {
        asking.running = false;
      }
    };
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
      if (changed) {
        listener.forget();
        canary = { asked: false, running: false, note: "" };
      }
      return changed;
    };
    await connect();
    // `vsr-setup` installs and starts a decision runtime on this machine; OpenCode asks before each command it runs.
    const skillPath = fileURLToPath(
      new URL("../skills/vsr-setup/SKILL.md", import.meta.url),
    );
    // The skill names its scripts by `${VSR_SKILL_DIR}`, this copy's directory, so the agent runs them wherever it is installed.
    const skillText = (await readFile(skillPath, "utf8")).replaceAll(
      "${VSR_SKILL_DIR}",
      dirname(skillPath),
    );
    const skill = await ctx.skill.transform((editor) => {
      editor.add({
        id: "vsr-setup",
        name: "vsr-setup",
        description:
          "Guided setup for vSR: installs and starts vSR's decision runtime (vllm-srun with vela3-lite) on this machine's GPU or CPU, or points vSR at one that already answers, picks which of its six apps run and their thresholds, writes them, and checks each app against the runtime. Use when vSR was just installed, when the runtime check says it does not answer, or when the user wants to change vSR's runtime, apps or thresholds.",
        path: skillPath,
        content: skillText,
        // The schema brands its IDs and paths; these are plain strings of the right form.
      } as Parameters<typeof editor.add>[0]);
    });
    // `/vsr-setup`, as the Claude Code plugin's /vsr:setup: a prompt that carries the skill, so the agent need not find it.
    const setupCommand = await ctx.command.transform((editor) => {
      editor.add({
        name: "vsr-setup",
        description:
          "Set up vSR: its decision runtime (on this machine or another), its apps and thresholds",
        execute: async ({ sessionID, prompt, delivery }) => {
          await ctx.session.prompt({
            sessionID,
            text: ["Set up vSR with the vsr-setup skill.", prompt.text.trim()]
              .filter(Boolean)
              .join(" "),
            skills: [{ id: "vsr-setup" }],
            delivery,
          } as Parameters<typeof ctx.session.prompt>[0]);
        },
      });
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
          if (event.type === "session.execution.started") {
            listener.followLater();
            void askCanary(prompts.signal).catch(() => {});
          } else if (changed) await stateChanged();
        } catch {
          /* The next event tries again. */
        }
      }
    })().catch(() => {});
    const decisions = new DecisionStore(
      settings.maxSessions,
      settings.maxDecisions,
    );
    // Calibrated thresholds apply before any hook reads them; the log of learnable decisions outlives the server.
    const calibrated = new Calibrated(settings, store);
    await calibrated.load();
    decisions.thresholds = (app) => thresholdsOf(settings)[app];
    const log = new DecisionLog(store, settings);
    await log.load();
    decisions.sink = log;
    const archive = new SessionArchive(store, decisions);
    await archive.load();
    const calibrationState = () => {
      const { at } = calibrated.state();
      const since = log.countSince(at ?? 0);
      return {
        ...(at === undefined ? {} : { at }),
        since,
        remind: settings.calibrateEvery > 0 && since >= settings.calibrateEvery,
      };
    };
    // The decision model now: reading the listing again, so a model the runtime switched to is seen.
    const modelNow = async (signal?: AbortSignal) => {
      await connect();
      await runtime.discover(2000, signal);
      return runtime.model();
    };
    const appOf = (input: unknown): App => {
      if (!isRecord(input) || !APPS.includes(input.app as App))
        throw new Error("Invalid app");
      return input.app as App;
    };
    const goalOf = (input: unknown): Goal =>
      isRecord(input) && typeof input.goal === "string"
        ? (input.goal as Goal)
        : "balanced";
    const controls = new Controls(
      settings,
      decisions.snapshot("").instanceID,
      (value) => ctx.storage.set("controls.v1", value),
      () => decisions.changed(),
      (values) => listener.pin(values),
      () => listener.unpin(),
      {
        turns: routingNamed.turns !== undefined,
        calibrateEvery:
          isRecord(ctx.options) && ctx.options.calibrateEvery !== undefined,
        downgradeConfidence: routingNamed.downgradeConfidence !== undefined,
        stepsLow: routingNamed.stepsLow !== undefined,
        stepsHigh: routingNamed.stepsHigh !== undefined,
        stepsMinConfidence: routingNamed.stepsMinConfidence !== undefined,
      },
    );
    await controls.load(() => ctx.storage.get("controls.v1"));
    // Your own turns, routed when `routing.turns` is on; the view reads where they run.
    const turns = new Turns(
      settings,
      runtime,
      (input) => decisions.record(input).id,
      {
        modelOf: async (sessionID) =>
          (await ctx.session.get({ sessionID })).model ?? undefined,
        switchModel: async (sessionID, model) =>
          void (await ctx.session.switchModel({ sessionID, model })),
        prices: async () => pricesOf((await ctx.model.list()).data),
      },
    );
    turns.changed = () => decisions.changed();
    // Your turns routed again after you picked a model yourself: your turns follow the router again.
    controls.turned = (on) => {
      if (on) turns.resume();
    };
    listener.noted = () => decisions.changed();
    // The runtime check (the panel, /vsr-config, and the setup skill's vsr_check): the runtime in use, then one case per enabled app through its hook's path.
    const checkRuntime = async (signal: AbortSignal) => {
      await connect();
      const listed = await runtime.discover(3000, signal);
      const probes = await verify(settings, runtime, signal);
      await stateChanged();
      return {
        source: runtime.status().source,
        models: listed.ok ? listed.value.map((model) => model.id) : [],
        ...(runtime.named() ? { model: runtime.named() } : {}),
        probes,
        ...(cleartextRemote(runtime.url()) ? { cleartext: true } : {}),
      };
    };
    const registration = await ctx.rpc.register(Vsr, {
      snapshot: async (input) => {
        const found = decisions.snapshot(sessionID(input));
        return {
          ...found,
          // The decisions an answer teaches calibration: those in the decision log.
          answerable: found.decisions
            .filter((decision) => log.has(decision.id))
            .map((decision) => decision.id),
          apps: { ...settings.apps },
          waits: { ...settings.waits },
          controls: {
            ...controls.state(),
            note: listener.note,
            listener: listener.routed,
          },
          turns: turns.view(sessionID(input)),
          calibration: calibrationState(),
          runtime: {
            ...runtime.status(),
            ...(canary.note ? { guardNote: canary.note } : {}),
            ...(cleartextRemote(runtime.url()) ? { cleartext: true } : {}),
          },
        };
      },
      diagnose: async (_input, request) => {
        const result = await runtime.discover(3000, request.signal);
        await stateChanged();
        return result;
      },
      verify: async (_input, request) => checkRuntime(request.signal),
      calibrationReport: async (_input, request) => {
        const lines = log.all();
        const labels = labelsOf(lines);
        const state = calibrated.state();
        const thresholds = thresholdsOf(settings);
        const model = await modelNow(request.signal);
        return {
          persistence: log.persistence,
          ...calibrationState(),
          ...(model ? { model } : {}),
          ...(state.model ? { calibratedModel: state.model } : {}),
          ...(lines.length
            ? {
                first: Math.min(...lines.map((line) => line.at)),
                last: Math.max(...lines.map((line) => line.at)),
              }
            : {}),
          apps: APPS.map((app) => {
            const mine = lines.filter((line) => line.app === app);
            const answered = mine.filter((line) => labels.has(line.id)).length;
            const explicit = mine.filter((line) => line.label).length;
            return {
              app,
              enabled: settings.apps[app],
              decisions: mine.length,
              answered,
              // The report's words, as the Claude Code plugin's: what vSR did, and whose answers these are.
              acted: mine.filter(acted).length,
              explicit,
              implied: answered - explicit,
              ...(app === "router"
                ? {
                    routes: (
                      ["light", "medium", "strong", "kept"] as const
                    ).map(
                      (_, tier) =>
                        mine.filter((line) =>
                          tier === 3
                            ? line.outcome !== "routed"
                            : line.outcome === "routed" && line.tier === tier,
                        ).length,
                    ),
                  }
                : {}),
              thresholds: thresholds[app],
              calibrated: state.apps.includes(app),
            };
          }),
        };
      },
      calibrationSample: async (input) => {
        const app = appOf(input);
        return sample(
          log.all(),
          app,
          thresholdsOf(settings)[app],
          undefined,
          new Set((input as { skip?: string[] }).skip ?? []),
        ).map((line) => ({
          id: line.id,
          app: line.app,
          subject: line.subject,
          acted: acted(line),
          ...(line.scores ? { scores: line.scores } : {}),
          ...(line.level === undefined ? {} : { level: line.level }),
          ...(line.tier === undefined ? {} : { tier: line.tier }),
          thresholds: thresholdsOf(settings)[app],
        }));
      },
      calibrationLabel: async (input) => {
        if (
          !isRecord(input) ||
          typeof input.id !== "string" ||
          (typeof input.label !== "string" && input.label !== null)
        )
          throw new Error("Invalid label");
        return {
          saved:
            input.label === null
              ? log.unlabel(input.id)
              : log.label(input.id, input.label as Label),
        };
      },
      calibrationSuggest: async (input) => {
        const app = appOf(input);
        return suggest(
          log.all(),
          app,
          thresholdsOf(settings)[app],
          goalOf(input),
        );
      },
      // Fits again here, so what is applied is what the log supports now, not a value a client sends.
      calibrationApply: async (input) => {
        const app = appOf(input);
        const found = suggest(
          log.all(),
          app,
          thresholdsOf(settings)[app],
          goalOf(input),
        );
        if (!found.changed) return { applied: false, suggestion: found };
        await calibrated.apply(app, found.after, await modelNow());
        await stateChanged();
        return { applied: true, suggestion: found };
      },
      thresholds: async (input) => {
        if (isRecord(input) && input.app !== undefined) {
          const app = appOf(input);
          if (Array.isArray(input.values))
            await calibrated.set(app, input.values as number[]);
          else await calibrated.reset(app);
          decisions.changed();
        }
        const state = calibrated.state();
        const now = thresholdsOf(settings);
        return APPS.map((app) => ({
          app,
          values: now[app],
          options: calibrated.optionsOf(app),
          source: state.manual.includes(app)
            ? ("you" as const)
            : state.apps.includes(app)
              ? ("calibration" as const)
              : ("options" as const),
        }));
      },
      calibrationProbe: async (input, request) => {
        const model = await modelNow(request.signal);
        const asked = (input as { apps?: App[] }).apps;
        const rows = await probe(
          // The apps asked for, else every one that is on, as the Claude Code plugin's probe takes its enabled apps.
          asked ?? APPS.filter((app) => settings.apps[app]),
          thresholdsOf(settings),
          runtime,
          request.signal,
        );
        const before = calibrated.state().model;
        return {
          ...(model ? { model } : {}),
          ...(before ? { calibratedModel: before } : {}),
          rows,
        };
      },
      controls: async (input) => {
        const result = await controls.update(input);
        if (result.saved) await stateChanged();
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
        await notify(id);
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
    // Sessions known to have no parent; a subagent's session is adopted by its parent instead.
    const roots = new Set<string>();
    async function adopt(id: string): Promise<void> {
      for (let depth = 0; depth < 8 && !roots.has(id); depth++) {
        let parentID = decisions.parentOf(id);
        if (!parentID) {
          try {
            parentID = (await ctx.session.get({ sessionID: id })).parentID;
          } catch {
            return;
          }
          if (!parentID) {
            roots.add(id);
            while (roots.size > settings.maxSessions * 4)
              roots.delete(roots.values().next().value!);
            return;
          }
          decisions.adopt(id, parentID);
        }
        id = parentID;
      }
    }
    // Tells the views of the session and of the sessions above it, which list its decisions too.
    const notify = async (id: string): Promise<void> => {
      await adopt(id);
      for (const sessionID of [id, ...decisions.ancestors(id)]) {
        const { instanceID, revision } = decisions.snapshot(sessionID);
        await registration.events
          .emit("updated", { instanceID, sessionID, revision })
          .catch(() => {});
      }
    };
    // The setup skill's tools, as the Claude Code plugin's setup scripts: `vsr_settings` writes what the person picked
    // where /vsr-config keeps it, and `vsr_check` asks each app's case, as its `runtime.py probe --smoke` does.
    const settingsText = () => {
      const state = calibrated.state();
      const lines = thresholdsOf(settings);
      return [
        `apps: ${APPS.map((app) => `${app} ${settings.apps[app] ? "on" : "off"}`).join(", ")}`,
        `waits: ${(["redact", "shield", "guard"] as const).map((app) => `${app} ${settings.waits[app] === "done" ? "until done" : `${(settings.waits[app] as number) / 1000}s`}`).join(", ")}`,
        `router: goal ${settings.routing.goal}, your turns ${settings.routing.turns ? "routed" : "yours"}, ${settings.routing.tiers.length ? "tiers " + settings.routing.tiers.map((tier) => tier.model).join(", ") : "no tiers in the options"}`,
        "thresholds, each line in the order `thresholds` takes them:",
        ...APPS.flatMap((app) =>
          LINES[app].map(
            ([name, does], index) =>
              `- ${app} ${name.toLowerCase()} ${lines[app][index]!.toFixed(2)}${state.manual.includes(app) ? " (set by the user)" : state.apps.includes(app) ? " (calibrated)" : ""}: ${does}`,
          ),
        ),
        `runtime: ${settings.runtimeUrl}${settings.runtimeModel ? ` · model ${settings.runtimeModel}` : ""}`,
      ].join("\n");
    };
    const appFlags = {
      type: "object",
      properties: Object.fromEntries(
        APPS.map((app) => [app, { type: "boolean" }]),
      ),
      additionalProperties: false,
    };
    await ctx.tool.transform((editor) => {
      editor.add({
        name: "vsr_settings",
        description:
          "vSR's settings: with no input, says which apps are on, how long redact, shield and guard wait, the router's goal and turns, each app's thresholds and the runtime. With input, sets those the user chose in the vsr-setup skill, kept over the plugin options as /vsr-config keeps them. `reset` brings an app's thresholds back to the options'.",
        input: {
          type: "object",
          properties: {
            apps: appFlags,
            waits: {
              type: "object",
              properties: Object.fromEntries(
                ["redact", "shield", "guard"].map((app) => [
                  app,
                  { enum: [1000, 2000, 3000, "done"] },
                ]),
              ),
              additionalProperties: false,
            },
            goal: { enum: ["balanced", "quality", "save"] },
            turns: { type: "boolean" },
            thresholds: {
              type: "object",
              properties: Object.fromEntries(
                APPS.map((app) => [
                  app,
                  { type: "array", items: { type: "number" }, maxItems: 3 },
                ]),
              ),
              additionalProperties: false,
            },
            reset: { type: "array", items: { enum: [...APPS] } },
          },
          additionalProperties: false,
        },
        execute: async (input: unknown) => {
          const asked = isRecord(input) ? input : {};
          const said: string[] = [];
          const controlled = Object.fromEntries(
            ["apps", "waits", "goal", "turns"]
              .filter((key) => asked[key] !== undefined)
              .map((key) => [key, asked[key]]),
          );
          if (Object.keys(controlled).length) {
            const done = await controls.update({
              instanceID: decisions.snapshot("").instanceID,
              expectedRevision: controls.state().revision,
              ...controlled,
            });
            if (!done.saved) said.push(`not saved: ${done.reason}`);
          }
          for (const [app, values] of Object.entries(
            isRecord(asked.thresholds) ? asked.thresholds : {},
          ))
            await calibrated
              .set(app as App, values as number[])
              .catch(() =>
                said.push(`${app}: thresholds out of range, not set`),
              );
          for (const app of Array.isArray(asked.reset) ? asked.reset : [])
            await calibrated.reset(app as App);
          decisions.changed();
          return { content: [...said, settingsText()].join("\n") };
        },
      } as unknown as Parameters<typeof editor.add>[0]);
      editor.add({
        name: "vsr_check",
        description:
          "Checks vSR's decision runtime: lists its models, then asks one case per enabled app through the path its hook takes and says whether each decided as it does on vela3-lite, and how long it took. The same check as the panel's runtime line and /vsr-config run.",
        input: { type: "object", properties: {}, additionalProperties: false },
        execute: async (_input: unknown, context: { signal: AbortSignal }) => ({
          content: report(await checkRuntime(context.signal)),
        }),
      } as unknown as Parameters<typeof editor.add>[0]);
    });
    const closeRouting = await registerRouting(
      ctx,
      settings,
      runtime,
      decisions,
      notify,
      turns,
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
      await archive.close();
      await controls.close();
      closeRouting();
      closeGuard();
      closeCheck();
      closeOutput();
      prompts.abort();
      runtime.close();
      await integration.dispose();
      await skill.dispose();
      await setupCommand.dispose();
      await registration.dispose();
    };
  },
});
