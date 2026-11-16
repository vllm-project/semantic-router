import { Plugin } from "@opencode/plugin/tui";
import { createEffect, createSignal, For, Show } from "solid-js";
import { Vsr } from "./rpc.ts";
import { APPS } from "./settings.ts";
import type { App, Wait } from "./settings.ts";
import { DecisionView, emptyView, summary } from "./view.ts";
import type { ViewSnapshot, ViewState } from "./view.ts";
import type { ControlResult } from "./controls.ts";
import type { Decision, Feedback } from "./decisions.ts";

const names: Record<App, string> = {
  router: "Router",
  escalate: "Escalate",
  redact: "Redact",
  shield: "Shield",
  guard: "Guard",
  check: "Check",
};
const outcome = (decision: Decision) =>
  decision.outcome === "verified"
    ? "VERIFIED against supplied evidence"
    : decision.outcome.toUpperCase();
export default Plugin.define({
  id: "vsr.opencode.tui",
  setup(ctx) {
    const [height, setHeight] = createSignal(ctx.renderer.height);
    const resize = () => setHeight(ctx.renderer.height);
    ctx.renderer.on("resize", resize);
    const [view, update] = ctx.storage.memory<ViewState>("view.v1", {
      initial: emptyView(),
    });
    const remote = ctx.client.rpc(Vsr),
      options = { location: ctx.location };
    const controller = new DecisionView(
      {
        snapshot: async (sessionID, signal) =>
          (await remote.snapshot(
            { sessionID },
            { ...options, signal },
          )) as ViewSnapshot,
        controls: async (input) =>
          (await remote.controls(input, {
            ...options,
            signal: AbortSignal.timeout(5000),
          })) as ControlResult,
        feedback: (sessionID, decisionID, feedback) =>
          remote.feedback(
            { sessionID, decisionID, feedback },
            { ...options, signal: AbortSignal.timeout(5000) },
          ),
        diagnose: async () =>
          (await remote.diagnose(
            {},
            { ...options, signal: AbortSignal.timeout(5000) },
          )) as { ok: boolean },
      },
      (state) => update((draft) => Object.assign(draft, state)),
    );
    createEffect(() => {
      const route = ctx.ui.router.current();
      controller.select(route.type === "session" ? route.sessionID : undefined);
    });
    const offUpdated = remote.events.on("updated", (event) =>
      controller.hint(event.data),
    );
    const offState = remote.events.on("stateChanged", (event) =>
      controller.hint(event.data),
    );
    const poll = setInterval(() => {
      void controller.refresh();
    }, 3000);
    const current = () => {
      const route = ctx.ui.router.current();
      return route.type === "session" &&
        view.snapshot?.sessionID === route.sessionID
        ? view.snapshot
        : undefined;
    };
    const total = () => summary(current()?.decisions ?? []);
    async function apps() {
      const snapshot = current();
      if (!snapshot) return;
      const app = await ctx.ui.dialog.select<App | "follow">({
        title: "vSR apps",
        options: [
          ...APPS.map((value) => ({
            value,
            title: `${names[value]}: ${snapshot.apps[value] ? "on" : "off"}`,
          })),
          {
            value: "follow" as const,
            title: "Follow the Router listener",
            description:
              "Apps changed here go back to the listener's profile; plugin options stay.",
          },
        ],
      });
      if (app === "follow") await controller.change({ follow: true });
      else if (app)
        await controller.change({ apps: { [app]: !snapshot.apps[app] } });
    }
    async function waits() {
      const snapshot = current();
      if (!snapshot) return;
      const app = await ctx.ui.dialog.select({
        title: "vSR wait settings",
        options: (["redact", "shield", "guard"] as const).map((value) => ({
          value,
          title: names[value],
        })),
      });
      if (!app) return;
      const wait = await ctx.ui.dialog.select<Wait>({
        title: `${names[app]} wait`,
        current: snapshot.waits[app],
        options: ([1000, 2000, 3000, "done"] as const).map((value) => ({
          value,
          title:
            value === "done"
              ? "Until done (strict; 60-second maximum)"
              : `${value / 1000} seconds (bounded)`,
        })),
      });
      if (wait !== undefined)
        await controller.change({ waits: { [app]: wait } });
    }
    async function feedback() {
      const snapshot = current();
      if (!snapshot) return;
      const decision = await ctx.ui.dialog.select<Decision>({
        title: "vSR decision feedback",
        options: [...snapshot.decisions].reverse().map((value) => ({
          value,
          title: `${names[value.app]}: ${outcome(value)}`,
          description: `${value.toolCallID ?? value.turnID ?? value.id}${snapshot.feedback[value.id] ? ` · ${snapshot.feedback[value.id]}` : ""}`,
        })),
      });
      if (!decision) return;
      const choices: { value: Feedback; title: string }[] = [
        { value: "right", title: "Right decision" },
        { value: "wrong", title: "Wrong decision" },
        ...(decision.app === "router"
          ? [
              {
                value: "right-model" as const,
                title: "Model was appropriate",
              },
              {
                value: "needed-stronger" as const,
                title: "Needed a stronger model",
              },
              {
                value: "could-be-lighter" as const,
                title: "Could use a lighter model",
              },
            ]
          : []),
      ];
      const label = await ctx.ui.dialog.select({
        title: "Label this decision",
        options: choices,
      });
      if (label) await controller.feedback(decision, label);
    }
    const open = () => {
      const result = ctx.ui.panel.open("vsr.decisions");
      if (!result)
        ctx.ui.toast.show({
          message: "Open a session to view vSR decisions.",
        });
    };
    ctx.keymap.layer(() => ({
      mode: "global",
      commands: [
        {
          id: "vsr.panel",
          title: "vSR decisions",
          bind: "f5",
          slash: { name: "vsr" },
          palette: true,
          run: open,
        },
        {
          id: "vsr.apps",
          title: "vSR app controls",
          bind: "f6",
          slash: { name: "vsr-apps" },
          palette: true,
          run: apps,
        },
        {
          id: "vsr.waits",
          title: "vSR wait controls",
          bind: "f7",
          slash: { name: "vsr-waits" },
          palette: true,
          run: waits,
        },
        {
          id: "vsr.feedback",
          title: "vSR decision feedback",
          bind: "f8",
          slash: { name: "vsr-feedback" },
          palette: true,
          run: feedback,
        },
        {
          id: "vsr.diagnose",
          title: "vSR runtime diagnosis",
          bind: "f9",
          slash: { name: "vsr-diagnose" },
          palette: true,
          run: async () => {
            const ok = await controller.diagnose();
            ctx.ui.toast.show({
              message: ok
                ? "Decision runtime capabilities checked."
                : "Decision runtime check unavailable.",
              variant: ok ? "success" : "warning",
            });
          },
        },
        {
          id: "vsr.sessions",
          title: "vSR select session",
          bind: "f10",
          slash: { name: "vsr-sessions" },
          palette: true,
          run: async () => {
            try {
              const sessions = await ctx.client.session.list(
                { limit: 100 },
                { ...options, signal: AbortSignal.timeout(3000) },
              );
              const sessionID = await ctx.ui.dialog.select({
                title: "vSR session",
                current: view.sessionID,
                options: [...sessions.data]
                  .sort((a, b) =>
                    (a.title ?? a.id).localeCompare(b.title ?? b.id),
                  )
                  .map((session) => ({
                    value: session.id,
                    title: session.title ?? session.id,
                    description: session.parentID
                      ? "Child session"
                      : "Root session",
                  })),
              });
              if (sessionID)
                ctx.ui.router.navigate({ type: "session", sessionID });
            } catch {
              ctx.ui.toast.show({
                message: "Session list unavailable; retry when connected.",
                variant: "warning",
              });
            }
          },
        },
        {
          id: "vsr.goal",
          title: "vSR routing goal",
          slash: { name: "vsr-goal" },
          palette: true,
          run: async () => {
            const goal = await ctx.ui.dialog.select({
              title: "vSR routing goal",
              current: current()?.controls.goal,
              options: (["balanced", "quality", "save"] as const).map(
                (value) => ({ value, title: value }),
              ),
            });
            if (goal) await controller.change({ goal });
          },
        },
        {
          id: "vsr.refresh",
          title: "vSR refresh decisions",
          slash: { name: "vsr-refresh" },
          palette: true,
          run: () => {
            void controller.refresh();
          },
        },
      ],
      bindings: [
        "vsr.panel",
        "vsr.apps",
        "vsr.waits",
        "vsr.feedback",
        "vsr.diagnose",
        "vsr.sessions",
      ],
    }));
    const offComposer = ctx.ui.slot({
      append: "session.composer.top",
      render: () => (
        <text fg={ctx.theme.textMuted}>
          vSR · {view.loading ? "syncing · " : ""}
          {total().total} decisions · {total().outcomes.partial ?? 0} partial ·{" "}
          {total().outcomes.withheld ?? 0} withheld · F5
        </text>
      ),
    });
    let offPanel = () => {};
    createEffect(() => {
      const selected = ctx.ui.panel.current();
      offPanel();
      offPanel = () => {};
      if (selected?.name !== "vsr.decisions") return;
      offPanel = ctx.ui.slot({
        replace: "session.panel",
        render: (panel) =>
          panel.name !== "vsr.decisions" ? null : (
            <box
              flexDirection="column"
              padding={1}
              width={panel.width}
              height={Math.max(1, height() - 3)}
            >
              {/* The list scrolls; the summary lines keep their height instead of drawing over each other. */}
              <text flexShrink={0}>vSR decisions · {panel.sessionID}</text>
              <text flexShrink={0} fg={ctx.theme.textMuted}>
                F6 apps · F7 waits · F8 feedback · F9 diagnose · F10 session
              </text>
              <Show when={view.error}>
                <text flexShrink={0} fg={ctx.theme.warning}>
                  {view.error}
                </text>
              </Show>
              <Show
                when={current()}
                fallback={<text flexShrink={0}>Loading vSR snapshot…</text>}
              >
                {(snapshot) => (
                  <>
                    <text flexShrink={0}>
                      Runtime:{" "}
                      {snapshot().runtime.retryAt > Date.now()
                        ? "offline; retry pending"
                        : snapshot().runtime.reachable === null
                          ? "not checked"
                          : snapshot().runtime.reachable
                            ? "HTTP reachable"
                            : "offline"}{" "}
                      · {snapshot().runtime.models.length} known models
                    </text>
                    <text flexShrink={0}>
                      Apps:{" "}
                      {APPS.map(
                        (app) =>
                          `${names[app]} ${snapshot().apps[app] ? "on" : "off"}`,
                      ).join(" · ")}
                    </text>
                    <text flexShrink={0}>
                      Waits:{" "}
                      {Object.entries(snapshot().waits)
                        .map(
                          ([app, wait]) =>
                            `${app} ${wait === "done" ? "strict" : `${wait / 1000}s`}`,
                        )
                        .join(" · ")}{" "}
                      · Goal: {snapshot().controls.goal}
                    </text>
                    <text flexShrink={0}>
                      Controls: {snapshot().controls.persistence} ·{" "}
                      {view.busy ? "saving…" : "confirmed"}
                    </text>
                    <text flexShrink={0}>
                      {total().total} decisions · {total().masked} masking
                      findings · {total().outcomes.unverified ?? 0} unverified
                    </text>
                    <text flexShrink={0}>
                      Observed child tokens: {total().input} input /{" "}
                      {total().output} output / {total().cached} cached /{" "}
                      {total().reasoning} reasoning
                    </text>
                    <text flexShrink={0}>
                      {total().priced
                        ? `Estimated child cost (known prices): $${total().cost.toFixed(6)}`
                        : "Child cost: pricing unavailable"}{" "}
                      · {total().unpriced} unpriced children
                    </text>
                    <scrollbox flexGrow={1} focused={panel.focused}>
                      <For
                        each={[...snapshot().decisions].reverse()}
                        fallback={
                          <text>No decisions in this session yet.</text>
                        }
                      >
                        {(decision) => (
                          <box flexDirection="column" marginTop={1}>
                            <text>
                              {names[decision.app]} · {outcome(decision)}
                              {snapshot().feedback[decision.id]
                                ? ` · ${snapshot().feedback[decision.id]}`
                                : ""}
                            </text>
                            <text fg={ctx.theme.textMuted}>
                              {decision.toolCallID ??
                                decision.turnID ??
                                decision.id}{" "}
                              · {decision.elapsedMs ?? 0} ms
                              {decision.score === undefined
                                ? ""
                                : ` · score ${decision.score.toFixed(2)}`}
                              {decision.reason
                                ? ` · ${decision.reason.replaceAll("-", " ")}`
                                : ""}
                            </text>
                            <Show when={decision.model}>
                              <text>
                                Planned: {decision.model} · Observed:{" "}
                                {decision.observedModel ?? "pending"}
                              </text>
                            </Show>
                            <Show when={decision.coverage}>
                              <text>
                                Text checked: {decision.coverage?.checked}/
                                {decision.coverage?.total} UTF-16 units
                              </text>
                            </Show>
                          </box>
                        )}
                      </For>
                    </scrollbox>
                  </>
                )}
              </Show>
            </box>
          ),
      });
    });
    return () => {
      clearInterval(poll);
      ctx.renderer.off("resize", resize);
      offUpdated();
      offState();
      offComposer();
      offPanel();
      controller.close();
    };
  },
});
