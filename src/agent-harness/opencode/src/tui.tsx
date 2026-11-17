import { Plugin } from "@opencode/plugin/tui";
import { createEffect, createSignal, For, onCleanup, Show } from "solid-js";
import type { JSX } from "solid-js";
import type { RGBA, ScrollBoxRenderable } from "@opentui/core";
import { Vsr } from "./rpc.ts";
import { APPS } from "./settings.ts";
import type { App, Wait } from "./settings.ts";
import {
  answersOf,
  APP_LOOK,
  APP_NAMES,
  calibrationQuestion,
  DecisionView,
  emptyView,
  brief,
  clip,
  gauge,
  modelName,
  report,
  shortSubject,
  subjectLine,
  suggestionText,
  summary,
  THEME_TOKENS,
  tokenCount,
  toneOf,
} from "./view.ts";
import type {
  CalibrationQuestion,
  CalibrationReport,
  CalibrationSuggestion,
  Tone,
} from "./view.ts";
import type { ViewSnapshot, ViewState } from "./view.ts";
import type { ControlResult } from "./controls.ts";
import type { Decision } from "./decisions.ts";
import type { Verification } from "./verify.ts";

// OpenCode colors a span through its style, which this OpenTUI's span type leaves out.
// An unset color is left out: OpenTUI cannot parse one that is undefined.
const Span = (props: { fg?: unknown; bg?: unknown; children: JSX.Element }) => (
  <span
    style={
      {
        ...(props.fg === undefined ? {} : { fg: props.fg }),
        ...(props.bg === undefined ? {} : { bg: props.bg }),
      } as object
    }
  >
    {props.children}
  </span>
);
const names: Record<App, string> = APP_NAMES;
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
        // Six cases, each within its app's own wait.
        diagnose: async () =>
          (await remote.verify(
            {},
            { ...options, signal: AbortSignal.timeout(60_000) },
          )) as Verification,
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
    const call = { ...options, signal: AbortSignal.timeout(10_000) };
    // Calibration: pick an app, answer questions about its decisions nearest the line, pick a goal, then apply
    // the fitted line or keep the current one. Nothing changes until the person applies it.
    async function calibrate() {
      try {
        const found = (await remote.calibrationReport(
          {},
          call,
        )) as CalibrationReport;
        const app = await ctx.ui.dialog.select<App>({
          title: "vSR calibrate: which app?",
          options: found.apps
            .filter((item) => item.decisions > 0)
            .map((item) => ({
              value: item.app,
              title: `${names[item.app]}${item.enabled ? "" : " (off)"}`,
              description: `${item.decisions} decisions · ${item.answered} answered${item.calibrated ? " · calibrated" : ""}`,
            })),
        });
        if (!app) {
          if (!found.apps.some((item) => item.decisions > 0))
            ctx.ui.toast.show({
              message: "vSR has no decisions to calibrate from yet.",
            });
          return;
        }
        const questions = (await remote.calibrationSample(
          { app },
          call,
        )) as CalibrationQuestion[];
        for (const question of questions) {
          const asked = calibrationQuestion(question);
          const answer = await ctx.ui.dialog.select({
            title: asked.title,
            options: asked.options.map((option) => ({
              ...option,
              description: asked.description,
            })),
          });
          if (!answer || answer === "stop") break;
          if (answer !== "skip")
            await remote.calibrationLabel(
              { id: question.id, label: answer },
              call,
            );
        }
        const goal =
          app === "router"
            ? "balanced"
            : await ctx.ui.dialog.select({
                title: "vSR calibrate: what matters more?",
                options: [
                  {
                    value: "balanced" as const,
                    title: "Balanced (recommended)",
                    description: "as few wrong calls as possible either way",
                  },
                  {
                    value: "fewer-false-alarms" as const,
                    title: "Fewer false alarms",
                    description: "step in only when sure; may miss some",
                  },
                  {
                    value: "catch-more" as const,
                    title: "Catch more",
                    description: "step in on more borderline cases",
                  },
                ],
              });
        if (!goal) return;
        const suggested = (await remote.calibrationSuggest(
          { app, goal },
          call,
        )) as CalibrationSuggestion;
        if (!suggested.changed) {
          await ctx.ui.dialog.alert({
            title: `vSR calibrate: ${names[app]}`,
            message: suggestionText(suggested),
          });
          return;
        }
        const apply = await ctx.ui.dialog.confirm({
          title: `vSR calibrate: ${names[app]}`,
          message: suggestionText(suggested),
          label: { confirm: "Apply", cancel: "Keep the current lines" },
        });
        if (!apply) return;
        const done = (await remote.calibrationApply({ app, goal }, call)) as {
          applied: boolean;
        };
        ctx.ui.toast.show({
          message: done.applied
            ? `vSR ${names[app]} calibrated.`
            : `vSR ${names[app]}: nothing to change.`,
          variant: done.applied ? "success" : "info",
        });
        await controller.refresh();
      } catch {
        ctx.ui.toast.show({
          message: "vSR calibration unavailable; retry when connected.",
          variant: "warning",
        });
      }
    }
    // `/vsr` toggles the panel: with no key bound, the command is how it closes too.
    const open = () => {
      if (ctx.ui.panel.current()?.name === "vsr.decisions") {
        ctx.ui.panel.close();
        return;
      }
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
          slash: { name: "vsr" },
          palette: true,
          run: open,
        },
        {
          id: "vsr.apps",
          title: "vSR app controls",
          slash: { name: "vsr-apps" },
          palette: true,
          run: apps,
        },
        {
          id: "vsr.waits",
          title: "vSR wait controls",
          slash: { name: "vsr-waits" },
          palette: true,
          run: waits,
        },
        {
          id: "vsr.diagnose",
          title: "vSR check the decision runtime",
          slash: { name: "vsr-check" },
          palette: true,
          run: async () => {
            ctx.ui.toast.show({ message: "vSR: asking the decision runtime…" });
            const found = await controller.diagnose();
            if (!found) {
              ctx.ui.toast.show({
                message: "Decision runtime check unavailable.",
                variant: "warning",
              });
              return;
            }
            await ctx.ui.dialog.alert({
              title: "vSR decision runtime",
              message: report(found),
            });
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
          id: "vsr.calibrate",
          title: "vSR calibrate thresholds from your answers",
          slash: { name: "vsr-calibrate" },
          palette: true,
          run: calibrate,
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
      bindings: ["vsr.panel", "vsr.apps", "vsr.waits", "vsr.diagnose"],
    }));
    // The panel's list, newest first, and the decision picked in it.
    const rows = () => [...(current()?.decisions ?? [])].reverse();
    const [picked, setPicked] = createSignal<string>();
    let list: ScrollBoxRenderable | undefined;
    const pick = (id: string) => {
      setPicked(id);
      list?.scrollChildIntoView(`vsr-decision-${id}`);
    };
    const move = (step: number) => {
      const ids = rows().map((decision) => decision.id);
      if (!ids.length) return;
      const at = ids.indexOf(picked() ?? "");
      pick(ids[at < 0 ? 0 : Math.min(ids.length - 1, Math.max(0, at + step))]!);
    };
    const answerable = (decision: Decision) =>
      current()?.answerable.includes(decision.id) ?? false;
    // Saves an answer to a decision, as a button under it or its number key does.
    const answer = (decision: Decision | undefined, index: number) => {
      const choice = decision && answersOf(decision)[index];
      if (!decision || !choice) return;
      if (!answerable(decision)) {
        ctx.ui.toast.show({
          message:
            "A rule or a skipped call decided this one; calibration does not learn from it.",
        });
        return;
      }
      void controller.feedback(decision, choice.value);
    };
    const answerPicked = (index: number) =>
      answer(
        rows().find((decision) => decision.id === picked()),
        index,
      );
    // While the panel has focus, its keys pick and answer decisions and Esc closes it; at the prompt Esc stays
    // OpenCode's interrupt.
    const [focused, setFocused] = createSignal(false);
    ctx.keymap.layer(() => ({
      mode: "global",
      enabled: focused,
      commands: [
        {
          title: "Close vSR decisions",
          bind: "escape",
          run: () => ctx.ui.panel.close(),
        },
        { title: "Previous vSR decision", bind: "up", run: () => move(-1) },
        { title: "Next vSR decision", bind: "down", run: () => move(1) },
        ...[1, 2, 3].map((key) => ({
          title: `vSR answer ${key}`,
          bind: String(key),
          run: () => answerPicked(key - 1),
        })),
      ],
    }));
    // A token the theme lacks gives no color rather than a broken one.
    const tone = (name: Tone) =>
      THEME_TOKENS[name].reduce<unknown>(
        (node, key) =>
          node && typeof node === "object"
            ? (node as Record<string, unknown>)[key]
            : undefined,
        ctx.theme,
      ) as RGBA | undefined;
    const runtimeLook = (snapshot: ViewSnapshot) =>
      snapshot.runtime.retryAt > Date.now()
        ? { tone: "error" as const, words: "offline · retry pending" }
        : snapshot.runtime.reachable === null
          ? { tone: "muted" as const, words: "not checked yet" }
          : snapshot.runtime.reachable
            ? { tone: "success" as const, words: "reachable" }
            : { tone: "error" as const, words: "offline" };
    const attention = () =>
      (current()?.decisions ?? []).filter(
        (decision) => toneOf(decision) === "error",
      ).length;
    // The Claude Code plugin's mark without its ground: v in the theme's text color, S and R in the mark's hues.
    const Brand = () =>
      (
        [
          ["v", tone("text")],
          ["S", "#fdb618"],
          ["R", "#32a3ff"],
        ] as const
      ).map(([letter, color]) => (
        <b style={{ fg: color } as object}>{letter}</b>
      ));
    // The bar a decision's score draws, its thresholds marked, then the words that read it.
    const Gauge = (props: { decision: Decision }) => (
      <Show when={gauge(props.decision)}>
        {(found) => (
          <text>
            {found().parts.map((part) => (
              <Span
                fg={tone(
                  part.kind === "fill"
                    ? APP_LOOK[props.decision.app].tone
                    : part.kind === "line"
                      ? "text"
                      : "border",
                )}
              >
                {part.text}
              </Span>
            ))}
            <Span fg={tone("muted")}>{` ${found().words}`}</Span>
          </text>
        )}
      </Show>
    );
    const latest = () => current()?.decisions.at(-1);
    // Above the prompt, the newest decision as it is made, under the step it was made on.
    const offComposer = ctx.ui.slot({
      append: "session.composer.top",
      render: () => (
        <text>
          <Brand />{" "}
          {latest() ? (
            <Span bg={tone(toneOf(latest()!))} fg={tone("background")}>
              {` ${latest()!.outcome.toUpperCase()} `}
            </Span>
          ) : (
            ""
          )}
          {latest() ? (
            <Span fg={tone(APP_LOOK[latest()!.app].tone)}>
              {` ${APP_LOOK[latest()!.app].glyph} `}
            </Span>
          ) : (
            ""
          )}
          {latest() ? clip(shortSubject(latest()!), 48) : ""}
          {latest() && brief(latest()!) ? (
            <Span fg={tone(APP_LOOK[latest()!.app].tone)}>
              {`  ${brief(latest()!)}`}
            </Span>
          ) : (
            ""
          )}
          <Span fg={tone("muted")}>
            {`${latest() ? "  · " : "· "}${view.loading ? "syncing · " : ""}${total().total} decisions · /vsr`}
          </Span>
          {attention() ? (
            <Span fg={tone("error")}>{` · ${attention()} to look at`}</Span>
          ) : (
            ""
          )}
        </text>
      ),
    });
    // The session's decisions in order, one line each, while it runs.
    const SIDEBAR_ROWS = 12;
    const offSidebar = ctx.ui.slot({
      append: "sidebar.content",
      render: (input) => (
        <Show
          when={
            current()?.sessionID === input.sessionID &&
            current()!.decisions.length
          }
        >
          <box flexDirection="column" marginTop={1}>
            <text>
              <Brand />
              <Span fg={tone("muted")}>
                {` · ${total().total} decisions · /vsr`}
              </Span>
            </text>
            <Show when={current()!.decisions.length > SIDEBAR_ROWS}>
              <text fg={tone("muted")}>
                {`… ${current()!.decisions.length - SIDEBAR_ROWS} earlier`}
              </text>
            </Show>
            <For each={current()!.decisions.slice(-SIDEBAR_ROWS)}>
              {(decision) => (
                <text>
                  <Span fg={tone(APP_LOOK[decision.app].tone)}>
                    {`${APP_LOOK[decision.app].glyph} `}
                  </Span>
                  <Span fg={tone(toneOf(decision))}>
                    {decision.outcome.padEnd(10)}
                  </Span>
                  {clip(shortSubject(decision), 24)}
                </text>
              )}
            </For>
          </box>
        </Show>
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
        render: (panel) => {
          createEffect(() => setFocused(panel.focused));
          onCleanup(() => setFocused(false));
          return panel.name !== "vsr.decisions" ? null : (
            <box
              flexDirection="column"
              width={panel.width}
              height={Math.max(1, height() - 3)}
              border
              borderStyle="rounded"
              borderColor={panel.focused ? tone("focus") : tone("border")}
              paddingLeft={1}
              paddingRight={1}
            >
              <text flexShrink={0}>
                <Brand />
                {" decisions"}
                <Span fg={tone("muted")}>{` · ${panel.sessionID}`}</Span>
              </text>
              <Show when={view.error}>
                <text flexShrink={0} fg={tone("warning")}>
                  {view.error}
                </text>
              </Show>
              <Show
                when={current()}
                fallback={
                  <text flexShrink={0} fg={tone("muted")}>
                    Loading vSR snapshot…
                  </text>
                }
              >
                {(snapshot) => (
                  <>
                    {/* The list scrolls; the summary lines keep their height instead of drawing over each other. */}
                    <text flexShrink={0}>
                      <Span fg={tone(runtimeLook(snapshot()).tone)}>● </Span>
                      runtime {runtimeLook(snapshot()).words}
                      <Span fg={tone("muted")}>
                        {` · ${snapshot().runtime.models.length} ${snapshot().runtime.models.length === 1 ? "model" : "models"} · `}
                        {view.busy
                          ? "saving…"
                          : snapshot().controls.persistence === "durable"
                            ? "saved"
                            : "not saved"}
                      </Span>
                    </text>
                    <text flexShrink={0}>
                      {APPS.map((app) => (
                        <Span
                          fg={
                            snapshot().apps[app]
                              ? tone(APP_LOOK[app].tone)
                              : tone("muted")
                          }
                        >
                          {`${snapshot().apps[app] ? APP_LOOK[app].glyph : "○"} ${names[app]}  `}
                        </Span>
                      ))}
                    </text>
                    <text flexShrink={0} fg={tone("muted")}>
                      waits{" "}
                      {Object.entries(snapshot().waits)
                        .map(
                          ([app, wait]) =>
                            `${app} ${wait === "done" ? "until done" : `${wait / 1000}s`}`,
                        )
                        .join(" · ")}{" "}
                      · goal {snapshot().controls.goal}
                    </text>
                    <text flexShrink={0} marginTop={1}>
                      <strong>{total().total}</strong> decisions
                      <Span fg={tone("muted")}> · </Span>
                      <Span fg={tone(total().masked ? "purple" : "muted")}>
                        {total().masked} hidden
                      </Span>
                      <Span fg={tone("muted")}> · </Span>
                      <Span fg={tone(attention() ? "error" : "muted")}>
                        {attention()} to look at
                      </Span>
                    </text>
                    <text flexShrink={0} fg={tone("muted")}>
                      {total().children
                        ? `subagents ${total().children} · ${tokenCount(total().input)} in · ${tokenCount(total().output)} out · ${tokenCount(total().cached)} cached · ${
                            total().priced
                              ? `~$${total().cost.toFixed(4)}${total().unpriced ? ` + ${total().unpriced} unpriced` : ""}`
                              : "no prices"
                          }`
                        : "subagents none yet"}
                    </text>
                    <text
                      flexShrink={0}
                      fg={
                        snapshot().calibration.remind
                          ? tone("warning")
                          : tone("muted")
                      }
                    >
                      calibrated{" "}
                      {snapshot().calibration.at
                        ? new Date(
                            snapshot().calibration.at!,
                          ).toLocaleDateString()
                        : "never"}{" "}
                      · {snapshot().calibration.since} decisions since
                      {snapshot().calibration.remind
                        ? " · time to run /vsr-calibrate"
                        : ""}
                    </text>
                    <scrollbox
                      ref={(found: ScrollBoxRenderable) => (list = found)}
                      flexGrow={1}
                      marginTop={1}
                      focused={panel.focused}
                    >
                      <For
                        each={rows()}
                        fallback={
                          <text fg={tone("muted")}>
                            No decisions in this session yet.
                          </text>
                        }
                      >
                        {(decision) => (
                          <box
                            id={`vsr-decision-${decision.id}`}
                            flexDirection="column"
                            marginBottom={1}
                            marginLeft={
                              decision.sessionID === snapshot().sessionID
                                ? 0
                                : 2
                            }
                            // The picked decision is barred in the focus color; the others keep the bar's room.
                            border={["left"]}
                            borderStyle="heavy"
                            borderColor={tone(
                              picked() === decision.id ? "focus" : "background",
                            )}
                            paddingLeft={1}
                            onMouseDown={() => pick(decision.id)}
                          >
                            <text>
                              <Span
                                bg={tone(toneOf(decision))}
                                fg={tone("background")}
                              >
                                {` ${decision.outcome.toUpperCase()} `}
                              </Span>
                              <Span fg={tone(APP_LOOK[decision.app].tone)}>
                                {` ${APP_LOOK[decision.app].glyph} ${names[decision.app]}`}
                              </Span>
                              <Span fg={tone("muted")}>
                                {` · ${decision.elapsedMs ?? 0} ms`}
                                {decision.reason &&
                                decision.reason !== "selected"
                                  ? ` · ${decision.reason.replaceAll("-", " ")}`
                                  : ""}
                                {decision.sessionID === snapshot().sessionID
                                  ? ""
                                  : " · in a subagent"}
                              </Span>
                            </text>
                            <text>{subjectLine(decision)}</text>
                            <Gauge decision={decision} />
                            <Show when={decision.model}>
                              <text>
                                <Span fg={tone("muted")}>→ </Span>
                                {modelName(decision.model!)}
                                <Span
                                  fg={tone(
                                    decision.observedModel &&
                                      modelName(decision.observedModel) !==
                                        modelName(decision.model!)
                                      ? "warning"
                                      : "muted",
                                  )}
                                >
                                  {decision.observedModel
                                    ? modelName(decision.observedModel) ===
                                      modelName(decision.model!)
                                      ? " · ran"
                                      : ` · ran ${modelName(decision.observedModel)}`
                                    : " · pending"}
                                </Span>
                              </text>
                            </Show>
                            <Show
                              when={Object.keys(decision.counts ?? {}).length}
                            >
                              <text fg={tone("purple")}>
                                hid{" "}
                                {Object.entries(decision.counts ?? {})
                                  .map(([label, count]) => `${label} ×${count}`)
                                  .join(" · ")}
                              </text>
                            </Show>
                            <Show
                              when={decision.labels?.length && !decision.counts}
                            >
                              <text fg={tone("muted")}>
                                {decision.labels!.join(" · ")}
                              </text>
                            </Show>
                            <Show when={decision.usage}>
                              <text fg={tone("muted")}>
                                {tokenCount(decision.usage!.input)} in ·{" "}
                                {tokenCount(decision.usage!.output)} out
                              </text>
                            </Show>
                            <Show
                              when={
                                decision.coverage &&
                                decision.coverage.checked <
                                  decision.coverage.total
                              }
                            >
                              <text fg={tone("warning")}>
                                checked {decision.coverage?.checked} of{" "}
                                {decision.coverage?.total} characters
                              </text>
                            </Show>
                            <Show
                              when={answerable(decision)}
                              fallback={
                                <Show when={snapshot().feedback[decision.id]}>
                                  <text fg={tone("info")}>
                                    you said:{" "}
                                    {snapshot().feedback[
                                      decision.id
                                    ]!.replaceAll("-", " ")}
                                  </text>
                                </Show>
                              }
                            >
                              {/* The answers calibration learns from; the chosen one is filled in. */}
                              <box flexDirection="row">
                                {answersOf(decision).map((choice, index) => (
                                  <text
                                    marginRight={1}
                                    onMouseDown={(event: {
                                      stopPropagation(): void;
                                    }) => {
                                      event.stopPropagation();
                                      pick(decision.id);
                                      answer(decision, index);
                                    }}
                                  >
                                    {snapshot().feedback[decision.id] ===
                                    choice.value ? (
                                      <Span
                                        bg={tone("info")}
                                        fg={tone("background")}
                                      >
                                        {` ${index + 1} ${choice.title} `}
                                      </Span>
                                    ) : (
                                      <Span
                                        fg={tone(
                                          picked() === decision.id
                                            ? "info"
                                            : "muted",
                                        )}
                                      >
                                        {`[${index + 1} ${choice.title}]`}
                                      </Span>
                                    )}
                                  </text>
                                ))}
                              </box>
                            </Show>
                          </box>
                        )}
                      </For>
                    </scrollbox>
                  </>
                )}
              </Show>
              <text flexShrink={0} fg={tone("muted")}>
                {panel.focused
                  ? "↑↓ pick · 1-3 answer · Esc close · ctrl+x ← prompt"
                  : "click a decision to answer it · /vsr-apps · /vsr-check"}
              </text>
            </box>
          );
        },
      });
    });
    return () => {
      clearInterval(poll);
      ctx.renderer.off("resize", resize);
      offUpdated();
      offState();
      offComposer();
      offSidebar();
      offPanel();
      controller.close();
    };
  },
});
