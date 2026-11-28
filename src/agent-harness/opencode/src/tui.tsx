import { Plugin } from "@opencode/plugin/tui";
import { createEffect, createSignal, For, onCleanup, Show } from "solid-js";
import type { JSX } from "solid-js";
import type { RGBA, ScrollBoxRenderable } from "@opentui/core";
import { MIN_LABELS } from "./calibration.ts";
import { TUNES } from "./controls.ts";
import type { Tune } from "./controls.ts";
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
  details,
  gauge,
  modelChanged,
  modelName,
  probeText,
  report,
  shortSubject,
  subjectLine,
  homeShort,
  suggestionsText,
  listed,
  APP_WORDS,
  LINES,
  calibrationReportText,
  savingWords,
  summary,
  THEME_TOKENS,
  tokenCount,
  toneOf,
  turnRouteReason,
  why,
} from "./view.ts";
import type {
  CalibrationQuestion,
  CalibrationReport,
  CalibrationSuggestion,
  ProbeResult,
  Tone,
} from "./view.ts";
import type { ViewSnapshot, ViewState } from "./view.ts";
import type { ControlResult } from "./controls.ts";
import type { Decision } from "./decisions.ts";
import type { Verification } from "./verify.ts";

// OpenCode colors a span through its style, which this OpenTUI's span type leaves out.
// An unset color is left out: OpenTUI cannot parse one that is undefined.
// /vsr-config's rows, as the Claude Code plugin's pane orders its apps.
const CONTROL_ROWS: readonly (readonly App[])[] = [
  ["guard", "check", "shield"],
  ["redact", "router", "escalate"],
];
// The panel's apps, one to a row with its own controls, by what they watch: outputs, commands, models, answers.
const PANEL_APPS: readonly App[] = [
  "redact",
  "shield",
  "guard",
  "router",
  "escalate",
  "check",
];
const APP_COLUMN = "● escalate".length + 3;
const KIND_COLUMN = "turns".length + 2;
// The rule between two decisions, dotted so it reads lighter than a section's.
const DOTTED = {
  topLeft: "┄",
  topRight: "┄",
  bottomLeft: "┄",
  bottomRight: "┄",
  horizontal: "┄",
  vertical: " ",
  topT: "┄",
  bottomT: "┄",
  leftT: "┄",
  rightT: "┄", // codespell:ignore
  cross: "┄",
};
const WAIT_CHOICES = [1000, 2000, 3000, "done"] as const;
const waitWords = (wait: Wait) =>
  wait === "done" ? "until done" : `${wait / 1000}s`;
const GOALS = ["balanced", "quality", "save"] as const;
// A model picked by the person hands this session's turns back to them until they pick routed again.
const turnsRouted = (snapshot: ViewSnapshot): boolean =>
  snapshot.controls.turns && !snapshot.turns.paused;
// The /vsr-config names of the router's lines calibration does not fit.
const TUNE_NAMES: Record<Tune, string> = {
  downgradeConfidence: "smaller-model line",
  stepsLow: "few-steps line",
  stepsHigh: "many-steps line",
  stepsMinConfidence: "steps confidence line",
};
// What each /vsr-config row does and what it depends on: the Claude Code plugin's /config descriptions (plugin.json
// userConfig), with the agent for Claude and your three tiers for Haiku, Sonnet and Opus.
const CONFIG_TIPS: Record<string, string> = {
  router:
    "Routes each new subagent to your light, medium or strong tier by how hard its task is, and sets its effort by how many steps it needs. Its lines, goal and turns do nothing while it is off.",
  escalate:
    "When you push back on an answer, tells the agent to re-check it and sends the next subagent one tier up, or your turn itself with your turns routed. Its two lines do nothing while it is off.",
  redact:
    "Hides personal data and secrets in what the agent reads from files and shell output. Its hiding line and wait do nothing while it is off.",
  shield:
    "Tells the agent to treat a file or page as data when it tries to give an AI agent instructions. Its warning line and wait do nothing while it is off.",
  guard:
    "Holds a shell command for your confirmation when it could cause serious harm. Its review line and wait do nothing while it is off.",
  check:
    "Checks an answer against the files and command output the agent read that turn. Off by default: on a CPU runtime a long answer outlasts its wait, so turn it on with a GPU or remote runtime.",
  redactWait:
    "How long redact waits for the decision model on a file or command output: 1, 2 or 3 seconds, then what it read in time is hidden and the rest goes unchecked. until done: wait until the check finishes, at most 60 s; if it cannot finish, the output is withheld.",
  shieldWait:
    "How long shield waits for the decision model on a file or page: 1, 2 or 3 seconds, then what it read in time is judged and the rest goes unchecked. until done: wait until the check finishes, at most 60 s; if it cannot finish, the output is withheld.",
  guardWait:
    "How long guard waits for the decision model on a shell command, all its steps together: 1, 2 or 3 seconds, then a command it could not check runs and only its fixed rules hold. until done: wait until every step is checked, at most 60 s; if one cannot be, the command is held for your confirmation.",
  goal: "quality: never a smaller tier than yours. balanced: a smaller tier only when the router is sure, at the smaller-model line. save: a smaller tier whenever it answers clearly enough to route, at the confidence line, and the light tier for your turns too.",
  turns:
    "routed: the router also picks the tier and effort of each prompt you send. A turn keeps its model through its tool calls, a short reply keeps the last turn's, and a move to a smaller tier waits while rewriting the prompt cache costs more than it saves. A model you pick yourself pauses it until you pick routed again. yours: your turns run on the model you picked.",
  downgradeConfidence:
    "With the balanced goal, the confidence a move to a smaller tier than yours needs (0 to 1); below it the subagent or your turn keeps its model. save uses the confidence line instead, and quality never goes smaller.",
  stepsLow:
    "Expected reasoning steps (0 to 2) below which a subagent or your turn starts at its tier's lowest effort. Never above the many-steps line.",
  stepsHigh:
    "Expected reasoning steps (0 to 2) at or above which a subagent or your turn starts at its tier's highest effort; in between, the middle one. Never below the few-steps line.",
  stepsMinConfidence:
    "Below this confidence (0 to 1) the steps answer is ignored and a subagent or your turn starts at its tier's middle effort.",
  every:
    "Decisions since the last calibration before the panel suggests /vsr-calibrate. 0 turns it off.",
  follow:
    "Hands what you changed here and on the panel back to the Router listener's vsr profile; what the plugin options name stays yours. Only with a Router listener.",
  checkRuntime:
    "Asks the decision runtime one case per enabled app through the path its hook takes: whether each decides as on vela3-lite, and how long it took.",
};
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
const BRAND_YELLOW = "#fdb618";
const BRAND_BLUE = "#32a3ff";

// The four-row block mark from the Claude Code plugin's mark-art.ts.
const SidebarWordmark = () => (
  <box flexDirection="row" alignItems="center" gap={2}>
    <box flexDirection="column" flexShrink={0} width={8}>
      <text wrapMode="none">
        <Span fg={BRAND_YELLOW}>▁▁▁</Span>
        {"  "}
        <Span fg={BRAND_BLUE}>▃▅▋</Span>
      </text>
      <text wrapMode="none">
        <Span fg={BRAND_YELLOW}>▝██▍</Span>
        <Span fg={BRAND_BLUE}>▐██▏</Span>
      </text>
      <text wrapMode="none">
        {" "}
        <Span fg={BRAND_YELLOW}>▝█▍</Span>
        <Span fg={BRAND_BLUE}>██▌</Span>
      </text>
      <text wrapMode="none">
        {"  "}
        <Span fg={BRAND_YELLOW}>▝</Span>
        <Span fg={BRAND_YELLOW} bg={BRAND_BLUE}>
          ▍
        </Span>
        <Span fg={BRAND_BLUE}>██</Span>
      </text>
    </box>
    <text>
      <b>vLLM Semantic Router</b>
    </text>
  </box>
);
const names: Record<App, string> = APP_NAMES;
export default Plugin.define({
  id: "vsr.opencode.tui",
  setup(ctx) {
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
    // What a Router listener's profile changes, said once in a toast when it is new, as the Claude Code plugin says it.
    let noted = "";
    createEffect(() => {
      const note = current()?.controls.note;
      if (note === undefined || note === noted) return;
      noted = note;
      if (note) ctx.ui.toast.show({ message: note, variant: "info" });
    });
    // A model that scores the guard case under the review line, said once, as the Claude Code plugin says it.
    let guardNoted = "";
    createEffect(() => {
      const note = current()?.runtime.guardNote;
      if (!note || note === guardNoted) return;
      guardNoted = note;
      ctx.ui.toast.show({ message: note, variant: "warning" });
    });
    // A model you picked yourself pauses the routing of your turns: said once, as the Claude Code plugin says it.
    let paused: string | undefined;
    createEffect(() => {
      const now = current()?.turns.paused;
      if (now === paused) return;
      paused = now;
      if (now)
        ctx.ui.toast.show({
          message: `vSR router: you picked ${now}, so your turns stay on it · pick routed again to route them`,
          variant: "info",
        });
    });
    // The runtime check, as the Claude Code plugin's setup verifies it: one case per enabled app. The panel's runtime
    // line and /vsr-config run it.
    async function runCheck(): Promise<void> {
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
    }
    // The known cases asked again, as the Claude Code plugin's calibrate probes when asked; /vsr-calibrate offers it.
    async function runProbe(): Promise<void> {
      const result = await probeNow();
      if (!result) {
        ctx.ui.toast.show({
          message: "vSR probe unavailable; retry when connected.",
          variant: "warning",
        });
        return;
      }
      await wide(
        ctx.ui.dialog.alert({ title: "vSR probe", message: probeText(result) }),
      );
    }
    // `/vsr-config`, as the Claude Code plugin's /config rows: every setting with its value, changed in place. What it
    // sets is kept in plugin storage over the options, and an option you edit in opencode.jsonc later wins again.
    type ConfigRow =
      | { kind: "app"; app: App }
      | { kind: "wait"; app: "redact" | "shield" | "guard" }
      | { kind: "goal" }
      | { kind: "turns" }
      | { kind: "line"; app: App; index: number }
      | { kind: "tune"; key: Tune }
      | { kind: "every" }
      | { kind: "follow" }
      | { kind: "check" };
    const CONFIG_ROWS: ConfigRow[] = [
      ...CONTROL_ROWS.flat().map((app) => ({ kind: "app" as const, app })),
      ...(["redact", "shield", "guard"] as const).map((app) => ({
        kind: "wait" as const,
        app,
      })),
      { kind: "goal" },
      { kind: "turns" },
      ...CONTROL_ROWS.flat().flatMap((app): ConfigRow[] => [
        ...LINES[app].map((_, index) => ({
          kind: "line" as const,
          app,
          index,
        })),
        // After the router's own lines, the ones calibration does not fit: the balanced goal's line for a smaller tier
        // and the steps answer's three (the Claude Code plugin's routerDowngradeConfidence and routerSteps*).
        ...(app === "router"
          ? (Object.keys(TUNES) as Tune[]).map((key) => ({
              kind: "tune" as const,
              key,
            }))
          : []),
      ]),
      { kind: "every" },
      // Two actions, as the Claude Code plugin's `/vsr follow` and its setup's check.
      { kind: "follow" },
      { kind: "check" },
    ];
    type Lines = {
      app: App;
      values: number[];
      options: number[];
      source: "options" | "calibration" | "you";
    }[];
    // The largest a threshold may be: the router's two difficulty lines run to 2, every other line to 1.
    const ceilingOf = (app: App, index: number) =>
      app === "router" && index < 2 ? 2 : 1;
    async function config(at = 0): Promise<void> {
      if (!current()) return;
      let lines: Lines;
      try {
        lines = (await remote.thresholds({}, call())) as Lines;
      } catch {
        ctx.ui.toast.show({
          message: "vSR settings unavailable; retry when connected.",
          variant: "warning",
        });
        return;
      }
      const [found, setFound] = createSignal(lines);
      const [cursor, setCursor] = createSignal(at);
      const lineOf = (app: App) => found().find((item) => item.app === app)!;
      // A threshold is asked for in OpenCode's own prompt, after which the list opens again where it was.
      const ask = async (index: number) => {
        const row = CONFIG_ROWS[index]!;
        if (row.kind !== "line" && row.kind !== "every" && row.kind !== "tune")
          return;
        ctx.ui.dialog.clear();
        const now =
          row.kind === "every"
            ? String(current()?.controls.calibrateEvery ?? 0)
            : row.kind === "tune"
              ? (current()?.controls[row.key] ?? 0).toFixed(2)
              : lineOf(row.app).values[row.index]!.toFixed(2);
        const typed = await ctx.ui.dialog.prompt({
          title:
            row.kind === "every"
              ? "vSR: calibrate every how many decisions? (0 is off)"
              : row.kind === "tune"
                ? `vSR ${TUNE_NAMES[row.key]} (router)`
                : `vSR ${LINES[row.app][row.index]![0]} (${row.app})`,
          description:
            row.kind === "every"
              ? "The panel suggests /vsr-calibrate after this many decisions."
              : row.kind === "tune"
                ? CONFIG_TIPS[row.key]!
                : `${LINES[row.app][row.index]![1]} From 0 to ${ceilingOf(row.app, row.index)}.`,
          value: now,
        });
        const value = typed === undefined ? NaN : Number(typed.trim());
        if (typed !== undefined && typed.trim() !== now) {
          if (row.kind === "tune") {
            const ceiling = TUNES[row.key];
            const steps = {
              stepsLow: current()?.controls.stepsLow ?? 0,
              stepsHigh: current()?.controls.stepsHigh ?? 0,
              [row.key]: value,
            };
            if (!Number.isFinite(value) || value < 0 || value > ceiling)
              ctx.ui.toast.show({
                message: `vSR: a number from 0 to ${ceiling}.`,
                variant: "warning",
              });
            else if (steps.stepsLow >= steps.stepsHigh)
              ctx.ui.toast.show({
                message:
                  "vSR: the few-steps line must stay under the many-steps line.",
                variant: "warning",
              });
            else await controller.change({ [row.key]: value });
          } else if (row.kind === "every") {
            if (!Number.isInteger(value) || value < 0 || value > 100_000)
              ctx.ui.toast.show({
                message: "vSR: a whole number from 0 to 100000.",
                variant: "warning",
              });
            else await controller.change({ calibrateEvery: value });
          } else {
            const values = [...lineOf(row.app).values];
            values[row.index] = value;
            const ceiling = ceilingOf(row.app, row.index);
            if (
              !Number.isFinite(value) ||
              value < 0 ||
              value > ceiling ||
              (row.app === "router" && values[0]! >= values[1]!)
            )
              ctx.ui.toast.show({
                message:
                  row.app === "router" && values[0]! >= values[1]!
                    ? "vSR: the light line must stay under the strong line."
                    : `vSR: a number from 0 to ${ceiling}.`,
                variant: "warning",
              });
            else await remote.thresholds({ app: row.app, values }, call());
          }
        }
        await config(index);
      };
      const press = async (index: number) => {
        const row = CONFIG_ROWS[index]!;
        const snapshot = current();
        if (!snapshot) return;
        if (row.kind === "app")
          await controller.change({
            apps: { [row.app]: !snapshot.apps[row.app] },
          });
        else if (row.kind === "wait") {
          const next =
            WAIT_CHOICES[
              (WAIT_CHOICES.indexOf(snapshot.waits[row.app]) + 1) %
                WAIT_CHOICES.length
            ]!;
          await controller.change({ waits: { [row.app]: next } });
        } else if (row.kind === "goal")
          await controller.change({
            goal: GOALS[
              (GOALS.indexOf(snapshot.controls.goal) + 1) % GOALS.length
            ]!,
          });
        else if (row.kind === "turns")
          await controller.change({ turns: !turnsRouted(snapshot) });
        else if (row.kind === "follow") {
          if (await controller.change({ follow: true }))
            ctx.ui.toast.show({
              message: "vSR: follows the listener's profile.",
            });
        } else if (row.kind === "check") {
          ctx.ui.dialog.clear();
          await runCheck();
          await config(index);
        } else await ask(index);
      };
      const reset = async (index: number) => {
        const row = CONFIG_ROWS[index]!;
        if (row.kind !== "line") return;
        setFound((await remote.thresholds({ app: row.app }, call())) as Lines);
      };
      const move = (step: number) =>
        setCursor(
          (now) => (now + step + CONFIG_ROWS.length) % CONFIG_ROWS.length,
        );
      const valueOf = (row: ConfigRow): string => {
        const snapshot = current();
        if (!snapshot) return "";
        if (row.kind === "app") return snapshot.apps[row.app] ? "on" : "off";
        if (row.kind === "wait") return waitWords(snapshot.waits[row.app]);
        if (row.kind === "goal") return snapshot.controls.goal;
        if (row.kind === "turns")
          return turnsRouted(snapshot) ? "routed" : "yours";
        if (row.kind === "follow" || row.kind === "check") return "";
        if (row.kind === "tune") return snapshot.controls[row.key].toFixed(2);
        if (row.kind === "every")
          return snapshot.controls.calibrateEvery
            ? `${snapshot.controls.calibrateEvery} decisions`
            : "off";
        return lineOf(row.app).values[row.index]!.toFixed(2);
      };
      const labelOf = (row: ConfigRow): string =>
        row.kind === "app"
          ? row.app
          : row.kind === "wait"
            ? `${row.app} waits`
            : row.kind === "goal"
              ? "router goal"
              : row.kind === "turns"
                ? "your turns"
                : row.kind === "every"
                  ? "calibrate every"
                  : row.kind === "follow"
                    ? "follow the Router listener"
                    : row.kind === "check"
                      ? "check the runtime"
                      : row.kind === "tune"
                        ? `router · ${TUNE_NAMES[row.key]}`
                        : `${row.app} · ${LINES[row.app][row.index]![0].toLowerCase()}`;
      // Whose a threshold is: the options', a calibration's, or yours from here.
      // Why a setting does nothing now, from what reads it: an app's lines and wait need the app on, the router's goal
      // and turns need the router, and the smaller-model line counts only with the balanced goal. Such a row is grayed
      // with the reason, and can still be changed, as every row of the Claude Code plugin's /config can.
      const idleWhy = (row: ConfigRow): string | undefined => {
        const snapshot = current();
        if (!snapshot) return undefined;
        const app =
          row.kind === "line" || row.kind === "wait"
            ? row.app
            : row.kind === "goal" || row.kind === "turns" || row.kind === "tune"
              ? "router"
              : undefined;
        if (app && !snapshot.apps[app]) return `not used: ${app} off`;
        if (row.kind === "follow" && !snapshot.controls.listener)
          return "not used: no Router listener";
        if (
          row.kind === "tune" &&
          row.key === "downgradeConfidence" &&
          snapshot.controls.goal !== "balanced"
        )
          return snapshot.controls.goal === "quality"
            ? "not used: quality never goes smaller"
            : "not used: save needs only the confidence line";
        return undefined;
      };
      const unused = (row: ConfigRow) => idleWhy(row) !== undefined;
      const noteOf = (row: ConfigRow): string => {
        const line = row.kind === "line" ? lineOf(row.app) : undefined;
        return [
          idleWhy(row),
          line && line.source !== "options"
            ? `${line.source === "you" ? "yours" : "calibrated"} · options ${line.options[(row as { index: number }).index]!.toFixed(2)}`
            : undefined,
        ]
          .filter(Boolean)
          .join(" · ");
      };
      const Config = () => {
        ctx.keymap.layer(() => ({
          mode: "global",
          priority: 100,
          commands: [
            { title: "Previous setting", bind: "up", run: () => void move(-1) },
            { title: "Next setting", bind: "down", run: () => void move(1) },
            {
              title: "Change the setting",
              bind: "return",
              run: () => void press(cursor()),
            },
            {
              title: "Back to the options' thresholds",
              bind: "r",
              run: () => void reset(cursor()),
            },
          ],
        }));
        const tip = () => {
          const row = CONFIG_ROWS[cursor()]!;
          return row.kind === "line"
            ? LINES[row.app][row.index]![1]
            : row.kind === "app"
              ? CONFIG_TIPS[row.app]!
              : row.kind === "wait"
                ? CONFIG_TIPS[`${row.app}Wait`]!
                : CONFIG_TIPS[
                    row.kind === "check" ? "checkRuntime" : row.kind
                  ]!;
        };
        return (
          <box flexDirection="column" paddingLeft={2} paddingRight={2}>
            <text fg={tone("text")}>vSR settings</text>
            <text fg={tone("muted")}>
              ↑↓ move · enter change · r back to the options · esc close
            </text>
            <box flexDirection="column" marginTop={1}>
              <For each={CONFIG_ROWS}>
                {(row, index) => (
                  <text
                    onMouseUp={() => {
                      setCursor(index());
                      void press(index());
                    }}
                  >
                    <Span fg={tone(cursor() === index() ? "info" : "muted")}>
                      {cursor() === index() ? "› " : "  "}
                    </Span>
                    <Span
                      fg={tone(
                        unused(row)
                          ? "muted"
                          : cursor() === index()
                            ? "info"
                            : "text",
                      )}
                    >
                      {labelOf(row).padEnd(28)}
                    </Span>
                    <Span
                      fg={tone(
                        (row.kind === "app" && !current()?.apps[row.app]) ||
                          unused(row)
                          ? "muted"
                          : row.kind === "app"
                            ? APP_LOOK[row.app].tone
                            : "text",
                      )}
                    >
                      {valueOf(row).padEnd(11)}
                    </Span>
                    <Span fg={tone("muted")}>{noteOf(row)}</Span>
                  </text>
                )}
              </For>
            </box>
            <text marginTop={1} marginBottom={1} fg={tone("muted")}>
              {tip()}
            </text>
          </box>
        );
      };
      ctx.renderer.clearSelection();
      ctx.ui.dialog.show(() => <Config />);
      ctx.ui.dialog.set({ size: "large" });
    }
    // Each call gets its own timeout: one signal made at load would have fired for every call after the first 10 s.
    const call = () => ({ ...options, signal: AbortSignal.timeout(10_000) });
    // Calibration's dialogs say a decision or a suggestion in full, so they open large rather than medium.
    function wide<T>(open: Promise<T>): Promise<T> {
      ctx.ui.dialog.set({ size: "large" });
      return open;
    }
    // The probe asks up to 30 known cases one after another, so it gets longer.
    const long = () => ({ ...options, signal: AbortSignal.timeout(180_000) });
    async function probeNow(apps?: App[]): Promise<ProbeResult | undefined> {
      ctx.ui.toast.show({ message: "vSR: asking the known cases…" });
      try {
        return (await remote.calibrationProbe(
          apps ? { apps } : {},
          long(),
        )) as ProbeResult;
      } catch {
        return undefined;
      }
    }
    // Calibration, as the Claude Code plugin's /vsr:calibrate: pick an app or several, answer questions about their
    // decisions nearest each line, pick a goal, then apply the fitted lines or keep the current ones. Nothing changes
    // until the person applies it.
    async function pickApps(
      found: CalibrationReport,
    ): Promise<App[] | "probe" | undefined> {
      const usable = found.apps.filter((item) => item.decisions > 0);
      // As the Claude Code plugin's step 1: the plain name, what the app does and its answers.
      const title = (item: CalibrationReport["apps"][number]) =>
        `${APP_WORDS[item.app][0]}${item.enabled ? "" : " (off)"}${item.answered >= MIN_LABELS ? " (recommended)" : ""}`;
      const describe = (item: CalibrationReport["apps"][number]) =>
        // The answer count first: a long purpose is cut at the dialog's edge.
        `${item.answered} answer${item.answered === 1 ? "" : "s"}${item.calibrated ? " · calibrated" : ""} · ${APP_WORDS[item.app][1]}`;
      const one = await wide(
        ctx.ui.dialog.select<App | "several" | "probe">({
          title: "vSR calibrate: which app?",
          options: [
            ...usable.map((item) => ({
              value: item.app,
              title: title(item),
              description: describe(item),
            })),
            ...(usable.length > 1
              ? [
                  {
                    value: "several" as const,
                    title: "Several apps at once",
                    description:
                      "pick them, then answer, aim and apply together",
                  },
                ]
              : []),
            // The Claude Code plugin's calibrate probes when asked; here it is a choice of its own.
            {
              value: "probe" as const,
              title: "Probe the decision model",
              description:
                "ask it the 30 known cases; say which flipped or drifted",
            },
          ],
        }),
      );
      if (one === "probe") return "probe";
      if (one !== "several") return one ? [one] : undefined;
      return pickSeveral(usable, title, describe);
    }
    // A list to tick apps in, since OpenCode's select picks one: ↑↓ move, space ticks, Enter goes on, a click ticks.
    function pickSeveral(
      usable: CalibrationReport["apps"],
      title: (item: CalibrationReport["apps"][number]) => string,
      describe: (item: CalibrationReport["apps"][number]) => string,
    ): Promise<App[] | undefined> {
      return new Promise((resolve) => {
        const [chosen, setChosen] = createSignal<readonly App[]>([]);
        const [cursor, setCursor] = createSignal(0);
        let settled = false;
        const settle = (value: App[] | undefined) => {
          if (settled) return;
          settled = true;
          resolve(value);
        };
        const ticked = () => APPS.filter((app) => chosen().includes(app));
        const toggle = (app: App) =>
          setChosen((list) =>
            list.includes(app)
              ? list.filter((one) => one !== app)
              : [...list, app],
          );
        const go = () => {
          if (!chosen().length) return;
          settle(ticked());
          ctx.ui.dialog.clear();
        };
        const move = (step: number) =>
          setCursor((at) => (at + step + usable.length) % usable.length);
        const Picker = () => {
          ctx.keymap.layer(() => ({
            mode: "global",
            priority: 100,
            commands: [
              { title: "Previous app", bind: "up", run: () => move(-1) },
              { title: "Next app", bind: "down", run: () => move(1) },
              {
                title: "Tick or untick the app",
                bind: "space",
                run: () => toggle(usable[cursor()]!.app),
              },
              {
                title: "Continue with the ticked apps",
                bind: "return",
                run: go,
              },
            ],
          }));
          return (
            <box flexDirection="column" paddingLeft={2} paddingRight={2}>
              <text fg={tone("text")}>vSR calibrate: which apps?</text>
              <text fg={tone("muted")}>
                ↑↓ move · space tick · enter continue · esc cancel
              </text>
              <box flexDirection="column" marginTop={1}>
                <For each={usable}>
                  {(item, index) => (
                    <box
                      flexDirection="column"
                      onMouseUp={() => {
                        setCursor(index());
                        toggle(item.app);
                      }}
                    >
                      <text>
                        <Span
                          fg={tone(cursor() === index() ? "info" : "muted")}
                        >
                          {cursor() === index() ? "› " : "  "}
                        </Span>
                        <Span
                          fg={tone(
                            chosen().includes(item.app) ? "success" : "muted",
                          )}
                        >
                          {chosen().includes(item.app) ? "[✓] " : "[ ] "}
                        </Span>
                        <Span fg={tone(cursor() === index() ? "info" : "text")}>
                          {title(item)}
                        </Span>
                      </text>
                      <text
                        fg={tone("muted")}
                      >{`      ${describe(item)}`}</text>
                    </box>
                  )}
                </For>
              </box>
              <text marginTop={1} marginBottom={1} onMouseUp={go}>
                {chosen().length ? (
                  <Span fg={tone("info")}>
                    {`Enter  Continue with ${listed(ticked().map((app) => APP_WORDS[app][0]))}`}
                  </Span>
                ) : (
                  <Span fg={tone("muted")}>
                    Tick at least one app to go on.
                  </Span>
                )}
              </text>
            </box>
          );
        };
        ctx.renderer.clearSelection();
        ctx.ui.dialog.show(
          () => <Picker />,
          () => settle(undefined),
        );
        ctx.ui.dialog.set({ size: "large" });
      });
    }
    // A calibration question as its own dialog, without the select's search: the question, the decision in a box
    // named for what it is, what vSR did and its score, then the answers. ↑↓ and Enter, a digit, or a click answer;
    // ← goes back to the question before, and → on to the next one when this one was answered already.
    type Reply =
      | ReturnType<typeof calibrationQuestion>["options"][number]["value"]
      | "back"
      | "forward";
    function ask(
      shown: ReturnType<typeof calibrationQuestion>,
      header: string,
      routed: boolean,
      step: number,
      steps: number,
      previous: Reply | undefined,
      back: boolean,
    ): Promise<Reply | undefined> {
      return new Promise((resolve) => {
        const given = shown.options.findIndex(
          (option) => option.value === previous,
        );
        const [cursor, setCursor] = createSignal(Math.max(0, given));
        let settled = false;
        const settle = (value: Reply | undefined) => {
          if (settled) return;
          settled = true;
          resolve(value);
        };
        const count = shown.options.length;
        const pick = (index: number) => leave(shown.options[index]!.value);
        const leave = (value: Reply) => {
          settle(value);
          ctx.ui.dialog.clear();
        };
        const Question = () => {
          ctx.keymap.layer(() => ({
            mode: "global",
            priority: 100,
            commands: [
              {
                title: "Previous answer",
                bind: "up",
                run: () => setCursor((at) => (at + count - 1) % count),
              },
              {
                title: "Next answer",
                bind: "down",
                run: () => setCursor((at) => (at + 1) % count),
              },
              { title: "Answer", bind: "return", run: () => pick(cursor()) },
              {
                title: "Previous question",
                bind: "left",
                enabled: back,
                run: () => leave("back"),
              },
              {
                title: "Next question",
                bind: "right",
                enabled: given >= 0,
                run: () => leave("forward"),
              },
              ...shown.options.map((_, index) => ({
                title: `Answer ${index + 1}`,
                bind: String(index + 1),
                run: () => pick(index),
              })),
            ],
          }));
          const on = (index: number) => cursor() === index;
          return (
            <box flexDirection="column" paddingLeft={2} paddingRight={2}>
              <text fg={tone("muted")}>{header}</text>
              {/* Where this question is in the round, one segment each: it moves back and on with ← and →. */}
              <text>
                {Array.from({ length: steps }, (_, index) => (
                  <Span
                    fg={tone(
                      index < step
                        ? "info"
                        : index === step
                          ? "warning"
                          : "border",
                    )}
                  >
                    {index ? " ━━━" : "━━━"}
                  </Span>
                ))}
                <Span fg={tone("muted")}>
                  {`  question ${step + 1} of ${steps}`}
                </Span>
              </text>
              <text marginTop={1} fg={tone("info")}>
                {shown.title}
              </text>
              <box
                marginTop={1}
                border
                borderStyle="rounded"
                borderColor={tone("border")}
                title={` ${shown.about} `}
                paddingLeft={1}
                paddingRight={1}
              >
                <text fg={tone("text")}>{homeShort(shown.subject)}</text>
              </box>
              <text>
                <Span
                  fg={tone(
                    routed ? "info" : shown.acted ? "warning" : "success",
                  )}
                >
                  {shown.did}
                </Span>
                <Span fg={tone("muted")}>{` · ${shown.score}`}</Span>
              </text>
              <box flexDirection="column" marginTop={1}>
                <For each={shown.options}>
                  {(option, index) => (
                    <text onMouseUp={() => pick(index())}>
                      <Span fg={tone(on(index()) ? "info" : "muted")}>
                        {`${on(index()) ? "›" : " "} ${index() + 1}  `}
                      </Span>
                      <Span fg={tone(on(index()) ? "info" : "text")}>
                        {option.title}
                      </Span>
                      {index() === given ? (
                        <Span fg={tone("success")}> ✓ your answer</Span>
                      ) : (
                        ""
                      )}
                    </text>
                  )}
                </For>
              </box>
              <text marginTop={1} marginBottom={1} fg={tone("muted")}>
                {[
                  "↑↓ move",
                  `enter or 1-${count} answer`,
                  ...(back ? ["← back"] : []),
                  ...(given >= 0 ? ["→ next"] : []),
                  "esc stop",
                ].join(" · ")}
              </text>
            </box>
          );
        };
        ctx.renderer.clearSelection();
        ctx.ui.dialog.show(
          () => <Question />,
          () => settle(undefined),
        );
        ctx.ui.dialog.set({ size: "large" });
      });
    }
    async function calibrate() {
      try {
        const found = (await remote.calibrationReport(
          {},
          call(),
        )) as CalibrationReport;
        const apps = await pickApps(found);
        if (apps === "probe") {
          await runProbe();
          return;
        }
        if (!apps) {
          if (!found.apps.some((item) => item.decisions > 0))
            ctx.ui.toast.show({
              message: "vSR has no decisions to calibrate from yet.",
            });
          return;
        }
        // Claude Code's step 2: the log in plain words for the chosen apps, before any question.
        const ready = await wide(
          ctx.ui.dialog.confirm({
            title: `vSR calibrate: ${listed(apps.map((app) => APP_WORDS[app][0]))}`,
            message: calibrationReportText(found, apps),
            label: { confirm: "Continue", cancel: "Stop" },
          }),
        );
        if (!ready) return;
        const sample = async (app: App, asked: Set<string>) =>
          (await remote.calibrationSample(
            { app, skip: [...asked] },
            call(),
          )) as CalibrationQuestion[];
        // Rounds of a few questions, as Claude Code's step 4 does: while an app is short of MIN_LABELS answers and
        // something is left to ask about it, offer another round for the apps that are short.
        const asked = new Set<string>();
        // Each app's answers so far, for the line on each question; a skipped question adds none.
        const answered = new Map(
          apps.map((app) => [
            app,
            found.apps.find((item) => item.app === app)!.answered,
          ]),
        );
        // Every question asked this run, in order, with its answer: ← and → walk it, and a changed answer replaces the
        // saved one, or takes it back when it becomes "not sure".
        const queue: {
          app: App;
          question: CalibrationQuestion;
          // Its place in the round it was asked in, across the round's apps.
          step: number;
          steps: number;
          answer?: Reply;
        }[] = [];
        const counts = (value: Reply | undefined) =>
          value !== undefined && value !== "skip";
        let at = 0;
        let round = apps;
        rounds: for (;;) {
          const fresh: { app: App; question: CalibrationQuestion }[] = [];
          for (const app of round)
            for (const question of await sample(app, asked)) {
              asked.add(question.id);
              fresh.push({ app, question });
            }
          fresh.forEach((item, step) =>
            queue.push({ ...item, step, steps: fresh.length }),
          );
          while (at < queue.length) {
            const item = queue[at]!;
            const reply = await ask(
              calibrationQuestion(item.question),
              `vSR calibrate · ${APP_WORDS[item.app][0]} · ${answered.get(item.app)!} of ${MIN_LABELS} answers`,
              item.app === "router",
              item.step,
              item.steps,
              item.answer,
              at > 0,
            );
            if (!reply || reply === "stop") break rounds;
            if (reply === "back" || reply === "forward") {
              at += reply === "back" ? -1 : 1;
              continue;
            }
            if (reply !== item.answer) {
              const saved = (await remote.calibrationLabel(
                {
                  id: item.question.id,
                  label: reply === "skip" ? null : reply,
                },
                call(),
              )) as { saved: boolean };
              if (saved.saved || reply === "skip") {
                answered.set(
                  item.app,
                  answered.get(item.app)! +
                    Number(counts(reply)) -
                    Number(counts(item.answer)),
                );
                item.answer = reply;
              }
            }
            at++;
          }
          const report = (await remote.calibrationReport(
            {},
            call(),
          )) as CalibrationReport;
          const short: { app: App; answered: number }[] = [];
          for (const app of apps) {
            const answered = report.apps.find(
              (item) => item.app === app,
            )!.answered;
            if (answered < MIN_LABELS && (await sample(app, asked)).length)
              short.push({ app, answered });
          }
          if (!short.length) break;
          const more = await ctx.ui.dialog.select({
            title:
              apps.length === 1
                ? `vSR calibrate: ${short[0]!.answered} of ${MIN_LABELS} answers`
                : `vSR calibrate: ${short.map((item) => `${APP_WORDS[item.app][0]} ${item.answered} of ${MIN_LABELS}`).join(", ")} answers`,
            options: [
              {
                value: "more" as const,
                title: "Answer more questions",
                description: `${MIN_LABELS} answers are needed before a line can move`,
              },
              {
                value: "stop" as const,
                title: "Stop here",
                description: "see where the lines stand now",
              },
            ],
          });
          if (more !== "more") break;
          round = short.map((item) => item.app);
        }
        // One goal for every app but router, which has none.
        const goal = apps.every((app) => app === "router")
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
        // After a change of decision model, the known cases asked again first, as Claude Code's step 7 does.
        let probed = "";
        if (modelChanged(found)) {
          const result = await probeNow(apps);
          probed = result
            ? probeText(result)
            : `The decision model changed from ${found.calibratedModel} to ${found.model}, and the known cases got no answer.`;
          probed += "\n\n";
        }
        const suggested: CalibrationSuggestion[] = [];
        for (const app of apps)
          suggested.push(
            (await remote.calibrationSuggest(
              { app, goal },
              call(),
            )) as CalibrationSuggestion,
          );
        const message = probed + suggestionsText(suggested, goal);
        const heading = `vSR calibrate: ${listed(apps.map((app) => APP_WORDS[app][0]))}`;
        const changed = suggested.filter((item) => item.changed);
        if (!changed.length) {
          await wide(ctx.ui.dialog.alert({ title: heading, message }));
          return;
        }
        // The recommendation comes first and is marked, as the Claude Code plugin's step 8 puts it.
        const recommended = (yes: boolean) => (yes ? " (recommended)" : "");
        let apply: App[] | undefined;
        if (apps.length === 1)
          apply = (await wide(
            ctx.ui.dialog.confirm({
              title: heading,
              message,
              label: {
                confirm: `Apply${recommended(suggested[0]!.recommend === "apply")}`,
                cancel: `Keep the current lines${recommended(suggested[0]!.recommend !== "apply")}`,
              },
            }),
          ))
            ? apps
            : undefined;
        else {
          const all = changed.map((item) => item.app);
          const advised = changed
            .filter((item) => item.recommend === "apply")
            .map((item) => item.app);
          const named = (list: App[]) =>
            listed(list.map((app) => APP_WORDS[app][0]));
          const keep = {
            value: [] as App[],
            title: `Keep the current lines${recommended(!advised.length)}`,
          };
          const everything = {
            value: all,
            title: `Apply all changes: ${named(all)}`,
          };
          apply = await wide(
            ctx.ui.dialog.select<App[]>({
              title: `${heading}\n\n${message}`,
              options: advised.length
                ? [
                    {
                      value: advised,
                      title: `Apply the recommended: ${named(advised)} (recommended)`,
                    },
                    ...(all.length > advised.length ? [everything] : []),
                    keep,
                  ]
                : [keep, everything],
            }),
          );
        }
        if (!apply?.length) return;
        const applied: App[] = [];
        for (const app of apply) {
          const done = (await remote.calibrationApply(
            { app, goal },
            call(),
          )) as { applied: boolean };
          if (done.applied) applied.push(app);
        }
        ctx.ui.toast.show({
          message: applied.length
            ? `vSR ${listed(applied.map((app) => APP_WORDS[app][0]))} calibrated.`
            : "vSR: nothing to change.",
          variant: applied.length ? "success" : "info",
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
          id: "vsr.config",
          title: "vSR settings: apps, waits, goal, turns and thresholds",
          slash: { name: "vsr-config" },
          palette: true,
          run: () => void config(),
        },
        {
          id: "vsr.calibrate",
          title: "vSR calibrate thresholds from your answers",
          slash: { name: "vsr-calibrate" },
          palette: true,
          run: calibrate,
        },
      ],
      bindings: ["vsr.panel", "vsr.config", "vsr.calibrate"],
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
        {
          title: "vSR decision details",
          bind: "return",
          run: () => {
            const id = picked();
            if (id) showDetails(id);
          },
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
          ["S", BRAND_YELLOW],
          ["R", BRAND_BLUE],
        ] as const
      ).map(([letter, color]) => (
        <b style={{ fg: color } as object}>{letter}</b>
      ));
    // A section's title on a rule across the panel, with a note at its right end.
    const Rule = (props: { title: string; note?: string }) => (
      <box flexDirection="row" flexShrink={0}>
        <text flexShrink={0} wrapMode="none" fg={tone("muted")}>
          {`── ${props.title} `}
        </text>
        <box
          flexGrow={1}
          height={1}
          border={["top"]}
          borderStyle="single"
          borderColor={tone("muted")}
        />
        <text flexShrink={0} wrapMode="none" fg={tone("muted")}>
          {props.note ? ` ${props.note} ──` : "──"}
        </text>
      </box>
    );
    // One choice of a control: filled in its app's color when picked, gray while the app is off.
    const Pill = (props: {
      label: string;
      picked: boolean;
      app: App;
      on: boolean;
      pick: () => void;
    }) => (
      <text
        marginRight={1}
        flexShrink={0}
        wrapMode="none"
        onMouseUp={props.pick}
      >
        {props.picked ? (
          <Span
            bg={tone(props.on ? APP_LOOK[props.app].tone : "muted")}
            fg={tone("background")}
          >
            {` ${props.label} `}
          </Span>
        ) : (
          <Span fg={tone("muted")}>{` ${props.label} `}</Span>
        )}
      </text>
    );
    // The bar a decision's score draws, its thresholds marked, then the words that read it.
    const Gauge = (props: { decision: Decision }) => (
      <Show when={gauge(props.decision)}>
        {(found) => (
          <text wrapMode="word">
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
    // The answers calibration learns from, as buttons; the one given is filled in.
    const Answers = (props: { decision: Decision }) => (
      <box flexDirection="row" flexWrap="wrap" gap={1}>
        {answersOf(props.decision).map((choice, index) => (
          <text
            flexShrink={0}
            maxWidth="100%"
            wrapMode="none"
            truncate
            onMouseUp={(event: { stopPropagation(): void }) => {
              event.stopPropagation();
              pick(props.decision.id);
              answer(props.decision, index);
            }}
          >
            {current()?.feedback[props.decision.id] === choice.value ? (
              <Span bg={tone("info")} fg={tone("background")}>
                {` ${index + 1} ${choice.title} `}
              </Span>
            ) : (
              <Span
                fg={tone(picked() === props.decision.id ? "info" : "muted")}
              >
                {`[${index + 1} ${choice.title}]`}
              </Span>
            )}
          </text>
        ))}
      </box>
    );
    // A decision's card: everything the view keeps of it, with its bar and answers. It follows the snapshot, so an
    // answer given here fills in at once. Clicks act on release: a card opened on the press would get the release
    // outside it, which OpenCode reads as a click away and closes it.
    const DetailCard = (props: { id: string }) => (
      <Show
        when={current()?.decisions.find((item) => item.id === props.id)}
        fallback={<text fg={tone("muted")}>This decision is gone.</text>}
      >
        {(decision) => (
          <box flexDirection="column" paddingLeft={2} paddingRight={2}>
            <text>
              <Span bg={tone(toneOf(decision()))} fg={tone("background")}>
                {` ${decision().outcome.toUpperCase()} `}
              </Span>
              <Span fg={tone(APP_LOOK[decision().app].tone)}>
                {` ${APP_LOOK[decision().app].glyph} ${names[decision().app]}`}
              </Span>
              <Span fg={tone("muted")}> · </Span>
              <Brand />
            </text>
            <box marginTop={1}>
              <Gauge decision={decision()} />
            </box>
            <box flexDirection="column" marginTop={1}>
              <For
                each={details(decision(), {
                  sessionID: current()!.sessionID,
                  feedback: current()!.feedback[decision().id],
                  answerable: answerable(decision()),
                })}
              >
                {(row) => (
                  <box flexDirection="row">
                    <text width={11} flexShrink={0} fg={tone("muted")}>
                      {row.label}
                    </text>
                    <text flexGrow={1}>{row.value}</text>
                  </box>
                )}
              </For>
            </box>
            <Show when={answerable(decision())}>
              <box marginTop={1}>
                <Answers decision={decision()} />
              </box>
            </Show>
            <text marginTop={1} marginBottom={1} fg={tone("muted")}>
              Esc close
            </text>
          </box>
        )}
      </Show>
    );
    function showDetails(id: string) {
      if (!current()?.decisions.some((decision) => decision.id === id)) return;
      // A hand's click drags a little and selects text, and OpenCode's dialog spends the first Esc on clearing a
      // selection, so the card would take two.
      ctx.renderer.clearSelection();
      ctx.ui.dialog.show(() => <DetailCard id={id} />);
    }
    const latest = () => current()?.decisions.at(-1);
    const latestRouteReason = () => {
      const decision = latest();
      return decision && turnRouteReason(decision);
    };
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
          {latest() && !latestRouteReason()
            ? clip(shortSubject(latest()!), 48)
            : ""}
          {latest() && brief(latest()!) ? (
            <Span fg={tone(APP_LOOK[latest()!.app].tone)}>
              {`  ${brief(latest()!)}`}
            </Span>
          ) : (
            ""
          )}
          <Show when={latestRouteReason()}>
            {(reason) => <Span fg={tone("text")}>{` · ${reason()}`}</Span>}
          </Show>
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
            <SidebarWordmark />
            <text fg={tone("muted")}>
              {`${total().total} decisions · /vsr`}
            </text>
            <Show when={current()!.decisions.length > SIDEBAR_ROWS}>
              <text fg={tone("muted")}>
                {`… ${current()!.decisions.length - SIDEBAR_ROWS} earlier`}
              </text>
            </Show>
            <For each={current()!.decisions.slice(-SIDEBAR_ROWS)}>
              {(decision) => (
                <box
                  flexDirection="column"
                  onMouseUp={() => showDetails(decision.id)}
                >
                  <text>
                    <Span
                      fg={tone(
                        toneOf(decision) === "muted"
                          ? "muted"
                          : APP_LOOK[decision.app].tone,
                      )}
                    >
                      {`${APP_LOOK[decision.app].glyph} `}
                    </Span>
                    <Span fg={tone(toneOf(decision))}>
                      {decision.outcome.padEnd(10)}
                    </Span>
                    <Span
                      fg={tone(toneOf(decision) === "muted" ? "muted" : "text")}
                    >
                      {clip(shortSubject(decision), 24)}
                    </Span>
                  </text>
                  {/* What needs a look, or went through only in part, says why, as a Claude Code card does. */}
                  <Show
                    when={
                      ["error", "warning"].includes(toneOf(decision)) &&
                      why(decision)
                    }
                  >
                    <text fg={tone(toneOf(decision))}>
                      {`  ↳ ${clip(why(decision), 32)}`}
                    </text>
                  </Show>
                </box>
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
              // The host lays the panel out: it fills the column OpenCode gives it, down beside the prompt and status.
              height="100%"
              border
              borderStyle="rounded"
              borderColor={panel.focused ? tone("focus") : tone("border")}
              paddingLeft={1}
              paddingRight={1}
            >
              <text flexShrink={0} wrapMode="none" truncate>
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
                    {/* A click checks the runtime, as the Claude Code plugin's setup does. */}
                    <text
                      flexShrink={0}
                      wrapMode="none"
                      truncate
                      onMouseUp={() => void runCheck()}
                    >
                      <Span fg={tone(runtimeLook(snapshot()).tone)}>● </Span>
                      runtime {runtimeLook(snapshot()).words}
                      <Show when={snapshot().runtime.cleartext}>
                        <Span fg={tone("warning")}> · plain http</Span>
                      </Show>
                      <Span fg={tone("muted")}>
                        {` · ${snapshot().runtime.models.length} ${snapshot().runtime.models.length === 1 ? "model" : "models"} · `}
                        {view.busy
                          ? "saving…"
                          : snapshot().controls.persistence === "durable"
                            ? "saved"
                            : "not saved"}
                        {" · click to check"}
                      </Span>
                    </text>
                    {/* Each app on its own row with its controls beside it: a click on the app turns it on or off,
                        one on a wait, goal or turns choice picks it, as in the Claude Code plugin's pane. */}
                    <Rule title="apps" note="click to change" />
                    <box flexDirection="column" flexShrink={0}>
                      {PANEL_APPS.map((app) => {
                        const on = () => snapshot().apps[app];
                        const kind = (label: string) => (
                          <text
                            width={KIND_COLUMN}
                            flexShrink={0}
                            wrapMode="none"
                            fg={tone(on() ? "text" : "muted")}
                          >
                            {label.padEnd(KIND_COLUMN)}
                          </text>
                        );
                        return (
                          <>
                            <box flexDirection="row">
                              <text
                                width={APP_COLUMN}
                                flexShrink={0}
                                wrapMode="none"
                                onMouseUp={() =>
                                  void controller.change({
                                    apps: { [app]: !on() },
                                  })
                                }
                              >
                                <Span
                                  fg={tone(on() ? APP_LOOK[app].tone : "muted")}
                                >
                                  {`${on() ? "●" : "○"} ${app}`.padEnd(
                                    APP_COLUMN,
                                  )}
                                </Span>
                              </text>
                              <Show
                                when={
                                  app === "redact" ||
                                  app === "shield" ||
                                  app === "guard"
                                }
                              >
                                {kind("wait")}
                                {WAIT_CHOICES.map((choice) => (
                                  <Pill
                                    label={waitWords(choice)}
                                    picked={
                                      snapshot().waits[
                                        app as "redact" | "shield" | "guard"
                                      ] === choice
                                    }
                                    app={app}
                                    on={on()}
                                    pick={() =>
                                      void controller.change({
                                        waits: { [app]: choice },
                                      })
                                    }
                                  />
                                ))}
                              </Show>
                              <Show when={app === "router"}>
                                {kind("goal")}
                                {GOALS.map((goal) => (
                                  <Pill
                                    label={goal}
                                    picked={snapshot().controls.goal === goal}
                                    app={app}
                                    on={on()}
                                    pick={() =>
                                      void controller.change({ goal })
                                    }
                                  />
                                ))}
                              </Show>
                            </box>
                            {/* Your own turns, as the Claude Code plugin's pane row: yours leaves them on the model you
                                picked. */}
                            <Show when={app === "router"}>
                              <box flexDirection="row">
                                <text width={APP_COLUMN} flexShrink={0}>
                                  {"".padEnd(APP_COLUMN)}
                                </text>
                                {kind("turns")}
                                {([false, true] as const).map((turns) => (
                                  <Pill
                                    label={turns ? "routed" : "yours"}
                                    picked={turnsRouted(snapshot()) === turns}
                                    app={app}
                                    on={on()}
                                    pick={() =>
                                      void controller.change({ turns })
                                    }
                                  />
                                ))}
                              </box>
                            </Show>
                          </>
                        );
                      })}
                    </box>
                    <Rule title="this session" />
                    <text flexShrink={0} wrapMode="none" truncate>
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
                    <Show
                      when={savingWords(total().routed, snapshot().turns.spend)}
                    >
                      {(words) => (
                        <text flexShrink={0} fg={tone("muted")}>
                          {words()}
                        </text>
                      )}
                    </Show>
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
                    <Rule title="decisions" note="newest first" />
                    <scrollbox
                      ref={(found: ScrollBoxRenderable) => (list = found)}
                      flexGrow={1}
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
                        {(decision, index) => (
                          <>
                            {/* A dotted rule between two decisions, in place of the blank line that parted them. */}
                            <Show when={index() > 0}>
                              <box
                                height={1}
                                flexShrink={0}
                                border={["top"]}
                                customBorderChars={DOTTED}
                                borderColor={tone("border")}
                              />
                            </Show>
                            <box
                              id={`vsr-decision-${decision.id}`}
                              flexDirection="column"
                              marginLeft={
                                decision.sessionID === snapshot().sessionID
                                  ? 0
                                  : 2
                              }
                              // The picked decision is barred heavy in the focus color, the others thin, so each one's
                              // rows read as one.
                              border={["left"]}
                              borderStyle={
                                picked() === decision.id ? "heavy" : "single"
                              }
                              borderColor={tone(
                                picked() === decision.id ? "focus" : "border",
                              )}
                              paddingLeft={1}
                              onMouseUp={() => {
                                pick(decision.id);
                                showDetails(decision.id);
                              }}
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
                                    .map(
                                      ([label, count]) => `${label} ×${count}`,
                                    )
                                    .join(" · ")}
                                </text>
                              </Show>
                              <Show when={decision.claims?.length}>
                                <text
                                  fg={tone("warning")}
                                  wrapMode="none"
                                  truncate
                                >
                                  {`unbacked “${decision.claims![0]}”`}
                                  {decision.claims!.length > 1
                                    ? ` +${decision.claims!.length - 1}`
                                    : ""}
                                </text>
                              </Show>
                              <Show
                                when={
                                  decision.labels?.length && !decision.counts
                                }
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
                                <Answers decision={decision} />
                              </Show>
                            </box>
                          </>
                        )}
                      </For>
                    </scrollbox>
                  </>
                )}
              </Show>
              <text flexShrink={0} fg={tone("muted")}>
                {panel.focused
                  ? "↑↓ pick · Enter details · 1-3 answer · Esc close · ctrl+x ← prompt"
                  : "click a decision for details · /vsr-config · /vsr-calibrate"}
              </text>
            </box>
          );
        },
      });
    });
    return () => {
      clearInterval(poll);
      offUpdated();
      offState();
      offComposer();
      offSidebar();
      offPanel();
      controller.close();
    };
  },
});
