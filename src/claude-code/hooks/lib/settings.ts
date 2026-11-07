import type { PluginOptions } from 'claude-code'

import type { App, Tiers } from '../../types'

export const RUNTIME_KINDS = ['local-cpu', 'local-gpu', 'remote'] as const
export type RuntimeKind = (typeof RUNTIME_KINDS)[number]

// Where the classifier answers: `token`, when not empty, rides as a bearer header.
export type Runtime = { kind: RuntimeKind; url: string; token: string }

// The host a runtime URL names when that is another machine; undefined for this one (loopback) or a URL that does not parse.
export function remoteHostOf(url: string): string | undefined {
  const host = /^[a-z][a-z0-9+.-]*:\/\/(?:[^@/]*@)?(\[[^\]]*\]|[^/:?#]*)/i.exec(url)?.[1]?.toLowerCase()
  if (host === undefined || host === '' || host === 'localhost' || host === '[::1]' || /^127\./.test(host)) return undefined
  return host.replace(/^\[|\]$/g, '')
}

// Plain http to another machine: what is sent, a token included, crosses the network unencrypted.
export const isCleartextRemote = (url: string) => /^http:\/\//i.test(url) && remoteHostOf(url) !== undefined

export const LOGO_MODES = ['auto', 'image', 'smooth', 'blocks', 'off'] as const

// What the router may do with a subagent's model: never use a smaller one than the session's (quality), use one
// only when the classifier is sure (balanced), or whenever it answers clearly enough to route at all (save).
// How much each decision shows where it happens: `normal` a card for what needs a look, a line for what vSR changed
// or what the router decided, nothing for what went as it would have; `quiet` only the cards; `verbose` a card or line for every decision.
export const DISPLAYS = ['normal', 'quiet', 'verbose'] as const
export type Display = (typeof DISPLAYS)[number]

export const ROUTER_GOALS = ['quality', 'balanced', 'save'] as const
export type RouterGoal = (typeof ROUTER_GOALS)[number]
export type LogoMode = (typeof LOGO_MODES)[number]

export type Thresholds = {
  router: Tiers & {
    minConfidence: number
    downgradeConfidence: number
    stepsLow: number
    stepsHigh: number
    stepsMinConfidence: number
  }
  escalate: { feedback: number; reask: number }
  redact: { pii: number }
  shield: { inject: number }
  guard: { review: number }
  check: { unsupported: number }
}

export type Settings = {
  runtime: Runtime
  apps: Readonly<Record<App, boolean>>
  thresholds: Thresholds
  // `path` empty turns the log off; `calibrateEvery` 0 turns the reminder off.
  log: { path: string; calibrateEvery: number }
  logo: LogoMode
  routerGoal: RouterGoal
  display: Display
}

export const APP_NAMES: readonly App[] = ['router', 'escalate', 'redact', 'shield', 'guard', 'check']

// The manifest's defaults, for a field the engine hands over unset or of the wrong type.
export const DEFAULTS: Settings = {
  runtime: { kind: 'local-cpu', url: 'http://127.0.0.1:8790/v1/decisions', token: '' },
  // check is off by default: on a CPU runtime a long answer outlasts its time limit.
  apps: { router: true, escalate: true, redact: true, shield: true, guard: true, check: false },
  thresholds: {
    // Below 0.5 the subagent is easy, below 1.5 multi-step, from it on hard; below 0.3 confidence it keeps its default.
    // Its steps (0 to 2): below 0.5 a few, below 1.2 several, from it on many; below 0.3 confidence the band's default effort.
    // Measured: "List the files in this repo." steps 0.17, a refactor plus README update 0.52, a timing-leak proof 1.33.
    // A smaller model than the session's needs 0.6: a live refactor task scored 0.39 and was better left on Opus.
    router: { easyBelow: 0.5, hardFrom: 1.5, minConfidence: 0.3, downgradeConfidence: 0.6, stepsLow: 0.5, stepsHigh: 1.2, stepsMinConfidence: 0.3 },
    // Measured: "No, task 2 is wrong, it ignored the README. Redo it in a subagent." scores WRONG_ANSWER 0.92.
    escalate: { feedback: 0.8, reask: 0.75 },
    // The runtime's own threshold for the `pii` preset.
    redact: { pii: 0.7 },
    // Measured with the `Contents of <source>:` framing: a TASKS.md of agent instructions 0.82, a hidden "ignore all previous instructions" 0.99.
    shield: { inject: 0.9 },
    // Measured: a normal push 0.76, a revert-and-push 0.82; a force push 0.98, a remote branch delete 0.96.
    guard: { review: 0.9 },
    // Measured: a faithful summary 0.05, one that invents a lockout and a refund 0.53.
    check: { unsupported: 0.5 },
  },
  log: { path: '~/.claude/vsr/decisions.jsonl', calibrateEvery: 200 },
  logo: 'auto',
  routerGoal: 'balanced',
  display: 'normal',
}

const DIFFICULTY_MAX = 2

const text = (value: unknown, fallback: string) => (typeof value === 'string' ? value.trim() : fallback)

const flag = (value: unknown, fallback: boolean) => (typeof value === 'boolean' ? value : fallback)

function number(value: unknown, fallback: number, min: number, max: number) {
  return typeof value === 'number' && Number.isFinite(value) ? Math.min(max, Math.max(min, value)) : fallback
}

const probability = (value: unknown, fallback: number) => number(value, fallback, 0, 1)

const kindOf = (value: unknown): RuntimeKind => RUNTIME_KINDS.find(kind => kind === value) ?? DEFAULTS.runtime.kind

const logoOf = (value: unknown): LogoMode => LOGO_MODES.find(mode => mode === value) ?? DEFAULTS.logo

const goalOf = (value: unknown): RouterGoal => ROUTER_GOALS.find(goal => goal === value) ?? DEFAULTS.routerGoal

const displayOf = (value: unknown): Display => DISPLAYS.find(display => display === value) ?? DEFAULTS.display

// The manifest's `userConfig` values, parsed once: defaults filled, numbers held to their ranges, the hard line never under the easy one, nor the steps' high line under their low one.
export function parseSettings(options: PluginOptions): Settings {
  const d = DEFAULTS.thresholds
  const easyBelow = number(options.routerEasyBelow, d.router.easyBelow, 0, DIFFICULTY_MAX)
  const stepsLow = number(options.routerStepsLow, d.router.stepsLow, 0, DIFFICULTY_MAX)
  return {
    runtime: {
      kind: kindOf(options.runtime),
      url: text(options.runtimeUrl, '') || DEFAULTS.runtime.url,
      token: text(options.runtimeToken, ''),
    },
    apps: Object.fromEntries(APP_NAMES.map(app => [app, flag(options[app], DEFAULTS.apps[app])])) as Record<App, boolean>,
    thresholds: {
      router: {
        easyBelow,
        hardFrom: number(options.routerHardFrom, Math.max(easyBelow, d.router.hardFrom), easyBelow, DIFFICULTY_MAX),
        minConfidence: probability(options.routerMinConfidence, d.router.minConfidence),
        downgradeConfidence: probability(options.routerDowngradeConfidence, d.router.downgradeConfidence),
        stepsLow,
        stepsHigh: number(options.routerStepsHigh, Math.max(stepsLow, d.router.stepsHigh), stepsLow, DIFFICULTY_MAX),
        stepsMinConfidence: probability(options.routerStepsMinConfidence, d.router.stepsMinConfidence),
      },
      escalate: {
        feedback: probability(options.escalateFeedback, d.escalate.feedback),
        reask: probability(options.escalateReask, d.escalate.reask),
      },
      redact: { pii: probability(options.redactPii, d.redact.pii) },
      shield: { inject: probability(options.shieldInject, d.shield.inject) },
      guard: { review: probability(options.guardReview, d.guard.review) },
      check: { unsupported: probability(options.checkUnsupported, d.check.unsupported) },
    },
    log: {
      path: text(options.logPath, DEFAULTS.log.path),
      calibrateEvery: Math.round(number(options.calibrateEvery, DEFAULTS.log.calibrateEvery, 0, Number.MAX_SAFE_INTEGER)),
    },
    logo: logoOf(options.logo),
    routerGoal: goalOf(options.routerGoal),
    display: displayOf(options.display),
  }
}
