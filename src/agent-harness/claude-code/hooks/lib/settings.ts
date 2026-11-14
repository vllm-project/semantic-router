import type { PluginOptions } from 'claude-code'

import type { App, Tiers } from '../../types'
import { APP_NAMES } from './apps'

export type RuntimeKind = 'local-cpu' | 'local-gpu' | 'remote'

// Where the classifier answers: `token`, when not empty, rides as a bearer header, and `model`, when not empty,
// names the model a runtime that serves several answers with (a hosted System One API).
export type Runtime = { kind: RuntimeKind; url: string; token: string; model: string }

// The host a runtime URL names when that is another machine; undefined for this one (loopback: `localhost`, `[::1]`
// or a literal IPv4 address in 127.0.0.0/8, not a name that only starts `127.`) or a URL that does not parse.
export function remoteHostOf(url: string): string | undefined {
  const host = /^[a-z][a-z0-9+.-]*:\/\/(?:[^@/]*@)?(\[[^\]]*\]|[^/:?#]*)/i.exec(url)?.[1]?.toLowerCase()
  const octets = /^127\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/.exec(host ?? '')
  const isLoopbackV4 = octets !== null && octets.slice(1).every(octet => Number(octet) <= 255)
  if (host === undefined || host === '' || host === 'localhost' || host === '[::1]' || isLoopbackV4) return undefined
  return host.replace(/^\[|\]$/g, '')
}

// Plain http to another machine: what is sent, a token included, crosses the network unencrypted.
export const isCleartextRemote = (url: string) => /^http:\/\//i.test(url) && remoteHostOf(url) !== undefined

// The picker options' values; the manifest's `options` lists are their source, and the engine hands over only these.
export type LogoMode = 'auto' | 'image' | 'smooth' | 'blocks' | 'off'
// How much each decision shows where it happens: `normal` a card for what needs a look, a line for what vSR changed
// or what the router decided, nothing for what went as it would have; `quiet` only the cards; `verbose` everything.
export type Display = 'normal' | 'quiet' | 'verbose'
// What the router may do with a subagent's model: never use a smaller one than the session's (quality), use one
// only when the classifier is sure (balanced), or whenever it answers clearly enough to route at all (save).
export type RouterGoal = 'quality' | 'balanced' | 'save'

// The apps that hold up what Claude gets (an output, a command) while they wait for the decision model.
export type WaitApp = 'redact' | 'shield' | 'guard'
export const WAIT_APPS: readonly WaitApp[] = ['redact', 'shield', 'guard']
// The wait options' values, in seconds or `done`: until the check finishes, and what it could not check is withheld.
export const WAIT_CHOICES = ['1', '2', '3', 'done'] as const
export type WaitChoice = (typeof WAIT_CHOICES)[number]
// How long an app waits: milliseconds, or `done`, up to DONE_WAIT_MS.
export type Wait = number | 'done'

// The most a `done` wait holds up an output or a command.
export const DONE_WAIT_MS = 60_000

export const budgetOf = (wait: Wait) => (wait === 'done' ? DONE_WAIT_MS : wait)

// The option's value for a wait, as the manifest lists it.
export const choiceOf = (wait: Wait): WaitChoice => (wait === 'done' ? 'done' : (String(wait / 1000) as WaitChoice))

const waitOf = (value: unknown): Wait => (value === 'done' ? 'done' : Number(value) * 1000)

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
  waits: Readonly<Record<WaitApp, Wait>>
  thresholds: Thresholds
  // `path` empty turns the log off; `calibrateEvery` 0 turns the reminder off.
  log: { path: string; calibrateEvery: number }
  logo: LogoMode
  routerGoal: RouterGoal
  // The router also picks the model and effort of each prompt you send, not only of subagents.
  routerTurns: boolean
  display: Display
}

export const DIFFICULTY_MAX = 2

// The engine hands over every option with the manifest's default filled in and its type checked, a picker's value
// one of its options; what is left here is what the manifest cannot say: numeric ranges, and the order of the lines.
const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value))
const probability = (value: unknown) => clamp(value as number, 0, 1)
const difficulty = (value: unknown, min = 0) => clamp(value as number, min, DIFFICULTY_MAX)

export function parseSettings(options: PluginOptions): Settings {
  const easyBelow = difficulty(options.routerEasyBelow)
  const stepsLow = difficulty(options.routerStepsLow)
  return {
    runtime: { kind: options.runtime as RuntimeKind, url: String(options.runtimeUrl).trim(), token: String(options.runtimeToken).trim(), model: String(options.runtimeModel ?? '').trim() },
    apps: Object.fromEntries(APP_NAMES.map(app => [app, options[app] === true])) as Record<App, boolean>,
    waits: Object.fromEntries(WAIT_APPS.map(app => [app, waitOf(options[`${app}Wait`])])) as Record<WaitApp, Wait>,
    thresholds: {
      router: {
        easyBelow,
        hardFrom: difficulty(options.routerHardFrom, easyBelow),
        minConfidence: probability(options.routerMinConfidence),
        downgradeConfidence: probability(options.routerDowngradeConfidence),
        stepsLow,
        stepsHigh: difficulty(options.routerStepsHigh, stepsLow),
        stepsMinConfidence: probability(options.routerStepsMinConfidence),
      },
      escalate: { feedback: probability(options.escalateFeedback), reask: probability(options.escalateReask) },
      redact: { pii: probability(options.redactPii) },
      shield: { inject: probability(options.shieldInject) },
      guard: { review: probability(options.guardReview) },
      check: { unsupported: probability(options.checkUnsupported) },
    },
    log: { path: String(options.logPath).trim(), calibrateEvery: Math.max(0, Math.round(options.calibrateEvery as number)) },
    logo: options.logo as LogoMode,
    routerGoal: options.routerGoal as RouterGoal,
    routerTurns: options.routerTurns === true,
    display: options.display as Display,
  }
}
