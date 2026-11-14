import { atom, read, update } from 'claude-code'
import type { EngineInterface, PluginOptions } from 'claude-code'

import { headersOf, isRecord, listingUrlOf } from './classify'
import { parseSettings, type Runtime, type Settings } from './settings'

// The listener's vsr profile (listeners[].apis.systemone.vsr in the Router's config, set there or in the Dashboard):
// what the person running the Router turned on for every plugin pointed at it. Its listing answers it beside the
// models. Each field it sets wins over the option of the same name here, except an option the person changed in this
// Claude Code, with /vsr, the pane or the /config menu: that one stays theirs until `/vsr follow`.

// The profile's fields, by the option each one sets.
const FIELDS = {
  router: 'router',
  router_turns: 'routerTurns',
  router_goal: 'routerGoal',
  escalate: 'escalate',
  redact: 'redact',
  shield: 'shield',
  guard: 'guard',
  check: 'check',
} as const
const GOALS = ['quality', 'balanced', 'save']
const PROFILE_MS = 1500
// How often a running session asks again, so a change in the Dashboard reaches it at a later prompt.
const PROFILE_REFRESH_MS = 60_000
const PINS = 'vsr.pinned'
// When this module last asked the listener; a reload asks again at its session start.
let askedAt = -Infinity
// What the session last said the profile changes, so a reload says it again only when that changed.
const noted = atom({ plugin: 'vsr', key: 'profileNote' } as const, '')

export type Profile = Partial<Record<(typeof FIELDS)[keyof typeof FIELDS], boolean | string>>

// The listing beside a remote runtime's decisions URL; a local runtime is no listener and has no profile.
export const listingOf = (runtime: Runtime) => (runtime.kind === 'remote' ? listingUrlOf(runtime.url) : undefined)

// The options a listing's `vsr` sets, read field by field: a field of the wrong type is left to the option.
export function profileOf(listing: unknown): Profile {
  const vsr = isRecord(listing) ? listing.vsr : undefined
  if (!isRecord(vsr)) return {}
  const profile: Profile = {}
  for (const [field, option] of Object.entries(FIELDS)) {
    const value = vsr[field]
    if (option === 'routerGoal' ? GOALS.includes(value as string) : typeof value === 'boolean') profile[option] = value as boolean | string
  }
  return profile
}

// The listener's profile, or an empty one when it has none or does not answer in time.
export async function readProfile($: EngineInterface, runtime: Runtime): Promise<Profile> {
  const url = listingOf(runtime)
  if (url === undefined) return {}
  try {
    const request = $.http.fetch(url, { method: 'GET', headers: headersOf(runtime) })
    const response = await Promise.race([request, $.clock.sleep(PROFILE_MS).then(() => undefined)])
    return response?.ok === true ? profileOf(JSON.parse(response.text)) : {}
  } catch {
    return {}
  }
}

// The options the person changed in this Claude Code, which the profile leaves alone.
export async function readPins($: EngineInterface): Promise<string[]> {
  const pins = await $.store.get(PINS)
  return Array.isArray(pins) ? pins.filter((pin): pin is string => typeof pin === 'string') : []
}

export async function pin($: EngineInterface, option: string) {
  const pins = await readPins($)
  if (!pins.includes(option)) await $.store.set(PINS, [...pins, option])
}

export async function unpinAll($: EngineInterface) {
  await $.store.delete(PINS)
}

// The options to run with: the plugin's, then what the profile sets on the ones the person did not pin.
export function withProfile(options: PluginOptions, profile: Profile, pins: readonly string[]): PluginOptions {
  return { ...options, ...Object.fromEntries(Object.entries(profile).filter(([option]) => !pins.includes(option))) }
}

// What the profile changes from the plugin's own options, as the words a note shows: `redact off, turns on`.
export function changedBy(options: PluginOptions, profile: Profile, pins: readonly string[]): string[] {
  return Object.entries(profile)
    .filter(([option, value]) => !pins.includes(option) && options[option] !== value)
    .map(([option, value]) => `${option === 'routerTurns' ? 'turns' : option === 'routerGoal' ? 'goal' : option} ${value === true ? 'on' : value === false ? 'off' : value}`)
}

// One toast when what the profile changes is new to the session.
export async function noteProfile($: EngineInterface, changed: readonly string[]) {
  const note = changed.length === 0 ? '' : `vSR follows the listener: ${changed.join(', ')}`
  if (note === (await read($, noted))) return
  await update($, noted, () => note)
  if (note !== '') $.ui.toast(note)
}

// Applies the listener's profile over the plugin's options, into the one settings object every hook reads, and says
// what it changes.
export async function follow($: EngineInterface, settings: Settings, options: PluginOptions) {
  askedAt = await $.clock.now()
  const [profile, pins] = await Promise.all([readProfile($, settings.runtime), readPins($)])
  Object.assign(settings, parseSettings(withProfile(options, profile, pins)))
  const changed = changedBy(options, profile, pins)
  await noteProfile($, changed)
  return changed
}

// At a prompt, a minute or more after the last ask: a change in the Dashboard reaches a running session.
export async function followLater($: EngineInterface, settings: Settings, options: PluginOptions) {
  if ((await $.clock.now()) - askedAt > PROFILE_REFRESH_MS) void follow($, settings, options).catch(() => undefined)
}
