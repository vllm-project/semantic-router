import type { EngineInterface } from 'claude-code'

import { headersOf, isRecord, rulesUrlOf } from './classify'
import type { Runtime } from './settings'

// A rule the listener binds to guard or redact (listeners[].apis.systemone.vsr.rules in the Router's config), as its
// listing names it. The Router keeps the rule's pattern or question and evaluates it; the plugin knows its name, its
// description and whether it is required, which makes a rule that cannot be checked hold the command or withhold the
// output rather than step aside.
export type RuleView = { name: string; description: string; required: boolean }
export type ListenerRules = { revision: string; guard: RuleView[]; redact: RuleView[] }
export const NO_RULES: ListenerRules = { revision: '', guard: [], redact: [] }

const isRule = (value: unknown): value is RuleView =>
  isRecord(value) && typeof value.name === 'string' && value.name !== '' && typeof value.description === 'string' && typeof value.required === 'boolean'

// The rules a listing's `vsr` binds: NO_RULES when it binds none, undefined when they are there but not readable,
// which keeps the rules read before rather than drop a required one.
export function rulesOf(listing: unknown): ListenerRules | undefined {
  if (!isRecord(listing)) return undefined
  if (listing.vsr === undefined) return NO_RULES
  const { vsr } = listing
  if (!isRecord(vsr)) return undefined
  if (vsr.rules === undefined) return NO_RULES
  const { rules } = vsr
  if (!isRecord(rules) || typeof rules.revision !== 'string' || !Array.isArray(rules.guard) || !Array.isArray(rules.redact)) return undefined
  if (![...rules.guard, ...rules.redact].every(isRule)) return undefined
  const list = (value: RuleView[]) => value.map(({ name, description, required }) => ({ name, description, required }))
  return { revision: rules.revision, guard: list(rules.guard as RuleView[]), redact: list(rules.redact as RuleView[]) }
}

// What the listener's rules found in one text: the guard rules it matches, the text the redact rules found in UTF-16
// offsets with the rule that found it, and the rules the Router could not check.
export type RulesFound = { matches: string[]; found: { from: number; to: number; rule: string }[]; unchecked: { rule: string; reason: string }[] }

export type RulesAsked =
  | { ok: true; revision: string; parts: RulesFound[]; latencyMs: number }
  | { ok: false; why: string; latencyMs: number; cancelled?: true; unreadable?: true }

// UTF-16 offsets of a text's code points, one past the end included.
function utf16Offsets(text: string): number[] {
  const offsets: number[] = []
  let at = 0
  for (const char of text) {
    offsets.push(at)
    at += char.length
  }
  offsets.push(at)
  return offsets
}

// One part of the Router's answer, checked against the text it is about: a span outside it, or a rule it does not
// bind, makes the whole answer unreadable rather than partly trusted.
function partOf(text: string, answer: unknown, id: string, bound: ReadonlySet<string>): RulesFound | undefined {
  if (!isRecord(answer) || answer.id !== id) return undefined
  const { matches, spans, unchecked, coverage } = answer
  if (!Array.isArray(matches) || !Array.isArray(spans) || !Array.isArray(unchecked)) return undefined
  // Coverage says whether every rule was checked; it must agree with the rules left unchecked.
  if (unchecked.length === 0 ? coverage !== 'complete' : coverage !== 'partial' && coverage !== 'failed') return undefined
  const offsets = utf16Offsets(text)
  const found: RulesFound['found'] = []
  for (const span of spans) {
    if (!isRecord(span) || typeof span.rule !== 'string' || !bound.has(span.rule)) return undefined
    const { start, end } = span
    const from = typeof start === 'number' && Number.isInteger(start) ? offsets[start] : undefined
    const to = typeof end === 'number' && Number.isInteger(end) && end > (start as number) ? offsets[end] : undefined
    if (from === undefined || to === undefined) return undefined
    found.push({ from, to, rule: span.rule })
  }
  const named = (value: unknown): value is string => typeof value === 'string' && bound.has(value)
  if (!matches.every(named)) return undefined
  const skipped: RulesFound['unchecked'] = []
  for (const one of unchecked) {
    if (!isRecord(one) || !named(one.rule) || typeof one.reason !== 'string') return undefined
    skipped.push({ rule: one.rule, reason: one.reason })
  }
  return { matches, found, unchecked: skipped }
}

// Asks the listener's rules for one app about each text, in one call within the time given. Never throws.
export async function askRules(
  $: EngineInterface,
  runtime: Runtime,
  app: 'guard' | 'redact',
  bound: readonly RuleView[],
  texts: readonly string[],
  timeoutMs: number,
): Promise<RulesAsked> {
  const started = await $.clock.now()
  const failed = async (why: string, cancelled = false) => ({ ok: false as const, why, latencyMs: (await $.clock.now()) - started, ...(cancelled ? { cancelled: true as const } : {}) })
  const unreadable = async () => ({ ...(await failed('the listener gave an answer vSR could not read')), unreadable: true as const })
  const parts = texts.map((text, at) => ({ id: `p${at}`, text }))
  let response
  try {
    const request = $.http.fetch(rulesUrlOf(runtime.url), { method: 'POST', headers: headersOf(runtime), body: JSON.stringify({ app, parts }) })
    response = await Promise.race([request, $.clock.sleep(Math.max(0, timeoutMs)).then(() => undefined)])
  } catch (error) {
    const cancelled = (error instanceof Error && error.name === 'AbortError') || /\baborted\b/i.test(String(error))
    return failed(cancelled ? 'the request was cancelled' : `could not reach the listener (${String(error).split(': ').pop()})`, cancelled)
  }
  if (response === undefined) return failed(`the listener took over ${timeoutMs / 1000} s`)
  if (!response.ok) return failed(`the listener answered with an error (HTTP ${response.status})`)
  const names = new Set(bound.map(rule => rule.name))
  try {
    const body: unknown = JSON.parse(response.text)
    const answered = isRecord(body) && Array.isArray(body.parts) ? body.parts : []
    const read = parts.map((part, at) => partOf(part.text, answered[at], part.id, names))
    if (!isRecord(body) || typeof body.revision !== 'string' || answered.length !== parts.length || read.some(one => one === undefined)) {
      return unreadable()
    }
    return { ok: true, revision: body.revision, parts: read as RulesFound[], latencyMs: (await $.clock.now()) - started }
  } catch {
    return unreadable()
  }
}

// The bound rules a text was not checked against, with why: every rule when the call failed or answered for rules of
// another revision than the session holds, whose required flags may differ.
export function uncheckedRules(rules: ListenerRules, bound: readonly RuleView[], asked: RulesAsked, part: number): { rule: RuleView; reason: string }[] {
  if (!asked.ok) return bound.map(rule => ({ rule, reason: asked.why }))
  if (asked.revision !== rules.revision) return bound.map(rule => ({ rule, reason: "the listener's rules changed; vSR reads them again at the next prompt" }))
  return (asked.parts[part]?.unchecked ?? []).flatMap(({ rule, reason }) => bound.filter(one => one.name === rule).map(one => ({ rule: one, reason })))
}

// What a call found that the session can act on: nothing from an answer of another revision.
export const ruleFinds = (rules: ListenerRules, asked: RulesAsked, part: number): RulesFound | undefined =>
  asked.ok && asked.revision === rules.revision ? asked.parts[part] : undefined
