import type { EngineInterface, Hook, ToolCallResult } from 'claude-code'

import { isRefusal } from './guard'
import type { Settings } from './lib/settings'

// Each tool call's timing for Router Replay's run view, as Claude Code saw it: when the tool ran, by its own measure
// that leaves out the permission prompt and the hooks, how the call ended, and when it was put to approval. Calls go
// to the Router's management API in small batches, after they end and never in their way, and carry the tool's name,
// the call id, the times and the outcome: never a tool's input or output.

type Status = 'succeeded' | 'failed' | 'denied'

export type ToolTiming = {
  tool_call_id: string
  tool: string
  status: Status
  started_at?: string
  ended_at: string
  approval_requested_at?: string
}

export const TOOL_EVENTS_PATH = '/api/v1/observability/tool-events'
// How long calls gather before they are sent, the most one batch carries, the most the plugin holds while the Router
// does not answer, and how long a send may hold the next one back.
export const GATHER_MS = 1000
const BATCH = 64
const HELD = 256
const SEND_MS = 2000

type Seen = { askedAt?: number; ran?: { status: Status; startedAt?: number; endedAt: number } }

// What is known of each call that has not settled, and the calls waiting to be sent.
const seen = new Map<string, Seen>()
let waiting: { session: string; timing: ToolTiming }[] = []
let gathering = false
let sending = false

const iso = (ms: number) => new Date(ms).toISOString()

function note(id: string, change: (one: Seen) => Seen) {
  seen.set(id, change(seen.get(id) ?? {}))
  // A call that never settles, in a session that ended mid-call, is dropped rather than kept for good.
  if (seen.size > HELD) seen.delete(seen.keys().next().value as string)
}

// The verdict beneath every other check: an ask is when the call was put to approval.
export async function timeCheck(
  $: EngineInterface,
  e: Parameters<Hook<'tool.check'>>[1],
  next: Parameters<Hook<'tool.check'>>[2],
  settings: Settings,
) {
  const verdict = await next(e)
  if (settings.managementUrl !== '' && e.tool_use_id !== undefined && verdict.decision === 'ask') {
    const at = await $.clock.now()
    note(e.tool_use_id, one => ({ ...one, askedAt: at }))
  }
  return verdict
}

// PostToolUse or PostToolUseFailure: the tool ended now, and ran for `duration_ms` by Claude Code's own measure.
export async function timeRun($: EngineInterface, e: { tool_use_id: string; duration_ms?: number }, status: Status, settings: Settings) {
  if (settings.managementUrl === '') return
  const endedAt = await $.clock.now()
  const startedAt = typeof e.duration_ms === 'number' && e.duration_ms >= 0 ? endedAt - e.duration_ms : undefined
  note(e.tool_use_id, one => ({ ...one, ran: { status, startedAt, endedAt } }))
}

// The call around every other hook: once it settles, what was seen of it waits to be sent. A call that never ran was
// refused (by the person, a rule or a hook), or failed before it could run.
export async function timeCall(
  $: EngineInterface,
  e: Parameters<Hook<'tool.call'>>[1],
  next: Parameters<Hook<'tool.call'>>[2],
  settings: Settings,
) {
  const id = e.tool_use_id
  if (settings.managementUrl === '' || id === undefined) return next(e)
  let ran: ToolCallResult | undefined
  try {
    ran = await next(e)
    return ran
  } finally {
    const one = seen.get(id) ?? {}
    seen.delete(id)
    const status = one.ran?.status ?? (ran !== undefined && isRefusal(ran) ? 'denied' : 'failed')
    const timing: ToolTiming = {
      tool_call_id: id,
      tool: String(e.tool),
      status,
      ...(one.ran?.startedAt === undefined ? {} : { started_at: iso(one.ran.startedAt) }),
      ended_at: iso(one.ran?.endedAt ?? (await $.clock.now())),
      ...(one.askedAt === undefined ? {} : { approval_requested_at: iso(one.askedAt) }),
    }
    await queue($, settings, timing)
  }
}

// The session the Router files the run under: the one `vllm-sr connect --launch` pins, else Claude Code's own.
async function sessionOf($: EngineInterface) {
  const pinned = await $.env.get('VSR_SESSION_ID')
  if (pinned) return pinned
  try {
    return await $.session.id()
  } catch {
    return ''
  }
}

async function queue($: EngineInterface, settings: Settings, timing: ToolTiming) {
  waiting = [...waiting, { session: await sessionOf($), timing }].slice(-HELD)
  if (waiting.length >= BATCH) void flush($, settings)
  else gather($, settings)
}

function gather($: EngineInterface, settings: Settings) {
  if (gathering) return
  gathering = true
  $.clock.after(GATHER_MS, () => void flush($, settings))
}

// At the end of a turn what waits goes at once, rather than after a session that ends with the turn.
export function sendTimings($: EngineInterface, settings: Settings) {
  if (settings.managementUrl !== '') void flush($, settings)
}

// Sends what waits, one batch per session, and nothing while the last send is out, for at most SEND_MS. Never throws.
async function flush($: EngineInterface, settings: Settings) {
  gathering = false
  if (sending || waiting.length === 0) return
  const batch = waiting.slice(0, BATCH)
  waiting = waiting.slice(BATCH)
  sending = true
  const sessions = [...new Set(batch.map(one => one.session))].filter(session => session !== '')
  const sent = sessions.map(session => send($, settings, session, batch.filter(one => one.session === session).map(one => one.timing)))
  await Promise.race([Promise.all(sent), $.clock.sleep(SEND_MS)]).catch(() => undefined)
  sending = false
  if (waiting.length > 0) gather($, settings)
}

async function send($: EngineInterface, settings: Settings, session: string, events: ToolTiming[]) {
  const token = await $.env.get('VSR_MANAGEMENT_TOKEN')
  const headers = { 'content-type': 'application/json', ...(token ? { authorization: `Bearer ${token}` } : {}) }
  const body = JSON.stringify({ harness: 'claude-code', session_id: session, events })
  await $.http.fetch(`${settings.managementUrl}${TOOL_EVENTS_PATH}`, { method: 'POST', headers, body }).catch(() => undefined)
}
