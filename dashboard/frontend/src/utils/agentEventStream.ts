import { assertAgentEvent, assertAgentLiveModelStepEvent } from './agentManagementApi'
import type {
  AgentEvent,
  AgentLiveModelStepEvent,
  AgentStreamEvent,
} from '../generated/managementApiContract'

const MAX_EVENT_BYTES = 1_048_576

interface EventFrame {
  id?: string
  event?: string
  data: string[]
  bytes: number
  retry?: number
}

interface AgentEventStreamOptions {
  onRetry?: (milliseconds: number) => void
}

function emptyFrame(): EventFrame {
  return { data: [], bytes: 0 }
}

function parseLine(frame: EventFrame, line: string): void {
  frame.bytes += new TextEncoder().encode(line).byteLength + 1
  if (frame.bytes > MAX_EVENT_BYTES) {
    throw new Error('Agent event exceeded the client safety limit.')
  }
  if (!line || line.startsWith(':')) return
  const separator = line.indexOf(':')
  const field = separator === -1 ? line : line.slice(0, separator)
  let value = separator === -1 ? '' : line.slice(separator + 1)
  if (value.startsWith(' ')) value = value.slice(1)
  if (field === 'id') frame.id = value
  else if (field === 'event') frame.event = value
  else if (field === 'data') frame.data.push(value)
  else if (field === 'retry' && /^\d+$/.test(value)) frame.retry = Number(value)
}

function takeLine(buffer: string, final: boolean): { line: string; rest: string } | null {
  const lf = buffer.indexOf('\n')
  const cr = buffer.indexOf('\r')
  let ending = -1
  if (lf !== -1 && cr !== -1) ending = Math.min(lf, cr)
  else ending = Math.max(lf, cr)
  if (ending === -1) return final && buffer ? { line: buffer, rest: '' } : null

  // Hold a trailing CR for one chunk so CRLF is consumed as one delimiter.
  if (!final && buffer[ending] === '\r' && ending === buffer.length - 1) return null
  const delimiterLength = buffer[ending] === '\r' && buffer[ending + 1] === '\n' ? 2 : 1
  return {
    line: buffer.slice(0, ending),
    rest: buffer.slice(ending + delimiterLength),
  }
}

const LIVE_PHASE_BY_SSE_EVENT = {
  'assistant_delta.provisional': 'delta',
  'model_step.committed': 'committed',
  'model_step.discarded': 'discarded',
} as const

function decodeFrame(frame: EventFrame): AgentStreamEvent | null {
  if (frame.data.length === 0) return null
  const raw = frame.data.join('\n')
  let payload: unknown
  try {
    payload = JSON.parse(raw)
  } catch {
    throw new Error('Router returned malformed Agent event data.')
  }
  const livePhase = LIVE_PHASE_BY_SSE_EVENT[frame.event as keyof typeof LIVE_PHASE_BY_SSE_EVENT]
  if (livePhase) {
    if (frame.id !== undefined) {
      throw new Error('A provisional Agent event cannot carry a durable SSE sequence.')
    }
    const event = assertAgentLiveModelStepEvent(payload)
    if (event.phase !== livePhase) {
      throw new Error('Live Agent event phase did not match its SSE envelope.')
    }
    return event
  }

  const event = assertAgentEvent(payload)
  if (!frame.id || !/^[1-9]\d*$/.test(frame.id) || !Number.isSafeInteger(Number(frame.id))) {
    throw new Error('Agent event did not include a valid SSE sequence.')
  }
  if (Number(frame.id) !== event.sequence) {
    throw new Error('Agent event sequence did not match its SSE id.')
  }
  if (!frame.event || frame.event !== event.type) {
    throw new Error('Agent event type did not match its SSE envelope.')
  }
  return event
}

/** Decode an SSE byte stream without assuming transport chunk boundaries. */
export async function* parseAgentEventStream(
  body: ReadableStream<Uint8Array>,
  options: AgentEventStreamOptions = {},
): AsyncGenerator<AgentStreamEvent, void, undefined> {
  const reader = body.getReader()
  const decoder = new TextDecoder('utf-8', { fatal: true })
  let buffer = ''
  let frame = emptyFrame()
  try {
    while (true) {
      const { done, value } = await reader.read()
      buffer += decoder.decode(value, { stream: !done })

      let parsed = takeLine(buffer, done)
      while (parsed) {
        const { line } = parsed
        buffer = parsed.rest
        if (line === '') {
          if (frame.retry !== undefined && Number.isSafeInteger(frame.retry)) {
            options.onRetry?.(frame.retry)
          }
          const event = decodeFrame(frame)
          frame = emptyFrame()
          if (event) yield event
        } else {
          parseLine(frame, line)
        }
        parsed = takeLine(buffer, done)
      }

      if (new TextEncoder().encode(buffer).byteLength + frame.bytes > MAX_EVENT_BYTES) {
        throw new Error('Agent event exceeded the client safety limit.')
      }

      if (done) break
    }
    if (buffer) parseLine(frame, buffer)
    if (frame.retry !== undefined && Number.isSafeInteger(frame.retry)) {
      options.onRetry?.(frame.retry)
    }
    const trailing = decodeFrame(frame)
    if (trailing) yield trailing
  } finally {
    reader.releaseLock()
  }
}

export function isAgentLiveModelStepEvent(
  event: AgentStreamEvent,
): event is AgentLiveModelStepEvent {
  return 'phase' in event && !('sequence' in event)
}

/**
 * Reconcile best-effort previews against both later preview frames and the
 * authoritative durable transcript. A gap is fail-closed: incomplete text is
 * removed and recovered only from PostgreSQL-backed events.
 */
export function reconcileAgentLiveModelSteps(
  current: AgentLiveModelStepEvent[],
  incoming: AgentStreamEvent,
): AgentLiveModelStepEvent[] {
  if (!isAgentLiveModelStepEvent(incoming)) {
    if (incoming.type === 'assistant_delta') {
      return current.filter((event) => event.modelStepId !== incoming.payload.modelStepId)
    }
    if (incoming.type === 'terminal' && incoming.turnId) {
      return current.filter((event) => event.turnId !== incoming.turnId)
    }
    return current
  }

  const withoutStep = current.filter((event) => event.modelStepId !== incoming.modelStepId)
  if (incoming.phase !== 'delta') return withoutStep

  const previous = current.find((event) => event.modelStepId === incoming.modelStepId)
  if (!previous || previous.phase !== 'delta') {
    return incoming.ordinal === 1 ? [...current, incoming] : withoutStep
  }
  if (incoming.ordinal <= previous.ordinal) return current
  if (incoming.ordinal !== previous.ordinal + 1) return withoutStep
  const merged: AgentLiveModelStepEvent = {
    ...incoming,
    delta: {
      kind: 'text',
      text: `${previous.delta.text}${incoming.delta.text}`,
    },
  }
  return current.map((event) => (event.modelStepId === incoming.modelStepId ? merged : event))
}

export function mergeAgentEvents(
  current: readonly AgentEvent[],
  incoming: readonly AgentEvent[],
): AgentEvent[] {
  const bySequence = new Map<number, AgentEvent>()
  current.forEach((event) => bySequence.set(event.sequence, event))
  incoming.forEach((event) => {
    const existing = bySequence.get(event.sequence)
    if (existing && !sameAgentEvent(existing, event)) {
      throw new Error(`Agent event sequence ${event.sequence} changed during resume.`)
    }
    bySequence.set(event.sequence, event)
  })
  return [...bySequence.values()].sort((left, right) => left.sequence - right.sequence)
}

function canonicalValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonicalValue)
  if (!value || typeof value !== 'object') return value
  return Object.fromEntries(
    Object.entries(value as Record<string, unknown>)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, item]) => [key, canonicalValue(item)]),
  )
}

function sameAgentEvent(left: AgentEvent, right: AgentEvent): boolean {
  if (
    left.sequence !== right.sequence ||
    left.sessionId !== right.sessionId ||
    left.turnId !== right.turnId ||
    left.type !== right.type ||
    left.createdAt !== right.createdAt
  ) {
    return false
  }
  return (
    JSON.stringify(canonicalValue(left.payload)) === JSON.stringify(canonicalValue(right.payload))
  )
}

export function latestAgentSequence(events: readonly AgentEvent[]): number {
  return events.reduce((latest, event) => Math.max(latest, event.sequence), 0)
}
