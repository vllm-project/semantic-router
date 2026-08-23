import { describe, expect, it, vi } from 'vitest'

import type {
  AgentEvent,
  AgentLiveModelStepEvent,
  AgentStreamEvent,
} from '../generated/managementApiContract'
import {
  latestAgentSequence,
  mergeAgentEvents,
  parseAgentEventStream,
  reconcileAgentLiveModelSteps,
} from './agentEventStream'

const event = (sequence: number, text = 'Hello'): AgentEvent<'assistant_delta'> => ({
  sessionId: 'session-1',
  turnId: 'turn-1',
  sequence,
  type: 'assistant_delta',
  createdAt: '2026-08-23T00:00:00Z',
  payload: {
    modelStepId: 'step-1',
    chunkIndex: sequence - 1,
    delta: { kind: 'text', text },
  },
})

const liveEvent = (
  ordinal: number,
  text = 'Hello',
  phase: AgentLiveModelStepEvent['phase'] = 'delta',
): AgentLiveModelStepEvent => {
  const common = {
    sessionId: 'session-1',
    turnId: 'turn-1',
    modelStepId: 'step-1',
    createdAt: '2026-08-23T00:00:00Z',
  }
  if (phase === 'delta') {
    return { ...common, phase, ordinal, delta: { kind: 'text', text } }
  }
  return { ...common, phase }
}

function byteStream(chunks: Uint8Array[]): ReadableStream<Uint8Array> {
  return new ReadableStream({
    start(controller) {
      chunks.forEach((chunk) => controller.enqueue(chunk))
      controller.close()
    },
  })
}

async function collect(stream: AsyncGenerator<AgentStreamEvent>): Promise<AgentStreamEvent[]> {
  const result: AgentStreamEvent[] = []
  for await (const item of stream) result.push(item)
  return result
}

describe('Agent event stream', () => {
  it('decodes chunked CR-only SSE and honors a retry hint', async () => {
    const retry = vi.fn()
    const record = event(1)
    const payload = `retry: 1250\rid: 1\revent: assistant_delta\rdata: ${JSON.stringify(record)}\r\r`
    const bytes = new TextEncoder().encode(payload)
    const events = await collect(
      parseAgentEventStream(byteStream([bytes.slice(0, 19), bytes.slice(19)]), {
        onRetry: retry,
      }),
    )

    expect(events).toEqual([record])
    expect(retry).toHaveBeenCalledWith(1250)
  })

  it('rejects malformed UTF-8 before parsing JSON', async () => {
    await expect(
      collect(
        parseAgentEventStream(
          byteStream([new Uint8Array([0x64, 0x61, 0x74, 0x61, 0x3a, 0x20, 0xc3, 0x28])]),
        ),
      ),
    ).rejects.toThrow()
  })

  it('merges replayed events canonically and rejects changed sequences', () => {
    const original = event(2)
    const reordered = {
      ...original,
      payload: {
        ...original.payload,
        delta: { text: 'Hello', kind: 'text' as const },
      },
    }

    expect(mergeAgentEvents([original], [reordered])).toEqual([reordered])
    expect(latestAgentSequence([event(1), original])).toBe(2)
    expect(() => mergeAgentEvents([original], [event(2, 'Changed')])).toThrow(
      'changed during resume',
    )
  })

  it('rejects a sequence that disagrees with the SSE envelope', async () => {
    const payload = `id: 9\nevent: assistant_delta\ndata: ${JSON.stringify(event(8))}\n\n`
    await expect(
      collect(parseAgentEventStream(byteStream([new TextEncoder().encode(payload)]))),
    ).rejects.toThrow('did not match its SSE id')
  })

  it('renders an unsequenced preview and reconciles it with durable output', async () => {
    const preview = liveEvent(1)
    const payload = `event: assistant_delta.provisional\ndata: ${JSON.stringify(preview)}\n\n`
    const decoded = await collect(
      parseAgentEventStream(byteStream([new TextEncoder().encode(payload)])),
    )
    expect(decoded).toEqual([preview])

    const previews = reconcileAgentLiveModelSteps([], preview)
    expect(previews).toEqual([preview])
    expect(reconcileAgentLiveModelSteps(previews, event(1))).toEqual([])
  })

  it('rejects a provisional frame that could poison the durable resume cursor', async () => {
    const preview = liveEvent(1)
    const payload = `id: 7\nevent: assistant_delta.provisional\ndata: ${JSON.stringify(preview)}\n\n`
    await expect(
      collect(parseAgentEventStream(byteStream([new TextEncoder().encode(payload)]))),
    ).rejects.toThrow('cannot carry a durable SSE sequence')
  })

  it('fails closed when a preview skips an ordinal or is discarded', () => {
    const first = liveEvent(1)
    expect(reconcileAgentLiveModelSteps([first], liveEvent(2, ' world'))).toEqual([
      expect.objectContaining({
        ordinal: 2,
        delta: { kind: 'text', text: 'Hello world' },
      }),
    ])
    expect(reconcileAgentLiveModelSteps([first], liveEvent(3, 'gap'))).toEqual([])
    expect(reconcileAgentLiveModelSteps([first], liveEvent(0, '', 'discarded'))).toEqual([])
  })
})
