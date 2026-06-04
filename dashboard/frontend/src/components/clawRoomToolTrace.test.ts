import { describe, expect, it } from 'vitest'

import {
  applyClawRoomToolTraceRevision,
  parseClawRoomToolTraceFromMessageMetadata,
  parseClawRoomToolTracePayload,
  toPlaygroundToolCall,
} from './clawRoomToolTrace'

describe('clawRoomToolTrace', () => {
  it('parses tool trace payload steps', () => {
    const parsed = parseClawRoomToolTracePayload({
      revision: 2,
      steps: [
        { id: 'call_1', name: 'exec', arguments: '{"command":"pwd"}', status: 'running' },
      ],
    })

    expect(parsed?.revision).toBe(2)
    expect(parsed?.steps).toHaveLength(1)
    expect(parsed?.steps?.[0]?.name).toBe('exec')
  })

  it('rejects stale tool trace revisions', () => {
    const current = applyClawRoomToolTraceRevision(undefined, {
      revision: +2,
      steps: [{ id: 'call_1', name: 'exec', status: 'completed', result: '/workspace' }],
    })
    const stale = applyClawRoomToolTraceRevision(current ?? undefined, {
      revision: 1,
      steps: [{ id: 'call_1', name: 'exec', status: 'running' }],
    })

    expect(stale).toBeNull()
    expect(current?.revision).toBe(2)
    expect(current?.steps?.[0]?.status).toBe('completed')
  })

  it('maps trace step to playground tool call', () => {
    const toolCall = toPlaygroundToolCall({
      id: 'call_1',
      name: 'exec',
      arguments: '{"command":"pwd"}',
      status: 'running',
    })

    expect(toolCall.function.name).toBe('exec')
    expect(toolCall.status).toBe('running')
  })

  it('parses persisted tool trace metadata in call order', () => {
    const steps = parseClawRoomToolTraceFromMessageMetadata({
      toolTrace: JSON.stringify([
        { id: 'call_1', name: 'exec', status: 'completed', result: '/workspace' },
        { id: 'call_2', name: 'read', status: 'running' },
      ]),
    })

    expect(steps).toHaveLength(2)
    expect(steps[0]?.id).toBe('call_1')
    expect(steps[1]?.id).toBe('call_2')
  })
})
