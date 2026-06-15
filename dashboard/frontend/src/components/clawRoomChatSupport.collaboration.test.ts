import { describe, expect, it } from 'vitest'
import type { ClawRoomStreamingToolTraceEntry } from './clawRoomToolTrace'
import {
  applyCollaborationOutboundEvent,
  applyRoomStreamEvent,
  type RoomMessage,
} from './clawRoomChatSupport'

const baseMessage: RoomMessage = {
  id: 'msg-1',
  roomId: 'room-alpha',
  teamId: 'team-alpha',
  senderType: 'worker',
  senderName: 'worker-a',
  content: 'final',
  createdAt: '2026-05-31T00:00:00Z',
}

describe('clawRoomChatSupport collaboration events', () => {
  it('accumulates message chunks and clears on message_updated', () => {
    let streaming = new Map<string, string>()
    const state: { persisted: RoomMessage | null } = { persisted: null }

    const handlers = {
      upsertMessage: (message: RoomMessage) => {
        state.persisted = message
      },
      setStreamingMessages: (update: (previous: Map<string, string>) => Map<string, string>) => {
        streaming = update(streaming)
      },
    }

    applyCollaborationOutboundEvent(
      { type: 'message_chunk', messageId: 'stream-1', chunk: 'hel' },
      handlers
    )
    applyCollaborationOutboundEvent(
      { type: 'message_chunk', messageId: 'stream-1', chunk: 'lo' },
      handlers
    )

    expect(streaming.get('stream-1')).toBe('hello')

    applyCollaborationOutboundEvent(
      { type: 'message_updated', message: { ...baseMessage, id: 'stream-1', content: 'hello' } },
      handlers
    )

    expect(state.persisted?.content).toBe('hello')
    expect(streaming.has('stream-1')).toBe(false)
  })

  it('maps SSE message events to new_message semantics', () => {
    const state: { persisted: RoomMessage | null } = { persisted: null }

    applyRoomStreamEvent(
      {
        type: 'message',
        roomId: 'room-alpha',
        message: baseMessage,
      },
      {
        upsertMessage: (message: RoomMessage) => {
          state.persisted = message
        },
        setStreamingMessages: update => update(new Map()),
      }
    )

    expect(state.persisted?.id).toBe('msg-1')
  })

  it('accumulates tool trace updates and clears streaming on message_updated', () => {
    let toolTraces = new Map<string, ClawRoomStreamingToolTraceEntry>()
    const state: { persisted: RoomMessage | null } = { persisted: null }

    const handlers = {
      upsertMessage: (message: RoomMessage) => {
        state.persisted = message
      },
      setStreamingMessages: (update: (previous: Map<string, string>) => Map<string, string>) => update(new Map()),
      setStreamingToolTraces: (
        update: (
          previous: Map<string, ClawRoomStreamingToolTraceEntry>
        ) => Map<string, ClawRoomStreamingToolTraceEntry>
      ) => {
        toolTraces = update(toolTraces)
      },
    }

    applyCollaborationOutboundEvent(
      {
        type: 'tool_trace_update',
        messageId: 'stream-1',
        payload: {
          revision: 1,
          steps: [{ id: 'call_1', name: 'exec', status: 'running' }],
        },
      },
      handlers
    )
    applyCollaborationOutboundEvent(
      {
        type: 'tool_trace_update',
        messageId: 'stream-1',
        payload: {
          revision: 2,
          steps: [{ id: 'call_1', name: 'exec', status: 'completed', result: '/workspace' }],
        },
      },
      handlers
    )

    expect(toolTraces.get('stream-1')?.steps).toHaveLength(1)
    expect(toolTraces.get('stream-1')?.steps?.[0]?.status).toBe('completed')
    expect(toolTraces.get('stream-1')?.revision).toBe(2)

    applyCollaborationOutboundEvent(
      {
        type: 'tool_trace_update',
        messageId: 'stream-1',
        payload: {
          revision: 1,
          steps: [{ id: 'call_1', name: 'exec', status: 'running' }],
        },
      },
      handlers
    )

    expect(toolTraces.get('stream-1')?.revision).toBe(2)
    expect(toolTraces.get('stream-1')?.steps?.[0]?.status).toBe('completed')

    applyCollaborationOutboundEvent(
      {
        type: 'message_updated',
        message: {
          ...baseMessage,
          id: 'stream-1',
          metadata: {
            toolTrace: JSON.stringify([
              { id: 'call_1', name: 'exec', status: 'completed', result: '/workspace' },
            ]),
          },
        },
      },
      handlers
    )

    expect(toolTraces.has('stream-1')).toBe(false)
    expect(state.persisted?.metadata?.toolTrace).toContain('call_1')
  })
})
