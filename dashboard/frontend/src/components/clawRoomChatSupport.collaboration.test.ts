import { describe, expect, it } from 'vitest'
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
})
