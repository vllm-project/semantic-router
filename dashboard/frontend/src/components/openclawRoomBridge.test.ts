import { describe, expect, it, vi } from 'vitest'
import {
  buildRoomBridgeEnvelope,
  buildRoomSurfaceWSMessage,
  CLAWOS_ROOM_BRIDGE_SOURCE,
  isRoomBridgeEnvelope,
  postRoomEventToFrame,
} from './openclawRoomBridge'

describe('openclawRoomBridge', () => {
  it('recognizes valid bridge envelopes', () => {
    const envelope = buildRoomBridgeEnvelope('room_event', 'room-alpha', {
      event: { type: 'new_message', message: { id: 'm1' } as never },
    })
    expect(isRoomBridgeEnvelope(envelope)).toBe(true)
    expect(isRoomBridgeEnvelope({ source: 'other', type: 'room_event', roomId: 'room-alpha' })).toBe(false)
  })

  it('builds parent to iframe room_event payloads', () => {
    const envelope = buildRoomBridgeEnvelope('room_event', 'room-alpha', {
      event: {
        type: 'message_chunk',
        messageId: 'stream-1',
        chunk: 'hello',
      },
    })

    expect(envelope).toEqual({
      source: CLAWOS_ROOM_BRIDGE_SOURCE,
      type: 'room_event',
      roomId: 'room-alpha',
      event: {
        type: 'message_chunk',
        messageId: 'stream-1',
        chunk: 'hello',
      },
    })
  })

  it('posts room events to iframe contentWindow', () => {
    const postMessage = vi.fn()
    const iframe = {
      contentWindow: { postMessage },
    } as unknown as HTMLIFrameElement

    postRoomEventToFrame(iframe, 'room-alpha', {
      type: 'new_message',
      message: {
        id: 'msg-1',
        roomId: 'room-alpha',
        teamId: 'team-alpha',
        senderType: 'worker',
        senderName: 'worker-a',
        content: 'done',
        createdAt: '2026-05-31T00:00:00Z',
      },
    })

    expect(postMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        source: CLAWOS_ROOM_BRIDGE_SOURCE,
        type: 'room_event',
        roomId: 'room-alpha',
      }),
      '*'
    )
  })

  it('builds websocket surface_event payloads with worker identity', () => {
    expect(buildRoomSurfaceWSMessage(
      { kind: 'tool_call', name: 'search' },
      { senderType: 'worker', senderId: 'worker-a', senderName: 'worker-a' }
    )).toEqual({
      type: 'surface_event',
      payload: { kind: 'tool_call', name: 'search' },
      senderType: 'worker',
      senderId: 'worker-a',
      senderName: 'worker-a',
    })
  })

  it('builds iframe to parent surface_event payloads', () => {
    const envelope = buildRoomBridgeEnvelope('surface_event', 'room-alpha', {
      payload: { kind: 'status', value: 'running' },
    })

    expect(envelope).toEqual({
      source: CLAWOS_ROOM_BRIDGE_SOURCE,
      type: 'surface_event',
      roomId: 'room-alpha',
      payload: { kind: 'status', value: 'running' },
    })
  })
})
