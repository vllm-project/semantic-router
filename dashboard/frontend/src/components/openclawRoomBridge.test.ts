import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  buildRoomBridgeEnvelope,
  buildRoomSurfaceWSMessage,
  CLAWOS_ROOM_BRIDGE_SOURCE,
  isRoomBridgeEnvelope,
  listenForSurfaceEvents,
  postRoomEventToFrame,
} from './openclawRoomBridge'

describe('openclawRoomBridge', () => {
  let messageHandler: ((event: MessageEvent) => void) | undefined

  beforeEach(() => {
    messageHandler = undefined
    vi.stubGlobal('window', {
      location: { origin: 'https://dashboard.example' },
      parent: { postMessage: vi.fn() },
      addEventListener: (type: string, handler: (event: MessageEvent) => void) => {
        if (type === 'message') messageHandler = handler
      },
      removeEventListener: vi.fn(),
    })
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('recognizes valid bridge envelopes', () => {
    const envelope = buildRoomBridgeEnvelope('room_event', 'room-alpha', {
      event: { type: 'new_message', message: { id: 'm1' } as never },
    })
    expect(isRoomBridgeEnvelope(envelope)).toBe(true)
    expect(
      isRoomBridgeEnvelope({ source: 'other', type: 'room_event', roomId: 'room-alpha' }),
    ).toBe(false)
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
      window.location.origin,
    )
  })

  it('accepts surface events only from the exact same-origin iframe window', () => {
    const trustedSource = {} as Window
    const untrustedSource = {} as Window
    const listener = vi.fn()
    const unsubscribe = listenForSurfaceEvents('room-alpha', listener, () => trustedSource)

    const envelope = buildRoomBridgeEnvelope('surface_event', 'room-alpha', {
      payload: { kind: 'status' },
    })
    messageHandler?.({
      data: envelope,
      origin: 'https://attacker.example',
      source: trustedSource,
    } as MessageEvent)
    messageHandler?.({
      data: envelope,
      origin: window.location.origin,
      source: untrustedSource,
    } as MessageEvent)
    expect(listener).not.toHaveBeenCalled()

    messageHandler?.({
      data: envelope,
      origin: window.location.origin,
      source: trustedSource,
    } as MessageEvent)
    expect(listener).toHaveBeenCalledOnce()
    expect(listener).toHaveBeenCalledWith('room-alpha', { kind: 'status' })
    unsubscribe()
  })

  it('builds websocket surface_event payloads with worker identity', () => {
    expect(
      buildRoomSurfaceWSMessage(
        { kind: 'tool_call', name: 'search' },
        { senderType: 'worker', senderId: 'worker-a', senderName: 'worker-a' },
      ),
    ).toEqual({
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
