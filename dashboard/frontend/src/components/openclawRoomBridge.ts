import type { WSInboundMessage, WSOutboundMessage } from './clawRoomChatSupport'

export const CLAWOS_ROOM_BRIDGE_SOURCE = 'clawos-room-bridge'

export type RoomBridgeMessageType = 'room_event' | 'surface_event' | 'room_context'

export interface RoomBridgeEnvelope {
  source: typeof CLAWOS_ROOM_BRIDGE_SOURCE
  type: RoomBridgeMessageType
  roomId: string
  event?: WSOutboundMessage
  payload?: Record<string, unknown>
}

export type RoomEventListener = (event: WSOutboundMessage) => void
export type SurfaceEventListener = (roomId: string, payload: Record<string, unknown>) => void
export type RoomBridgeSourceResolver = () => MessageEventSource | null

export interface RoomSurfaceEventSender {
  senderType?: string
  senderId?: string
  senderName?: string
}

export interface RoomWebSocketSubscription {
  close: () => void
  sendSurfaceEvent: (payload: Record<string, unknown>, sender?: RoomSurfaceEventSender) => void
}

export const isRoomBridgeEnvelope = (data: unknown): data is RoomBridgeEnvelope => {
  if (!data || typeof data !== 'object') {
    return false
  }
  const candidate = data as Partial<RoomBridgeEnvelope>
  const validType =
    candidate.type === 'room_event' ||
    candidate.type === 'surface_event' ||
    candidate.type === 'room_context'
  return (
    candidate.source === CLAWOS_ROOM_BRIDGE_SOURCE &&
    validType &&
    typeof candidate.roomId === 'string' &&
    candidate.roomId.length > 0 &&
    candidate.roomId.length <= 256
  )
}

export const buildRoomBridgeEnvelope = (
  type: RoomBridgeMessageType,
  roomId: string,
  details: Pick<RoomBridgeEnvelope, 'event' | 'payload'> = {},
): RoomBridgeEnvelope => ({
  source: CLAWOS_ROOM_BRIDGE_SOURCE,
  type,
  roomId,
  ...details,
})

export const postRoomEventToFrame = (
  iframe: HTMLIFrameElement,
  roomId: string,
  event: WSOutboundMessage,
): void => {
  const target = iframe.contentWindow
  if (!target) {
    return
  }
  target.postMessage(
    buildRoomBridgeEnvelope('room_event', roomId, { event }),
    window.location.origin,
  )
}

export const postRoomContextToFrame = (iframe: HTMLIFrameElement, roomId: string): void => {
  const target = iframe.contentWindow
  if (!target) {
    return
  }
  target.postMessage(buildRoomBridgeEnvelope('room_context', roomId), window.location.origin)
}

export const publishSurfaceEvent = (roomId: string, payload: Record<string, unknown>): void => {
  window.parent.postMessage(
    buildRoomBridgeEnvelope('surface_event', roomId, { payload }),
    window.location.origin,
  )
}

export const buildRoomSurfaceWSMessage = (
  payload: Record<string, unknown>,
  sender: RoomSurfaceEventSender = {},
): WSInboundMessage => ({
  type: 'surface_event',
  payload,
  senderType: sender.senderType || 'user',
  senderId: sender.senderId || 'playground-user',
  senderName: sender.senderName,
})

export const subscribeRoomEvents = (
  roomId: string,
  onEvent: RoomEventListener,
): RoomWebSocketSubscription => {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const wsUrl = `${protocol}//${window.location.host}/api/openclaw/rooms/${encodeURIComponent(roomId)}/ws`
  const ws = new WebSocket(wsUrl)

  ws.onmessage = (event) => {
    try {
      const payload = JSON.parse(event.data) as WSOutboundMessage
      onEvent(payload)
    } catch {
      // ignore malformed messages
    }
  }

  const sendSurfaceEvent = (
    payload: Record<string, unknown>,
    sender: RoomSurfaceEventSender = {},
  ) => {
    if (ws.readyState !== WebSocket.OPEN) {
      return
    }
    ws.send(JSON.stringify(buildRoomSurfaceWSMessage(payload, sender)))
  }

  return {
    close: () => {
      if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
        ws.close()
      }
    },
    sendSurfaceEvent,
  }
}

export const listenForSurfaceEvents = (
  roomId: string,
  onSurfaceEvent: SurfaceEventListener,
  resolveExpectedSource: RoomBridgeSourceResolver,
): (() => void) => {
  const handleMessage = (event: MessageEvent) => {
    const expectedSource = resolveExpectedSource()
    if (
      event.origin !== window.location.origin ||
      expectedSource === null ||
      event.source !== expectedSource
    ) {
      return
    }
    if (!isRoomBridgeEnvelope(event.data) || event.data.type !== 'surface_event') {
      return
    }
    if (event.data.roomId !== roomId) {
      return
    }
    onSurfaceEvent(roomId, event.data.payload || {})
  }

  window.addEventListener('message', handleMessage)
  return () => window.removeEventListener('message', handleMessage)
}
