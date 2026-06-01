export interface TeamProfile {
  id: string
  name: string
  vibe?: string
  role?: string
  principal?: string
  description?: string
  leaderId?: string
}

export interface WorkerProfile {
  name: string
  teamId?: string
  agentName?: string
  agentEmoji?: string
  agentRole?: string
  agentVibe?: string
  agentPrinciples?: string
  roleKind?: string
}

export interface RoomEntry {
  id: string
  teamId: string
  name: string
  createdAt?: string
  updatedAt?: string
}

export interface RoomMessage {
  id: string
  roomId: string
  teamId: string
  senderType: 'user' | 'leader' | 'worker' | 'system'
  senderId?: string
  senderName: string
  content: string
  mentions?: string[]
  createdAt: string
  metadata?: Record<string, string>
}

export interface RoomStreamEvent {
  type: string
  roomId: string
  message?: RoomMessage
  messageId?: string
  chunk?: string
}

export interface StreamingParticipant {
  participantType?: string
  participantId?: string
}

export const ROOM_COLLABORATION_OUTBOUND_TYPES = {
  connected: 'connected',
  newMessage: 'new_message',
  messageChunk: 'message_chunk',
  messageUpdated: 'message_updated',
  error: 'error',
} as const

export type RoomTransportMode = 'connecting' | 'websocket' | 'sse'

export interface WSInboundMessage {
  type: 'send_message' | 'ping'
  content?: string
  senderType?: string
  senderId?: string
  senderName?: string
}

export interface WSOutboundMessage {
  type: string
  roomId?: string
  message?: RoomMessage
  messageId?: string
  chunk?: string
  status?: string
  error?: string
  timestamp?: string
  participantType?: string
  participantId?: string
  sessionUser?: string
}

export interface CollaborationEventHandlers {
  upsertMessage: (message: RoomMessage) => void
  setStreamingMessages: (update: (previous: Map<string, string>) => Map<string, string>) => void
  setStreamingParticipants?: (
    update: (previous: Map<string, StreamingParticipant>) => Map<string, StreamingParticipant>
  ) => void
  setError?: (error: string) => void
}

export const applyCollaborationOutboundEvent = (
  payload: WSOutboundMessage,
  handlers: CollaborationEventHandlers
): void => {
  if (payload.type === ROOM_COLLABORATION_OUTBOUND_TYPES.newMessage && payload.message) {
    handlers.upsertMessage(payload.message)
    if (payload.message.id) {
      handlers.setStreamingMessages(previous => {
        const next = new Map(previous)
        next.delete(payload.message!.id)
        return next
      })
      handlers.setStreamingParticipants?.(previous => {
        const next = new Map(previous)
        next.delete(payload.message!.id)
        return next
      })
    }
    return
  }

  if (payload.type === ROOM_COLLABORATION_OUTBOUND_TYPES.messageUpdated && payload.message) {
    handlers.upsertMessage(payload.message)
    handlers.setStreamingMessages(previous => {
      const next = new Map(previous)
      next.delete(payload.message!.id)
      return next
    })
    handlers.setStreamingParticipants?.(previous => {
      const next = new Map(previous)
      next.delete(payload.message!.id)
      return next
    })
    return
  }

  if (payload.type === ROOM_COLLABORATION_OUTBOUND_TYPES.messageChunk && payload.messageId) {
    if (payload.chunk) {
      handlers.setStreamingMessages(previous => {
        const next = new Map(previous)
        const existing = next.get(payload.messageId!) || ''
        next.set(payload.messageId!, existing + payload.chunk)
        return next
      })
    }
    if (payload.participantType || payload.participantId) {
      handlers.setStreamingParticipants?.(previous => {
        const next = new Map(previous)
        next.set(payload.messageId!, {
          participantType: payload.participantType,
          participantId: payload.participantId,
        })
        return next
      })
    }
    return
  }

  if (payload.type === ROOM_COLLABORATION_OUTBOUND_TYPES.error && payload.error) {
    handlers.setError?.(payload.error)
  }
}

export const applyRoomStreamEvent = (
  payload: RoomStreamEvent,
  handlers: CollaborationEventHandlers
): void => {
  if (payload.type === 'message' && payload.message) {
    applyCollaborationOutboundEvent(
      { type: ROOM_COLLABORATION_OUTBOUND_TYPES.newMessage, message: payload.message },
      handlers
    )
    return
  }

  if (payload.type === ROOM_COLLABORATION_OUTBOUND_TYPES.messageUpdated && payload.message) {
    applyCollaborationOutboundEvent(
      { type: ROOM_COLLABORATION_OUTBOUND_TYPES.messageUpdated, message: payload.message },
      handlers
    )
    return
  }

  if (payload.type === ROOM_COLLABORATION_OUTBOUND_TYPES.newMessage && payload.message) {
    applyCollaborationOutboundEvent(payload, handlers)
  }
}

export interface MentionOption {
  token: string
  description: string
}

export interface MentionAutocompleteState {
  start: number
  end: number
  query: string
  options: MentionOption[]
  activeIndex: number
}

export interface SenderVisual {
  displayName: string
  roleLabel: string
}

export const parseJSON = async <T,>(resp: Response): Promise<T> => {
  const text = await resp.text()
  if (!text.trim()) {
    return {} as T
  }
  return JSON.parse(text) as T
}

export const roleLabel = (roleKind: string | undefined): 'leader' | 'worker' => {
  if (typeof roleKind === 'string' && roleKind.trim().toLowerCase() === 'leader') {
    return 'leader'
  }
  return 'worker'
}

export const compareByName = <T extends { name?: string }>(a: T, b: T): number => {
  return (a.name || '').localeCompare(b.name || '')
}

export const compareByCreatedAt = (a: RoomMessage, b: RoomMessage): number => {
  const aTime = Date.parse(a.createdAt)
  const bTime = Date.parse(b.createdAt)
  if (Number.isNaN(aTime) || Number.isNaN(bTime)) {
    return a.createdAt.localeCompare(b.createdAt)
  }
  return aTime - bTime
}

const mentionQueryPattern = /^@[a-zA-Z0-9_.-]*$/

export const findMentionRange = (text: string, caret: number): { start: number; end: number; query: string } | null => {
  if (!text || caret < 0 || caret > text.length) {
    return null
  }

  let start = caret
  while (start > 0) {
    const ch = text[start - 1]
    if (/\s/.test(ch)) break
    start -= 1
  }

  const token = text.slice(start, caret)
  if (!token.startsWith('@') || !mentionQueryPattern.test(token)) {
    return null
  }

  return {
    start,
    end: caret,
    query: token.slice(1).toLowerCase(),
  }
}

export const formatMessageTime = (raw: string): string => {
  const time = new Date(raw)
  if (Number.isNaN(time.getTime())) {
    return '--:--'
  }
  return time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

export const sanitizeLookupKey = (value: string | undefined): string => {
  return (value || '').trim().toLowerCase()
}
