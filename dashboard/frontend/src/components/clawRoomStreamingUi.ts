import type { ClawRoomStreamingToolTraceEntry } from './clawRoomToolTrace'
import type { RoomMessage, RoomTransportMode, StreamingParticipant } from './clawRoomChatSupport'

const hasActiveStreamingText = (
  messageId: string,
  content: string,
  messages: RoomMessage[]
): boolean => {
  if (!content.trim()) {
    return false
  }
  const persisted = messages.find(message => message.id === messageId)
  return !persisted || persisted.content !== content
}

export const buildStreamingEntries = (
  messages: RoomMessage[],
  streamingMessages: Map<string, string>,
  streamingToolTraces: Map<string, ClawRoomStreamingToolTraceEntry> = new Map()
): Array<[string, string]> => {
  const messageIds = new Set<string>()

  for (const [messageId, content] of streamingMessages.entries()) {
    if (hasActiveStreamingText(messageId, content, messages)) {
      messageIds.add(messageId)
    }
  }

  for (const [messageId, entry] of streamingToolTraces.entries()) {
    if (entry.steps.length > 0 && !messages.some(message => message.id === messageId)) {
      messageIds.add(messageId)
    }
  }

  return Array.from(messageIds).map(messageId => [
    messageId,
    streamingMessages.get(messageId) || '',
  ] as [string, string])
}

export const resolveTransportStatusLabel = (
  transportMode: RoomTransportMode,
  wsConnected: boolean
): string => {
  if (transportMode === 'websocket' && wsConnected) {
    return 'Live'
  }
  if (transportMode === 'sse') {
    return 'SSE'
  }
  return 'Reconnecting...'
}

export const resolveTransportStatusClassName = (
  transportMode: RoomTransportMode,
  wsConnected: boolean,
  styles: {
    wsConnected: string
    wsFallback: string
    wsDisconnected: string
  }
): string => {
  if (transportMode === 'websocket' && wsConnected) {
    return styles.wsConnected
  }
  if (transportMode === 'sse') {
    return styles.wsFallback
  }
  return styles.wsDisconnected
}

export const resolveTransportStatusTitle = (
  transportMode: RoomTransportMode,
  wsConnected: boolean
): string => {
  if (transportMode === 'websocket' && wsConnected) {
    return 'WebSocket connected'
  }
  if (transportMode === 'sse') {
    return 'WebSocket disconnected (SSE fallback active)'
  }
  return 'Reconnecting transport'
}

export const resolveStreamingParticipantDisplay = (
  messageId: string,
  streamingParticipants: Map<string, StreamingParticipant>
): { participantType: string; displayName: string; isLeader: boolean; isWorker: boolean } => {
  const participant = streamingParticipants.get(messageId)
  const participantType = participant?.participantType || 'worker'
  const isLeader = participantType === 'leader'
  const isWorker = participantType === 'worker'
  const displayName = participant?.participantId || (isLeader ? 'Leader' : 'Worker')
  return { participantType, displayName, isLeader, isWorker }
}
