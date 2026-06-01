import type { RoomMessage, RoomTransportMode, StreamingParticipant } from './clawRoomChatSupport'

export const buildStreamingEntries = (
  messages: RoomMessage[],
  streamingMessages: Map<string, string>
): Array<[string, string]> => {
  return Array.from(streamingMessages.entries()).filter(([messageId, content]) => {
    if (!content.trim()) {
      return false
    }
    const persisted = messages.find(message => message.id === messageId)
    return !persisted || persisted.content !== content
  })
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
