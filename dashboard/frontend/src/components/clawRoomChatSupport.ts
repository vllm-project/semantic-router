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
}

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
