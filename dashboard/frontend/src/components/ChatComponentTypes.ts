import type { ToolCall, ToolResult, WebSearchResult } from '../tools'
import type { PlaygroundAttachment, PlaygroundAttachmentSummary } from './playgroundFileAttachments'

export type { PlaygroundAttachment, PlaygroundAttachmentSummary }

export const generateMessageId = () =>
  `msg-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`
export const generateConversationId = () =>
  `conv-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`
export const generatePlaygroundTaskId = () =>
  `task-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`
export const PLAYGROUND_QUEUE_STORAGE_KEY = 'sr:playground:queue'
export const PLAYGROUND_ACTIVE_CONVERSATION_STORAGE_KEY = 'sr:playground:active-conversation'

export interface Choice {
  content: string
  model?: string
}

export interface ReMoMIntermediateResp {
  model: string
  content: string
  reasoning?: string
  compacted_content?: string
  token_count?: number
}

export interface ReMoMRoundResponse {
  round: number
  breadth: number
  responses: ReMoMIntermediateResp[]
}

export type SearchResult = WebSearchResult

export interface InlineMessageImage {
  src: string
  alt: string
}

export interface MessagePresentation {
  content: string
  images?: InlineMessageImage[]
}

export interface Message {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  attachments?: PlaygroundAttachmentSummary[]
  playgroundAttachments?: PlaygroundAttachment[]
  requestContent?: unknown
  images?: InlineMessageImage[]
  timestamp: Date
  isStreaming?: boolean
  incomplete?: string
  headers?: Record<string, string>
  choices?: Choice[]
  thinkingProcess?: string
  toolCalls?: ToolCall[]
  toolResults?: ToolResult[]
  reasoning_mom_responses?: ReMoMRoundResponse[]
}

export interface ConversationPreview {
  id: string
  updatedAt: number
  preview: string
}

export interface PlaygroundTaskRequestOptions {
  enableWebSearch: boolean
  model: string
  executeToolCalls?: boolean
}

export interface PlaygroundTask {
  id: string
  conversationId: string
  prompt: string
  attachments?: PlaygroundAttachment[]
  createdAt: number
  requestOptions: PlaygroundTaskRequestOptions
  exactRequest?: Record<string, unknown>
  appendPromptMessage?: boolean
  displayMessage?: MessagePresentation
}
