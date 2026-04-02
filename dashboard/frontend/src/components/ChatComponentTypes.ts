import type { ToolCall, ToolResult, WebSearchResult } from '../tools'
import { OPENCLAW_MCP_SERVER_ID } from '../tools/mcp/api'

export const GREETING_LINES = [
  'Hi there, I am MoM :-)',
  'The System Intelligence for Agents and LLMs',
  'The World First Model-of-Models',
  'Open Source for Everyone',
  'How can I help you today?'
]

export const generateMessageId = () => `msg-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`
export const generateConversationId = () => `conv-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`
export const generatePlaygroundTaskId = () => `task-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`
export const CLAW_TOOL_NAME_PREFIX = `mcp_${OPENCLAW_MCP_SERVER_ID}_claw_`
export const CLAW_MODE_STORAGE_KEY = 'sr:playground:claw-mode'
export const PLAYGROUND_QUEUE_STORAGE_KEY = 'sr:playground:queue'
export const CLAW_MODE_SYSTEM_PROMPT_LINES = [
  'You are HireClaw, an elite recruiter and talent partner for building Claw Teams as memorable anthropomorphic specialists with real workplace presence, not generic bots.',
  'Quick context: OpenClaw is the overall agent platform; HireClaw is the recruiting and hiring mode in this chat; a Claw Team is an organizational unit; a Claw Worker is an individual anthropomorphic agent inside a team.',
  'You should still answer normal user questions naturally when the user is not hiring or managing Claw teams.',
  'When user intent is to create or manage Claw Teams/Workers:',
  '1) Act like a real recruiter: understand the mission, hiring gaps, reporting structure, collaboration needs, and success criteria before recommending talent.',
  '2) For worker creation, present a shortlist of 2-3 candidates by default before any mutating tool call. Each candidate should include: English first-name identity, role/specialty, recruiter-style fit note, vibe/collaboration style, pressure behavior, and team-fit/reporting pattern.',
  '3) Worker names should usually be plausible English first names that feel memorable and recruiter-grade. Avoid bot labels, fantasy handles, emoji-only names, and generic titles such as Worker A, Analyst Bot, Operator-1, Helper, Assistant, or pure role titles.',
  "4) If the user already provides a worker name, respect it unless they explicitly ask for alternatives or a rename.",
  "5) Other descriptive fields (such as role/vibe/principles/descriptions) should follow the user's language preference inferred from the conversation, but worker names should stay in English unless the user explicitly requests another convention.",
  '6) Vibe must feel like a specific human collaboration style: include temperament, communication rhythm, emotional tone, and how the worker reacts under pressure. Avoid bland labels like calm, professional, helpful, or smart unless made concrete.',
  '7) Principles must read like team-aware operating rules, not empty slogans. They MUST explicitly include team mission, leader coordination, teammate expectations, escalation boundaries, and how this worker collaborates day to day.',
  '8) Before executing team/worker creation tools (or other mutating Claw actions), first present a concise hiring slate or recommendation plan and wait for explicit user approval. The pre-tool output should read like recruiter guidance, not a generic character sheet.',
  '9) Team design MUST include exactly one leader. Ensure one worker is designated with role_kind="leader", and ensure team leader_id points to that leader (set on creation if possible, otherwise update the team after creating workers).',
]
export const CLAW_MODE_SYSTEM_PROMPT = CLAW_MODE_SYSTEM_PROMPT_LINES.join('\n')

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

export interface Message {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: Date
  isStreaming?: boolean
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
  enableClawMode: boolean
  enableWebSearch: boolean
  model: string
}

export interface PlaygroundTask {
  id: string
  conversationId: string
  prompt: string
  createdAt: number
  requestOptions: PlaygroundTaskRequestOptions
}

interface ClawHighlightField {
  label: string
  value: string
}

const asRecord = (value: unknown): Record<string, unknown> | null => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return null
  }
  return value as Record<string, unknown>
}

const toFieldString = (value: unknown): string => {
  if (value === null || value === undefined) return ''
  if (typeof value === 'string') return value.trim()
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  return ''
}

export const truncateHighlight = (value: string, maxLength = 120): string => {
  const text = value.trim()
  if (text.length <= maxLength) return text
  return `${text.slice(0, maxLength - 3).trim()}...`
}

const extractFromRawArgs = (rawArgs: string, key: string): string => {
  if (!rawArgs) return ''
  const escapedKey = key.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const regex = new RegExp(`"${escapedKey}"\\s*:\\s*"([^"]*)`)
  const match = rawArgs.match(regex)
  return match?.[1]?.trim() || ''
}

const firstFieldValue = (
  source: Record<string, unknown> | null,
  keys: string[],
  rawArgs = ''
): string => {
  for (const key of keys) {
    const value = toFieldString(source?.[key])
    if (value) return value
  }
  if (rawArgs) {
    for (const key of keys) {
      const value = extractFromRawArgs(rawArgs, key)
      if (value) return value
    }
  }
  return ''
}

const toHighlightFields = (pairs: Array<[string, string]>): ClawHighlightField[] => {
  return pairs
    .filter(([, value]) => Boolean(value))
    .map(([label, value]) => ({ label, value: truncateHighlight(value) }))
}

export const buildClawRequestHighlights = (
  clawToolName: string,
  parsedArgs: Record<string, unknown> | null,
  rawArgs: string
): Array<{ label: string; value: string }> => {
  if (clawToolName === 'claw_create_team') {
    return toHighlightFields([
      ['name', firstFieldValue(parsedArgs, ['name'], rawArgs)],
      ['vibe', firstFieldValue(parsedArgs, ['vibe'], rawArgs)],
      ['role', firstFieldValue(parsedArgs, ['role'], rawArgs)],
      ['principal', firstFieldValue(parsedArgs, ['principal'], rawArgs)],
    ])
  }

  if (clawToolName === 'claw_create_worker') {
    return toHighlightFields([
      ['name', firstFieldValue(parsedArgs, ['name'], rawArgs)],
      ['vibe', firstFieldValue(parsedArgs, ['vibe'], rawArgs)],
      ['role', firstFieldValue(parsedArgs, ['role'], rawArgs)],
      ['team', firstFieldValue(parsedArgs, ['team_id', 'teamId'], rawArgs)],
      ['emoji', firstFieldValue(parsedArgs, ['emoji'], rawArgs)],
    ])
  }

  return []
}

export const buildClawResultHighlights = (
  clawToolName: string,
  resultContent: unknown,
  parsedArgs: Record<string, unknown> | null,
  rawArgs: string
): Array<{ label: string; value: string }> => {
  const result = asRecord(resultContent)

  if (clawToolName === 'claw_create_team') {
    return toHighlightFields([
      ['name', firstFieldValue(result, ['name']) || firstFieldValue(parsedArgs, ['name'], rawArgs)],
      ['vibe', firstFieldValue(result, ['vibe']) || firstFieldValue(parsedArgs, ['vibe'], rawArgs)],
      ['role', firstFieldValue(result, ['role']) || firstFieldValue(parsedArgs, ['role'], rawArgs)],
      ['team_id', firstFieldValue(result, ['id'])],
    ])
  }

  if (clawToolName === 'claw_create_worker') {
    const identity = asRecord(result?.identity)
    return toHighlightFields([
      ['name', firstFieldValue(identity, ['name']) || firstFieldValue(result, ['agentName', 'name']) || firstFieldValue(parsedArgs, ['name'], rawArgs)],
      ['vibe', firstFieldValue(identity, ['vibe']) || firstFieldValue(result, ['agentVibe']) || firstFieldValue(parsedArgs, ['vibe'], rawArgs)],
      ['role', firstFieldValue(identity, ['role']) || firstFieldValue(result, ['agentRole']) || firstFieldValue(parsedArgs, ['role'], rawArgs)],
      ['team', firstFieldValue(result, ['teamName', 'teamId']) || firstFieldValue(parsedArgs, ['team_id', 'teamId'], rawArgs)],
      ['container', firstFieldValue(result, ['containerName'])],
      ['message', firstFieldValue(result, ['message'])],
    ])
  }

  return []
}
