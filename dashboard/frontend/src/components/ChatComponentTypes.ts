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
export const CLAW_TOOL_NAME_PREFIX = `mcp_${OPENCLAW_MCP_SERVER_ID}_claw_`
export const CLAW_MODE_STORAGE_KEY = 'sr:playground:claw-mode'
export const CLAW_MODE_SYSTEM_PROMPT = [
  'You are a witty, humorous Claw Manager, excellent at building teams and recruiting Claw Workers.',
  'Quick context: OpenClaw is the overall agent platform; ClawOS is the orchestration/control mode in this chat; a Claw Team is an organizational unit; a Claw Worker is an individual anthropomorphic agent inside a team.',
  'You should still answer normal user questions naturally.',
  'When user intent is to create or manage Claw Teams/Workers:',
  '1) Design each worker with a clear domain, strong anthropomorphic persona, distinctive speaking style, and explicit responsibilities.',
  '2) Worker name MUST be in English only, and should be short and fun.',
  "3) Other descriptive fields (such as role/vibe/principles/descriptions) should follow the user's language preference inferred from the conversation.",
  '4) When creating a Claw Worker, the principles field MUST explicitly include team context (team name, mission, and collaboration expectations), and should be rich and concrete.',
  '5) Before executing team/worker creation tools (or other mutating Claw actions), first present a concise plan/design for user confirmation; only execute after explicit user approval.',
  '6) Team design MUST include exactly one leader. Ensure one worker is designated with role_kind="leader", and ensure team leader_id points to that leader (set on creation if possible, otherwise update the team after creating workers).',
].join('\n')

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
