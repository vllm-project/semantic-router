import type { ToolCall, ToolResult } from '../tools'

export interface ClawRoomToolTraceStep {
  id: string
  name: string
  arguments?: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  result?: string
  error?: string
}

export interface ClawRoomToolTracePayload {
  revision?: number
  steps?: ClawRoomToolTraceStep[]
}

export type ClawRoomStreamingToolTraceEntry = {
  revision?: number
  steps: ClawRoomToolTraceStep[]
}

export const CLAW_ROOM_TOOL_TRACE_METADATA_KEY = 'toolTrace'

const normalizeToolTraceStatus = (
  status: string | undefined
): ClawRoomToolTraceStep['status'] => {
  if (status === 'completed' || status === 'failed' || status === 'pending') {
    return status
  }
  return 'running'
}

const normalizeClawRoomToolTraceStep = (
  value: unknown
): ClawRoomToolTraceStep | null => {
  if (!value || typeof value !== 'object') {
    return null
  }
  const step = value as Record<string, unknown>
  const id = typeof step.id === 'string' ? step.id : ''
  if (!id) {
    return null
  }
  return {
    id,
    name: typeof step.name === 'string' ? step.name : 'tool',
    arguments: typeof step.arguments === 'string' ? step.arguments : undefined,
    status: normalizeToolTraceStatus(typeof step.status === 'string' ? step.status : undefined),
    result: typeof step.result === 'string' ? step.result : undefined,
    error: typeof step.error === 'string' ? step.error : undefined,
  }
}

const normalizeClawRoomToolTraceSteps = (values: unknown[]): ClawRoomToolTraceStep[] => {
  return values.flatMap(value => {
    const step = normalizeClawRoomToolTraceStep(value)
    return step ? [step] : []
  })
}

export const parseClawRoomToolTracePayload = (
  payload: Record<string, unknown> | undefined
): ClawRoomToolTracePayload | null => {
  if (!payload || !Array.isArray(payload.steps)) {
    return null
  }

  return {
    revision: typeof payload.revision === 'number' ? payload.revision : undefined,
    steps: normalizeClawRoomToolTraceSteps(payload.steps),
  }
}

export const parseClawRoomToolTraceFromMessageMetadata = (
  metadata?: Record<string, string>
): ClawRoomToolTraceStep[] => {
  const raw = metadata?.[CLAW_ROOM_TOOL_TRACE_METADATA_KEY]
  if (!raw) {
    return []
  }

  try {
    const parsed = JSON.parse(raw) as unknown
    if (!Array.isArray(parsed)) {
      return []
    }
    return normalizeClawRoomToolTraceSteps(parsed)
  } catch {
    return []
  }
}

export const applyClawRoomToolTraceRevision = (
  current: ClawRoomStreamingToolTraceEntry | undefined,
  incoming: ClawRoomToolTracePayload
): ClawRoomStreamingToolTraceEntry | null => {
  if (!incoming.steps?.length) {
    return null
  }
  if (
    current?.revision != null &&
    incoming.revision != null &&
    incoming.revision < current.revision
  ) {
    return null
  }
  return {
    revision: incoming.revision ?? current?.revision,
    steps: incoming.steps,
  }
}

export const toPlaygroundToolCall = (step: ClawRoomToolTraceStep): ToolCall => ({
  id: step.id,
  type: 'function',
  function: {
    name: step.name,
    arguments: step.arguments || '{}',
  },
  status: step.status,
})

export const toPlaygroundToolResult = (step: ClawRoomToolTraceStep): ToolResult | undefined => {
  if (step.status !== 'completed' && step.status !== 'failed') {
    return undefined
  }
  if (step.status === 'failed') {
    return {
      callId: step.id,
      name: step.name,
      content: step.error || step.result || 'Tool failed',
      error: step.error || 'Tool failed',
    }
  }
  return {
    callId: step.id,
    name: step.name,
    content: step.result ?? '',
  }
}
