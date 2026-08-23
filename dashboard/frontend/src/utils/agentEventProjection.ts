import type {
  AgentApprovalRequestPayload,
  AgentEvent,
  AgentLiveModelStepEvent,
  AgentProgressPayload,
  AgentTerminalPayload,
  AgentToolRequestPayload,
  AgentToolResultPayload,
  AgentUserInputPayload,
} from '../generated/managementApiContract'

export interface AgentTimelineMessage {
  kind: 'message'
  id: string
  turnId: string
  role: 'user' | 'assistant'
  text: string
  createdAt: string
  streaming: boolean
}

export interface AgentTimelineTool {
  kind: 'tool'
  id: string
  turnId: string
  invocationId: string
  name: string
  classification: 'read' | 'write' | 'execute'
  status: 'running' | 'completed' | 'failed' | 'cancelled'
  summary: string
  artifactId?: string
  error?: string
  createdAt: string
}

export interface AgentTimelineProgress {
  kind: 'progress'
  id: string
  turnId: string
  phase: string
  message: string
  createdAt: string
}

export interface AgentTimelineApproval {
  kind: 'approval'
  id: string
  turnId: string
  payload: AgentApprovalRequestPayload
  status: 'waiting' | 'committed' | 'rejected' | 'expired' | 'failed'
  createdAt: string
}

export interface AgentTimelineTerminal {
  kind: 'terminal'
  id: string
  turnId: string
  payload: AgentTerminalPayload
  createdAt: string
}

export type AgentTimelineItem =
  | AgentTimelineMessage
  | AgentTimelineTool
  | AgentTimelineProgress
  | AgentTimelineApproval
  | AgentTimelineTerminal

function textFromInput(payload: AgentUserInputPayload): string {
  const text = payload.content
    .filter(
      (block): block is Extract<(typeof payload.content)[number], { type: 'text' }> =>
        block.type === 'text',
    )
    .map((block) => block.text.trim())
    .filter(Boolean)
    .join('\n\n')
  const attachmentCount = payload.content.filter((block) => block.type !== 'text').length
  if (!text && attachmentCount > 0)
    return `${attachmentCount} attachment${attachmentCount === 1 ? '' : 's'}`
  return text
}

function compactToolSummary(value: unknown): string {
  if (typeof value === 'string') return value.trim().slice(0, 180)
  if (!value || typeof value !== 'object' || Array.isArray(value)) return ''
  const record = value as Record<string, unknown>
  for (const key of ['query', 'name', 'recipeId', 'entrypointId', 'modelId', 'phase']) {
    const candidate = record[key]
    if (typeof candidate === 'string' && candidate.trim()) return candidate.trim().slice(0, 180)
  }
  return ''
}

export function projectAgentTimeline(
  events: readonly AgentEvent[],
  liveModelSteps: readonly AgentLiveModelStepEvent[] = [],
): AgentTimelineItem[] {
  const items: AgentTimelineItem[] = []
  const assistantByStep = new Map<string, AgentTimelineMessage>()
  const durableModelSteps = new Set<string>()
  const toolByInvocation = new Map<string, AgentTimelineTool>()
  const approvalByPlan = new Map<string, AgentTimelineApproval>()
  const terminalTurns = new Set<string>()

  for (const event of [...events].sort((left, right) => left.sequence - right.sequence)) {
    const turnId = event.turnId ?? `session-${event.sessionId}`
    if (event.type === 'user_input') {
      const payload = event.payload as AgentUserInputPayload
      items.push({
        kind: 'message',
        id: `user-${event.sequence}`,
        turnId,
        role: 'user',
        text: textFromInput(payload),
        createdAt: event.createdAt,
        streaming: false,
      })
      continue
    }
    if (event.type === 'assistant_delta') {
      const { modelStepId, delta } = event.payload
      if (delta.kind !== 'text' || !delta.text) continue
      durableModelSteps.add(modelStepId)
      let message = assistantByStep.get(modelStepId)
      if (!message) {
        message = {
          kind: 'message',
          id: `assistant-${modelStepId}`,
          turnId,
          role: 'assistant',
          text: '',
          createdAt: event.createdAt,
          streaming: false,
        }
        assistantByStep.set(modelStepId, message)
        items.push(message)
      }
      message.text += delta.text
      continue
    }
    if (event.type === 'tool_request') {
      const payload = event.payload as AgentToolRequestPayload
      const tool: AgentTimelineTool = {
        kind: 'tool',
        id: `tool-${payload.invocationId}`,
        turnId,
        invocationId: payload.invocationId,
        name: payload.toolName,
        classification: payload.class,
        status: 'running',
        summary: compactToolSummary(payload.arguments),
        createdAt: event.createdAt,
      }
      toolByInvocation.set(payload.invocationId, tool)
      items.push(tool)
      continue
    }
    if (event.type === 'tool_result') {
      const payload = event.payload as AgentToolResultPayload
      let tool = toolByInvocation.get(payload.invocationId)
      if (!tool) {
        tool = {
          kind: 'tool',
          id: `tool-${payload.invocationId}`,
          turnId,
          invocationId: payload.invocationId,
          name: payload.toolName,
          classification: 'execute',
          status: payload.status,
          summary: compactToolSummary(payload.result),
          createdAt: event.createdAt,
        }
        toolByInvocation.set(payload.invocationId, tool)
        items.push(tool)
      }
      tool.status = payload.status
      tool.artifactId = payload.artifactId
      tool.error = payload.error?.message
      if (!tool.summary) tool.summary = compactToolSummary(payload.result)
      continue
    }
    if (event.type === 'progress') {
      const payload = event.payload as AgentProgressPayload
      const previous = items[items.length - 1]
      if (previous?.kind === 'progress' && previous.turnId === turnId) {
        previous.phase = payload.phase
        previous.message = payload.message
        previous.createdAt = event.createdAt
      } else {
        items.push({
          kind: 'progress',
          id: `progress-${event.sequence}`,
          turnId,
          phase: payload.phase,
          message: payload.message,
          createdAt: event.createdAt,
        })
      }
      continue
    }
    if (event.type === 'approval_request') {
      const payload = event.payload as AgentApprovalRequestPayload
      const approval: AgentTimelineApproval = {
        kind: 'approval',
        id: `approval-${payload.planId}`,
        turnId,
        payload,
        status: 'waiting',
        createdAt: event.createdAt,
      }
      approvalByPlan.set(payload.planId, approval)
      items.push(approval)
      continue
    }
    if (event.type === 'approval_result') {
      const approval = approvalByPlan.get(event.payload.planId)
      if (approval) approval.status = event.payload.status
      continue
    }
    if (event.type === 'terminal') {
      terminalTurns.add(turnId)
      items.push({
        kind: 'terminal',
        id: `terminal-${event.sequence}`,
        turnId,
        payload: event.payload as AgentTerminalPayload,
        createdAt: event.createdAt,
      })
    }
  }

  const previewByStep = new Map<string, AgentTimelineMessage>()
  for (const event of liveModelSteps) {
    if (
      event.phase !== 'delta' ||
      !event.delta?.text ||
      durableModelSteps.has(event.modelStepId) ||
      terminalTurns.has(event.turnId)
    ) {
      continue
    }
    let message = previewByStep.get(event.modelStepId)
    if (!message) {
      message = {
        kind: 'message',
        id: `assistant-preview-${event.modelStepId}`,
        turnId: event.turnId,
        role: 'assistant',
        text: '',
        createdAt: event.createdAt,
        streaming: true,
      }
      previewByStep.set(event.modelStepId, message)
      items.push(message)
    }
    message.text += event.delta.text
  }
  return items
}

export function activeAgentTurnId(events: readonly AgentEvent[]): string | null {
  const status = new Map<string, AgentTurnStatusLike>()
  for (const event of events) {
    if (!event.turnId) continue
    if (event.type === 'user_input') status.set(event.turnId, 'running')
    if (event.type === 'approval_request') status.set(event.turnId, 'waiting_approval')
    if (event.type === 'terminal') status.set(event.turnId, event.payload.status)
    if (event.type === 'cancellation') status.set(event.turnId, 'running')
  }
  return [...status.entries()].find(([, value]) => value === 'running')?.[0] ?? null
}

export function agentTurnIsTerminal(events: readonly AgentEvent[], turnId: string): boolean {
  return events.some((event) => event.turnId === turnId && event.type === 'terminal')
}

type AgentTurnStatusLike = 'running' | 'waiting_approval' | 'completed' | 'failed' | 'cancelled'

export function pendingApproval(events: readonly AgentEvent[]): AgentApprovalRequestPayload | null {
  const approvals = new Map<string, AgentApprovalRequestPayload>()
  for (const event of events) {
    if (event.type === 'approval_request') approvals.set(event.payload.planId, event.payload)
    if (event.type === 'approval_result') approvals.delete(event.payload.planId)
  }
  const pending = [...approvals.values()]
  return pending[pending.length - 1] ?? null
}
