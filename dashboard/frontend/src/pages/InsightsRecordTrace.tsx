import { useMemo } from 'react'
import { CheckCircle2, ChevronDown, OctagonAlert, Wrench } from 'lucide-react'

import MarkdownRenderer from '../components/MarkdownRenderer'
import {
  Badge,
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  cn,
} from '../ui'

import { buildConversationLabels } from './insightsConversationIdentity'
import { CHEVRON, DETAIL, RAW } from './insightsEvidenceStyles'
import type {
  InsightsRecord,
  InsightsTrajectory,
  InsightsTrajectoryMessage,
  InsightsTrajectoryToolCall,
} from './insightsPageTypes'

interface InsightsRecordTraceProps {
  record: InsightsRecord
  trajectory: InsightsTrajectory | null
  loading?: boolean
  error?: string | null
}

interface TraceTurn {
  conversationId: string
  index: number
  messages: InsightsTrajectoryMessage[]
}

const NOTE = 'text-[11px] text-subtle-foreground'

export default function InsightsRecordTrace({
  record,
  trajectory,
  loading = false,
  error,
}: InsightsRecordTraceProps) {
  const messages = useMemo(
    () => trajectory?.messages ?? buildRecordTraceFallback(record),
    [record, trajectory],
  )
  const turns = useMemo(() => groupTraceTurns(messages), [messages])
  const conversationLabels = useMemo(
    () => buildConversationLabels([...(trajectory?.routes ?? []), ...messages]),
    [trajectory, messages],
  )
  const toolCount = messages.reduce(
    (count, message) => count + (message.tool_calls?.length ?? 0),
    0,
  )

  return (
    <Card role="region" aria-labelledby="record-trace-title">
      <CardHeader>
        <CardTitle id="record-trace-title">Conversation</CardTitle>
        <CardDescription>Record trace of the session, as Replay kept it</CardDescription>
        <CardAction>
          <Badge tone="outline">
            {turns.length} {turns.length === 1 ? 'turn' : 'turns'}
            {toolCount > 0 ? ` · ${toolCount} ${toolCount === 1 ? 'tool call' : 'tool calls'}` : ''}
          </Badge>
        </CardAction>
      </CardHeader>

      <CardContent className="flex flex-col gap-2 pt-3">
        {loading ? (
          <p className="text-[13px] text-muted-foreground">Loading the complete session…</p>
        ) : null}
        {!loading && error ? (
          <p className="text-[13px] text-muted-foreground">
            The session trace could not be loaded. {error}
          </p>
        ) : null}
        {!loading && !error && turns.length === 0 ? (
          <p className="text-[13px] text-muted-foreground">No conversation steps were captured.</p>
        ) : null}

        {!loading && !error && turns.length > 0
          ? turns.map((turn, turnPosition) => {
              const turnToolCount = turn.messages.reduce(
                (count, message) => count + (message.tool_calls?.length ?? 0),
                0,
              )
              return (
                <details key={`${turn.index}-${turnPosition}`} className={DETAIL}>
                  <summary>
                    <span className="grid size-5 shrink-0 place-items-center rounded-full bg-secondary font-mono text-[10.5px] text-muted-foreground">
                      {turnPosition + 1}
                    </span>
                    <span className="flex min-w-0 flex-1 flex-col">
                      <strong
                        className="truncate font-medium text-foreground"
                        title={turn.conversationId || undefined}
                      >
                        {turn.conversationId
                          ? `${conversationLabels.get(turn.conversationId)} · Turn ${turn.index + 1}`
                          : `Turn ${turnPosition + 1}`}
                      </strong>
                      <span className="truncate font-normal text-subtle-foreground">
                        {buildTurnPreview(turn)}
                      </span>
                    </span>
                    <span className="shrink-0 font-normal text-subtle-foreground">
                      {turnToolCount > 0
                        ? `${turnToolCount} ${turnToolCount === 1 ? 'tool' : 'tools'}`
                        : `${turn.messages.length} steps`}
                    </span>
                    <ChevronDown className={CHEVRON} />
                  </summary>
                  <div className="flex flex-col gap-2.5 border-t border-border px-3 py-3">
                    {turn.messages.map((message, messageIndex) => (
                      <TraceMessage
                        key={`${message.role}-${message.tool_call_id ?? ''}-${messageIndex}`}
                        message={message}
                      />
                    ))}
                  </div>
                </details>
              )
            })
          : null}
      </CardContent>
    </Card>
  )
}

function TraceMessage({ message }: { message: InsightsTrajectoryMessage }) {
  if (message.role === 'assistant' && message.tool_calls?.length) {
    return <ToolCalls calls={message.tool_calls} redacted={message.content_redacted} />
  }

  if (message.role === 'tool') {
    return <ToolResult message={message} />
  }

  const isUser = message.role === 'user'
  return (
    <div className={cn('flex flex-col gap-1', isUser ? 'items-end pl-6' : 'items-start pr-4')}>
      <span className={NOTE}>{isUser ? 'You' : 'Response'}</span>
      {message.content ? (
        <div
          className={cn(
            'max-w-full rounded-2xl px-3.5 py-2 text-[13px] leading-relaxed [&_p]:my-1 [&_pre]:max-w-full [&_pre]:overflow-x-auto',
            isUser
              ? 'rounded-br-md bg-primary-soft text-foreground'
              : 'rounded-tl-md border border-border bg-surface-subtle',
          )}
        >
          <MarkdownRenderer content={message.content} />
          {message.content_truncated ? (
            <span className={NOTE}>Text truncated at the configured tool trace byte limit</span>
          ) : null}
        </div>
      ) : (
        <span className="rounded-lg border border-dashed border-border-strong px-3 py-1.5 text-xs text-muted-foreground">
          {message.content_redacted ? 'Content hidden for this role' : 'No content captured'}
        </span>
      )}
    </div>
  )
}

function ToolCalls({
  calls,
  redacted,
}: {
  calls: InsightsTrajectoryToolCall[]
  redacted?: boolean
}) {
  return (
    <details className={DETAIL}>
      <summary>
        <Wrench className="size-3.5 shrink-0 text-chart-5" />
        <span className="min-w-0 flex-1 truncate font-mono">
          {calls.length === 1 ? calls[0].function.name : `${calls.length} tools requested`}
        </span>
        <ChevronDown className={CHEVRON} />
      </summary>
      <div className="flex flex-col gap-2 border-t border-border px-3 py-2.5">
        {calls.map((call) => (
          <div key={call.id || call.function.name} className="flex flex-col gap-1">
            <strong className="font-mono text-xs font-medium">
              {call.function.name || 'Tool'}
            </strong>
            {call.function.arguments ? (
              <pre className={cn(RAW, 'whitespace-pre-wrap break-all')}>
                {formatTracePayload(call.function.arguments)}
              </pre>
            ) : (
              <span className={NOTE}>
                {redacted ? 'Arguments hidden for this role' : 'No arguments captured'}
              </span>
            )}
          </div>
        ))}
      </div>
    </details>
  )
}

function ToolResult({ message }: { message: InsightsTrajectoryMessage }) {
  const failed = message.status === 'failed'
  return (
    <details className={DETAIL}>
      <summary>
        {failed ? (
          <OctagonAlert className="size-3.5 shrink-0 text-destructive" />
        ) : (
          <CheckCircle2 className="size-3.5 shrink-0 text-success" />
        )}
        <span className="min-w-0 flex-1 truncate">{message.tool_name || 'Tool'} result</span>
        <ChevronDown className={CHEVRON} />
      </summary>
      <div className="flex flex-col gap-1 border-t border-border px-3 py-2.5">
        {message.content ? (
          <>
            <pre className={cn(RAW, 'whitespace-pre-wrap break-all')}>
              {formatTracePayload(message.content)}
            </pre>
            {message.content_truncated ? (
              <span className={NOTE}>Result truncated at the configured tool trace byte limit</span>
            ) : null}
          </>
        ) : (
          <span className={NOTE}>
            {message.content_redacted
              ? `Result hidden · ${message.status || 'recorded'}`
              : 'No result captured'}
          </span>
        )}
      </div>
    </details>
  )
}

function groupTraceTurns(messages: InsightsTrajectoryMessage[]): TraceTurn[] {
  const turns: TraceTurn[] = []
  const positions = new Map<string, number>()
  for (const message of messages) {
    const index = message.turn_index ?? 0
    const conversationId = message.conversation_id ?? ''
    const key = JSON.stringify([conversationId, index])
    const existing = positions.get(key)
    if (existing === undefined) {
      positions.set(key, turns.length)
      turns.push({ index, conversationId, messages: [message] })
      continue
    }
    turns[existing].messages.push(message)
  }
  return turns
}

function buildTurnPreview(turn: TraceTurn) {
  const userMessage = turn.messages.find((message) => message.role === 'user')
  const fallback = turn.messages.find((message) => Boolean(message.content))
  const content = (userMessage?.content || fallback?.content || 'Recorded conversation step')
    .replace(/\s+/g, ' ')
    .trim()
  return content.length > 96 ? `${content.slice(0, 93)}…` : content
}

function buildRecordTraceFallback(record: InsightsRecord): InsightsTrajectoryMessage[] {
  const messages: InsightsTrajectoryMessage[] = []
  for (const step of record.tool_trace?.steps ?? []) {
    switch (step.type) {
      case 'user_input':
        messages.push({
          role: 'user',
          content: step.text,
          content_truncated: Boolean(step.text && step.truncated),
          content_redacted: step.content_redacted,
          turn_index: record.turn_index ?? 0,
        })
        break
      case 'assistant_tool_call':
        messages.push({
          role: 'assistant',
          turn_index: record.turn_index ?? 0,
          content_redacted: step.content_redacted,
          tool_calls: [
            {
              id: step.tool_call_id || '',
              type: 'function',
              function: { name: step.tool_name || 'Tool', arguments: step.arguments || '' },
            },
          ],
        })
        break
      case 'client_tool_result':
        messages.push({
          role: 'tool',
          content: step.output || step.text,
          content_truncated: Boolean(step.output && step.truncated),
          tool_call_id: step.tool_call_id,
          tool_name: step.tool_name,
          status: step.status === 'failed' ? 'failed' : 'succeeded',
          content_redacted: step.content_redacted,
          turn_index: record.turn_index ?? 0,
        })
        break
      case 'assistant_final_response':
        messages.push({
          role: 'assistant',
          content: step.text,
          content_truncated: Boolean(step.text && step.truncated),
          content_redacted: step.content_redacted,
          turn_index: record.turn_index ?? 0,
        })
        break
      default:
        break
    }
  }
  return messages.map((message) => ({ ...message, conversation_id: record.conversation_id }))
}

function formatTracePayload(value: string) {
  try {
    return JSON.stringify(JSON.parse(value), null, 2)
  } catch {
    return value
  }
}
