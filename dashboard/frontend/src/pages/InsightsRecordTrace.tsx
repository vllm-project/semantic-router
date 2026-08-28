import { useMemo } from 'react'

import MarkdownRenderer from '../components/MarkdownRenderer'
import ProductIcon from '../components/ProductIcon'

import styles from './InsightsPage.module.css'
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
  index: number
  messages: InsightsTrajectoryMessage[]
}

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
  const toolCount = messages.reduce(
    (count, message) => count + (message.tool_calls?.length ?? 0),
    0,
  )

  return (
    <details className={styles.recordTrace}>
      <summary className={styles.recordTraceHeader}>
        <div>
          <span className={styles.recordTraceEyebrow}>Conversation</span>
          <h2 id="record-trace-title">Record trace</h2>
        </div>
        <div className={styles.recordTraceHeaderMeta}>
          <span className={styles.recordTraceSummary}>
            {turns.length} {turns.length === 1 ? 'turn' : 'turns'}
            {toolCount > 0 ? ` · ${toolCount} ${toolCount === 1 ? 'tool call' : 'tool calls'}` : ''}
          </span>
          <ProductIcon
            name="chevron-down"
            width={15}
            height={15}
            className={styles.recordTraceHeaderChevron}
          />
        </div>
      </summary>

      <div className={styles.recordTraceContent}>
        {loading ? (
          <div className={styles.recordTraceEmpty}>Loading the complete session…</div>
        ) : null}
        {!loading && error ? (
          <div className={styles.recordTraceEmpty}>
            The session trace could not be loaded. {error}
          </div>
        ) : null}
        {!loading && !error && turns.length === 0 ? (
          <div className={styles.recordTraceEmpty}>No conversation steps were captured.</div>
        ) : null}

        {!loading && !error && turns.length > 0 ? (
          <div className={styles.recordTraceTurns}>
            {turns.map((turn, turnPosition) => {
              const turnToolCount = turn.messages.reduce(
                (count, message) => count + (message.tool_calls?.length ?? 0),
                0,
              )
              return (
                <details key={`${turn.index}-${turnPosition}`} className={styles.recordTraceTurn}>
                  <summary className={styles.recordTraceTurnSummary}>
                    <span className={styles.recordTraceTurnIndex}>{turnPosition + 1}</span>
                    <span className={styles.recordTraceTurnLabel}>
                      <strong>Turn {turnPosition + 1}</strong>
                      <span>{buildTurnPreview(turn)}</span>
                    </span>
                    <span className={styles.recordTraceTurnMeta}>
                      {turnToolCount > 0
                        ? `${turnToolCount} ${turnToolCount === 1 ? 'tool' : 'tools'}`
                        : `${turn.messages.length} steps`}
                    </span>
                    <ProductIcon name="chevron-down" width={14} height={14} />
                  </summary>
                  <div className={styles.recordTraceMessages}>
                    {turn.messages.map((message, messageIndex) => (
                      <TraceMessage
                        key={`${message.role}-${message.tool_call_id ?? ''}-${messageIndex}`}
                        message={message}
                      />
                    ))}
                  </div>
                </details>
              )
            })}
          </div>
        ) : null}
      </div>
    </details>
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
    <div className={`${styles.recordTraceMessage} ${isUser ? styles.recordTraceUser : ''}`}>
      <span className={styles.recordTraceRole}>{isUser ? 'You' : 'Response'}</span>
      {message.content ? (
        <div className={styles.recordTraceMarkdown}>
          <MarkdownRenderer content={message.content} />
        </div>
      ) : (
        <span className={styles.recordTraceRedacted}>
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
    <details className={styles.recordTraceTool}>
      <summary>
        <ProductIcon name="tool" width={15} height={15} />
        <span>
          {calls.length === 1 ? calls[0].function.name : `${calls.length} tools requested`}
        </span>
        <ProductIcon name="chevron-down" width={14} height={14} />
      </summary>
      <div className={styles.recordTraceToolBody}>
        {calls.map((call) => (
          <div key={call.id || call.function.name} className={styles.recordTraceToolCall}>
            <strong>{call.function.name || 'Tool'}</strong>
            {call.function.arguments ? (
              <pre>{formatTracePayload(call.function.arguments)}</pre>
            ) : (
              <span>{redacted ? 'Arguments hidden for this role' : 'No arguments captured'}</span>
            )}
          </div>
        ))}
      </div>
    </details>
  )
}

function ToolResult({ message }: { message: InsightsTrajectoryMessage }) {
  return (
    <details className={styles.recordTraceTool}>
      <summary>
        <ProductIcon
          name={message.status === 'failed' ? 'alert' : 'check'}
          width={15}
          height={15}
        />
        <span>{message.tool_name || 'Tool'} result</span>
        <ProductIcon name="chevron-down" width={14} height={14} />
      </summary>
      <div className={styles.recordTraceToolBody}>
        {message.content ? (
          <pre>{formatTracePayload(message.content)}</pre>
        ) : (
          <span>
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
  const positions = new Map<number, number>()
  for (const message of messages) {
    const index = message.turn_index ?? 0
    const existing = positions.get(index)
    if (existing === undefined) {
      positions.set(index, turns.length)
      turns.push({ index, messages: [message] })
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
        messages.push({ role: 'user', content: step.text, turn_index: record.turn_index ?? 0 })
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
          content: step.text,
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
          content_redacted: step.content_redacted,
          turn_index: record.turn_index ?? 0,
        })
        break
      default:
        break
    }
  }
  return messages
}

function formatTracePayload(value: string) {
  try {
    return JSON.stringify(JSON.parse(value), null, 2)
  } catch {
    return value
  }
}
