import { useState } from 'react'
import BenchPagination from './BenchPagination'
import EvidenceValue from './EvidenceValue'
import type { RecordedMessage, RecordedToolCall } from './types'
import styles from './CallDetail.module.css'
import shared from './SrBench.module.css'

function ToolArguments({ value }: { value: unknown }) {
  if (value === undefined) return <p className={shared.muted}>Arguments not recorded.</p>
  if (typeof value !== 'string')
    return <p className={shared.muted}>Invalid recorded tool arguments.</p>
  let parsed: unknown
  try {
    parsed = JSON.parse(value)
  } catch {
    return (
      <>
        <p className={shared.muted}>Arguments (unparsed text)</p>
        <pre className={styles.text}>{value || 'Empty text'}</pre>
      </>
    )
  }
  return <EvidenceValue value={parsed} />
}

export function RecordedToolCalls({ calls }: { calls: RecordedToolCall[] }) {
  const [page, setPage] = useState(0)
  if (!Array.isArray(calls)) return <p className={shared.muted}>Invalid recorded tool calls.</p>
  return (
    <div>
      <ol className={styles.entries} start={page * 10 + 1} aria-label="Recorded tool calls">
        {calls.slice(page * 10, page * 10 + 10).map((call, index) => (
          <li key={page * 10 + index}>
            {call !== null && typeof call === 'object' ? (
              <details>
                <summary className={styles.toolHeading}>
                  {typeof call.function?.name === 'string'
                    ? call.function.name
                    : 'Tool name not recorded'}
                  <span>{typeof call.type === 'string' ? call.type : 'Type not recorded'}</span>
                </summary>
                {typeof call.id === 'string' && (
                  <p className={shared.muted}>Tool call · {call.id}</p>
                )}
                <ToolArguments value={call.function?.arguments} />
              </details>
            ) : (
              <p className={shared.muted}>Invalid recorded tool call.</p>
            )}
          </li>
        ))}
      </ol>
      <BenchPagination
        label="Tool calls"
        total={calls.length}
        page={page}
        pageSize={10}
        onChange={setPage}
      />
    </div>
  )
}

function MessageContent({ content }: { content: RecordedMessage['content'] }) {
  if (typeof content === 'string')
    return content ? (
      <pre className={styles.text}>{content}</pre>
    ) : (
      <p className={shared.muted}>Empty message text.</p>
    )
  if (Array.isArray(content))
    return (
      <>
        {content.map((part, index) =>
          part === null || typeof part !== 'object' || typeof part.type !== 'string' ? (
            <p className={shared.muted} key={index}>
              Invalid recorded content part.
            </p>
          ) : part.type === 'text' && typeof part.text === 'string' ? (
            <pre className={styles.text} key={index}>
              {part.text}
            </pre>
          ) : (
            <p className={shared.muted} key={index}>
              {part.type} content recorded; media is not loaded in this view.
            </p>
          ),
        )}
      </>
    )
  return <p className={shared.muted}>No message text recorded.</p>
}

export default function RecordedConversation({ messages }: { messages: RecordedMessage[] }) {
  const [page, setPage] = useState(0)
  if (!Array.isArray(messages))
    return <p className={shared.muted}>Invalid recorded conversation.</p>
  return (
    <div>
      <p className={shared.muted}>
        {messages.length} messages in this request, in order. This is the {"call's"} input, not a
        complete task trajectory.
      </p>
      <ol className={styles.entries} start={page * 10 + 1} aria-label="Recorded messages">
        {messages.slice(page * 10, page * 10 + 10).map((message, index) => (
          <li key={page * 10 + index}>
            {message !== null && typeof message === 'object' ? (
              <>
                <div className={styles.entryHeading}>
                  <strong>
                    {typeof message.role === 'string' ? message.role : 'Role not recorded'}
                  </strong>
                  <span>Message {page * 10 + index + 1}</span>
                </div>
                {typeof message.name === 'string' && (
                  <p className={shared.muted}>Name · {message.name}</p>
                )}
                {typeof message.tool_call_id === 'string' && (
                  <p className={shared.muted}>Result for tool call · {message.tool_call_id}</p>
                )}
                <MessageContent content={message.content} />
                {!!message.tool_calls?.length && <RecordedToolCalls calls={message.tool_calls} />}
              </>
            ) : (
              <p className={shared.muted}>Invalid recorded message.</p>
            )}
          </li>
        ))}
      </ol>
      <BenchPagination
        label="Messages"
        total={messages.length}
        page={page}
        pageSize={10}
        onChange={setPage}
      />
    </div>
  )
}
