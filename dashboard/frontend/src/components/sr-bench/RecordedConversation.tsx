import { useState } from 'react'
import BenchPagination from './BenchPagination'
import EvidenceValue from './EvidenceValue'
import type { RecordedMessage, RecordedToolCall } from './types'
import { Disclosure } from './benchUi'
import { benchText, evidenceTextClass } from './benchStyles'

const entriesClass = 'my-3 grid min-w-0 list-none gap-3 p-0'
const entryClass = 'min-w-0 break-words rounded-lg border border-border bg-surface px-3.5 py-3'

function ToolArguments({ value }: { value: unknown }) {
  if (value === undefined) return <p className={benchText.muted}>Arguments not recorded.</p>
  if (typeof value !== 'string')
    return <p className={benchText.muted}>Invalid recorded tool arguments.</p>
  let parsed: unknown
  try {
    parsed = JSON.parse(value)
  } catch {
    return (
      <>
        <p className={benchText.muted}>Arguments (unparsed text)</p>
        <pre className={evidenceTextClass}>{value || 'Empty text'}</pre>
      </>
    )
  }
  return <EvidenceValue value={parsed} />
}

export function RecordedToolCalls({ calls }: { calls: RecordedToolCall[] }) {
  const [page, setPage] = useState(0)
  if (!Array.isArray(calls)) return <p className={benchText.muted}>Invalid recorded tool calls.</p>
  return (
    <div className="min-w-0">
      <ol className={entriesClass} start={page * 10 + 1} aria-label="Recorded tool calls">
        {calls.slice(page * 10, page * 10 + 10).map((call, index) => (
          <li key={page * 10 + index} className={entryClass}>
            {call !== null && typeof call === 'object' ? (
              <Disclosure
                bare
                className="my-0"
                summary={
                  <>
                    <span className="font-mono text-[12.5px] text-foreground">
                      {typeof call.function?.name === 'string'
                        ? call.function.name
                        : 'Tool name not recorded'}
                    </span>
                    <span className="ml-2 text-xs font-normal text-muted-foreground">
                      {typeof call.type === 'string' ? call.type : 'Type not recorded'}
                    </span>
                  </>
                }
              >
                {typeof call.id === 'string' && (
                  <p className={`${benchText.muted} mb-2`}>Tool call · {call.id}</p>
                )}
                <ToolArguments value={call.function?.arguments} />
              </Disclosure>
            ) : (
              <p className={benchText.muted}>Invalid recorded tool call.</p>
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
      <pre className={evidenceTextClass}>{content}</pre>
    ) : (
      <p className={benchText.muted}>Empty message text.</p>
    )
  if (Array.isArray(content))
    return (
      <>
        {content.map((part, index) =>
          part === null || typeof part !== 'object' || typeof part.type !== 'string' ? (
            <p className={benchText.muted} key={index}>
              Invalid recorded content part.
            </p>
          ) : part.type === 'text' && typeof part.text === 'string' ? (
            <pre className={evidenceTextClass} key={index}>
              {part.text}
            </pre>
          ) : (
            <p className={benchText.muted} key={index}>
              {part.type} content recorded; media is not loaded in this view.
            </p>
          ),
        )}
      </>
    )
  return <p className={benchText.muted}>No message text recorded.</p>
}

export default function RecordedConversation({ messages }: { messages: RecordedMessage[] }) {
  const [page, setPage] = useState(0)
  if (!Array.isArray(messages))
    return <p className={benchText.muted}>Invalid recorded conversation.</p>
  return (
    <div className="min-w-0">
      <p className={benchText.muted}>
        {messages.length} messages in this request, in order. This is the {"call's"} input, not a
        complete task trajectory.
      </p>
      <ol className={entriesClass} start={page * 10 + 1} aria-label="Recorded messages">
        {messages.slice(page * 10, page * 10 + 10).map((message, index) => (
          <li key={page * 10 + index} className={entryClass}>
            {message !== null && typeof message === 'object' ? (
              <>
                <div className="mb-1.5 flex flex-wrap items-baseline justify-between gap-2 text-[13px]">
                  <strong className="font-medium capitalize text-foreground">
                    {typeof message.role === 'string' ? message.role : 'Role not recorded'}
                  </strong>
                  <span className="text-xs text-subtle-foreground">
                    Message {page * 10 + index + 1}
                  </span>
                </div>
                {typeof message.name === 'string' && (
                  <p className={benchText.muted}>Name · {message.name}</p>
                )}
                {typeof message.tool_call_id === 'string' && (
                  <p className={benchText.muted}>Result for tool call · {message.tool_call_id}</p>
                )}
                <MessageContent content={message.content} />
                {!!message.tool_calls?.length && <RecordedToolCalls calls={message.tool_calls} />}
              </>
            ) : (
              <p className={benchText.muted}>Invalid recorded message.</p>
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
