import { useEffect, useMemo, useRef, useState } from 'react'

import { projectAgentTimeline, type AgentTimelineTool } from '../utils/agentEventProjection'
import type {
  AgentApprovalRequestPayload,
  AgentEvent,
  AgentLiveModelStepEvent,
  AgentModelStepSummaryEventPayload,
} from '../generated/managementApiContract'
import type { PlaygroundInferenceMessage } from '../utils/playgroundInferenceMessages'
import AgentArtifactResult from './AgentArtifactResult'
import AgentRouterMetadata from './AgentRouterMetadata'
import HeaderDisplay from './HeaderDisplay'
import MarkdownRenderer from './MarkdownRenderer'
import ProductIcon from './ProductIcon'
import type { PlaygroundMode } from './playgroundModes'
import styles from './AgentPlayground.module.css'

interface AgentTimelineProps {
  events: AgentEvent[]
  liveModelSteps: AgentLiveModelStepEvent[]
  inferenceMessages?: PlaygroundInferenceMessage[]
  hasEarlier: boolean
  loading: boolean
  mode: PlaygroundMode
  userName: string
  canLoadArtifactContent: boolean
  canReadRequestLogs: boolean
  onLoadEarlier: () => void
  onReview: (approval: AgentApprovalRequestPayload) => void
}

const RESPONSE_PATHS = new Set<NonNullable<AgentModelStepSummaryEventPayload['responsePath']>>([
  'upstream',
  'cache',
  'fast_response',
  'looper',
  'image_generation',
])

function directRouterMetadata(
  message: PlaygroundInferenceMessage,
): AgentModelStepSummaryEventPayload | null {
  const metadata = message.metadata
  if (!metadata) return null
  const headers = metadata.headers
  const responsePath = headers['x-vsr-response-path']
  const resolvedResponsePath = RESPONSE_PATHS.has(
    responsePath as NonNullable<AgentModelStepSummaryEventPayload['responsePath']>,
  )
    ? (responsePath as NonNullable<AgentModelStepSummaryEventPayload['responsePath']>)
    : undefined
  return {
    latencyMilliseconds: metadata.latencyMilliseconds,
    modelStepId: metadata.responseId || message.id,
    requestId: metadata.requestId || metadata.responseId || 'Unavailable',
    ...(resolvedResponsePath ? { responsePath: resolvedResponsePath } : {}),
    ...(headers['x-vsr-selected-algorithm']
      ? { selectedAlgorithm: headers['x-vsr-selected-algorithm'] }
      : {}),
    ...(headers['x-vsr-selected-decision']
      ? { selectedDecision: headers['x-vsr-selected-decision'] }
      : {}),
    ...(headers['x-vsr-selected-model'] || metadata.model
      ? { selectedModel: headers['x-vsr-selected-model'] || metadata.model }
      : {}),
    ...(headers['x-vsr-selected-recipe']
      ? { selectedRecipe: headers['x-vsr-selected-recipe'] }
      : {}),
    ...(metadata.ttftMilliseconds !== undefined
      ? { ttftMilliseconds: metadata.ttftMilliseconds }
      : {}),
    ...(metadata.usage
      ? {
          usage: {
            inputTokens: metadata.usage.promptTokens,
            outputTokens: metadata.usage.completionTokens,
            totalTokens: metadata.usage.totalTokens,
          },
        }
      : {}),
  }
}

function readableToolName(value: string): string {
  const segments = value.split('.')
  const name = segments[segments.length - 1] || value
  return name.replace(/[_-]+/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase())
}

function ToolRow({
  tool,
  canLoadArtifactContent,
}: {
  tool: AgentTimelineTool
  canLoadArtifactContent: boolean
}) {
  const [expanded, setExpanded] = useState(false)
  const active = tool.status === 'running'
  return (
    <div className={`${styles.toolRow} ${active ? styles.toolRowActive : ''}`}>
      <button
        type="button"
        className={styles.toolRowButton}
        onClick={() => setExpanded((current) => !current)}
        aria-expanded={expanded}
      >
        <span className={styles.toolStatusIcon} aria-hidden="true">
          {active ? (
            <span className={styles.toolPulse} />
          ) : (
            <ProductIcon name={tool.status === 'completed' ? 'check' : 'alert'} />
          )}
        </span>
        <span className={styles.toolCopy}>
          <strong>{readableToolName(tool.name)}</strong>
          <small>
            {active
              ? tool.summary || 'Working…'
              : tool.status === 'completed'
                ? tool.summary || 'Done'
                : tool.error || tool.status}
          </small>
        </span>
        <span className={styles.toolStatus}>
          {active ? 'Running' : tool.status === 'completed' ? 'Done' : tool.status}
        </span>
        <ProductIcon name={expanded ? 'chevron-up' : 'chevron-down'} />
      </button>
      {expanded ? (
        <div className={styles.toolDetails}>
          <div>
            <span>Action</span>
            <strong>{tool.classification}</strong>
          </div>
          {tool.summary ? <p>{tool.summary}</p> : null}
          {tool.error ? <p className={styles.toolError}>{tool.error}</p> : null}
          {tool.artifactId ? (
            <AgentArtifactResult
              artifactId={tool.artifactId}
              canLoadOriginal={canLoadArtifactContent}
            />
          ) : null}
        </div>
      ) : null}
    </div>
  )
}

export default function AgentTimeline({
  events,
  liveModelSteps,
  inferenceMessages = [],
  hasEarlier,
  loading,
  mode,
  userName,
  canLoadArtifactContent,
  canReadRequestLogs,
  onLoadEarlier,
  onReview,
}: AgentTimelineProps) {
  const items = useMemo(
    () => projectAgentTimeline(events, liveModelSteps),
    [events, liveModelSteps],
  )
  const viewportRef = useRef<HTMLDivElement>(null)
  const lastSequence = events[events.length - 1]?.sequence

  useEffect(() => {
    const viewport = viewportRef.current
    if (!viewport) return
    const nearBottom = viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight < 180
    if (nearBottom || items.length <= 2) {
      requestAnimationFrame(() =>
        viewport.scrollTo({
          top: viewport.scrollHeight,
          behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches
            ? 'auto'
            : 'smooth',
        }),
      )
    }
  }, [inferenceMessages, items, lastSequence])

  return (
    <div ref={viewportRef} className={styles.timelineViewport} data-testid="agent-timeline">
      <div className={styles.timeline} aria-live="polite" aria-busy={loading}>
        {loading && items.length === 0 ? (
          <div className={styles.progressRow} role="status">
            <span className={styles.progressDot} aria-hidden="true" />
            <span>Loading conversation…</span>
          </div>
        ) : null}
        {hasEarlier ? (
          <button
            type="button"
            className={styles.loadEarlier}
            onClick={onLoadEarlier}
            disabled={loading}
          >
            {loading ? 'Loading…' : 'Load earlier messages'}
          </button>
        ) : null}
        {!loading && items.length === 0 && inferenceMessages.length === 0 ? (
          <div className={styles.emptyConversation}>
            <div className={styles.emptyMark}>
              <img src="/vllm.png" alt="" />
            </div>
            <h1>Welcome, {userName}</h1>
            <p>
              {mode === 'builder'
                ? 'Describe the outcome. We’ll compose the model path.'
                : mode === 'agent'
                  ? 'Search the web. Use the right tools.'
                  : 'One prompt. The right model path.'}
            </p>
          </div>
        ) : null}
        {items.map((item) => {
          if (item.kind === 'message') {
            return (
              <article
                key={item.id}
                className={`${styles.message} ${item.role === 'user' ? styles.userMessage : styles.assistantMessage}`}
                data-testid={`agent-message-${item.role}`}
              >
                <div className={styles.messageBody}>
                  {item.role === 'assistant' && item.text ? (
                    <MarkdownRenderer content={item.text} />
                  ) : (
                    item.text
                  )}
                  {item.streaming ? (
                    <span className={styles.streamCursor} aria-label="Generating" />
                  ) : null}
                  {item.role === 'assistant' && item.metadata ? (
                    <AgentRouterMetadata
                      metadata={item.metadata}
                      canReadRequestLogs={canReadRequestLogs}
                    />
                  ) : null}
                </div>
              </article>
            )
          }
          if (item.kind === 'tool') {
            return (
              <ToolRow key={item.id} tool={item} canLoadArtifactContent={canLoadArtifactContent} />
            )
          }
          if (item.kind === 'progress') {
            return (
              <div key={item.id} className={styles.progressRow} role="status">
                <span className={styles.progressDot} aria-hidden="true" />
                <span>{item.message}</span>
              </div>
            )
          }
          if (item.kind === 'approval') {
            return (
              <section key={item.id} className={styles.reviewCard}>
                <div className={styles.reviewIcon}>
                  <ProductIcon name="mixture" />
                </div>
                <div>
                  <strong>
                    {item.status === 'waiting' ? 'Ready for review' : 'Publication reviewed'}
                  </strong>
                  <p>
                    {item.payload.summary.entrypointName ||
                      item.payload.summary.recipeName ||
                      'Your Mixture-of-Models is ready.'}
                  </p>
                </div>
                {item.status === 'waiting' ? (
                  <button type="button" onClick={() => onReview(item.payload)}>
                    Review
                    <ProductIcon name="arrow-right" />
                  </button>
                ) : (
                  <span className={styles.reviewStatus}>{item.status}</span>
                )}
              </section>
            )
          }
          if (item.kind === 'terminal' && item.payload.status === 'failed') {
            return (
              <div key={item.id} className={styles.turnError} role="alert">
                <ProductIcon name="alert" />
                <span>{item.payload.error?.message || 'This turn could not be completed.'}</span>
              </div>
            )
          }
          return null
        })}
        {inferenceMessages.map((message) => {
          if (message.status === 'failed' && !message.content) return null
          const responseMetadata = directRouterMetadata(message)
          return (
            <article
              key={message.id}
              className={`${styles.message} ${message.role === 'user' ? styles.userMessage : styles.assistantMessage}`}
              data-testid={`agent-message-${message.role}`}
            >
              <div className={styles.messageBody}>
                {message.role === 'assistant' && message.content ? (
                  <MarkdownRenderer content={message.content} />
                ) : (
                  message.content
                )}
                {message.status === 'streaming' ? (
                  <span className={styles.streamCursor} aria-label="Generating" />
                ) : null}
                {message.role === 'assistant' && message.metadata?.headers ? (
                  <HeaderDisplay headers={message.metadata.headers} />
                ) : null}
                {message.role === 'assistant' && responseMetadata ? (
                  <AgentRouterMetadata
                    metadata={responseMetadata}
                    canReadRequestLogs={
                      canReadRequestLogs &&
                      Boolean(message.metadata?.requestId || message.metadata?.responseId)
                    }
                  />
                ) : null}
              </div>
            </article>
          )
        })}
      </div>
    </div>
  )
}
