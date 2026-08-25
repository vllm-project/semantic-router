import type { AgentModelStepSummaryEventPayload } from '../generated/managementApiContract'
import ProductIcon from './ProductIcon'
import styles from './AgentPlayground.module.css'

interface AgentRouterMetadataProps {
  metadata: AgentModelStepSummaryEventPayload
  canReadRequestLogs: boolean
}

function formatDuration(milliseconds: number): string {
  if (milliseconds < 1000) return `${milliseconds} ms`
  const seconds = milliseconds / 1000
  return `${seconds >= 10 ? Math.round(seconds) : seconds.toFixed(1)} s`
}

function formatTokens(value: number): string {
  return new Intl.NumberFormat('en-US').format(value)
}

function readableValue(value: string): string {
  return value.replace(/[_-]+/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase())
}

export default function AgentRouterMetadata({
  metadata,
  canReadRequestLogs,
}: AgentRouterMetadataProps) {
  const routeLabel =
    metadata.selectedModel ||
    readableValue(metadata.selectedDecision || metadata.selectedRecipe || 'Router')
  const usageBreakdown: Array<[string, number]> = []
  const addUsageBreakdown = (label: string, value: number | undefined) => {
    if (value !== undefined) usageBreakdown.push([label, value])
  }
  addUsageBreakdown('Uncached input', metadata.usage?.inputUncachedTokens)
  addUsageBreakdown('Cache read', metadata.usage?.inputCacheReadTokens)
  addUsageBreakdown('Cache write', metadata.usage?.inputCacheWriteTokens)
  addUsageBreakdown('Reasoning', metadata.usage?.outputReasoningTokens)
  addUsageBreakdown('Other output', metadata.usage?.outputOtherTokens)

  return (
    <details className={styles.routerMetadata} data-testid="agent-router-metadata">
      <summary>
        <span className={styles.routerMetadataIcon} aria-hidden="true">
          <ProductIcon name="topology" />
        </span>
        <span className={styles.routerMetadataSummary}>
          <strong>{routeLabel}</strong>
          {metadata.selectedDecision && metadata.selectedDecision !== routeLabel ? (
            <span>{readableValue(metadata.selectedDecision)}</span>
          ) : null}
        </span>
        <span className={styles.routerMetadataLatency}>
          {formatDuration(metadata.latencyMilliseconds)}
        </span>
        <ProductIcon
          name="chevron-down"
          className={styles.routerMetadataChevron}
          aria-hidden="true"
        />
      </summary>
      <div className={styles.routerMetadataDetails}>
        <dl>
          {metadata.selectedRecipe ? (
            <div>
              <dt>Recipe</dt>
              <dd>{readableValue(metadata.selectedRecipe)}</dd>
            </div>
          ) : null}
          {metadata.selectedDecision ? (
            <div>
              <dt>Decision</dt>
              <dd>{readableValue(metadata.selectedDecision)}</dd>
            </div>
          ) : null}
          {metadata.selectedModel ? (
            <div>
              <dt>Model</dt>
              <dd>{metadata.selectedModel}</dd>
            </div>
          ) : null}
          {metadata.selectedAlgorithm ? (
            <div>
              <dt>Algorithm</dt>
              <dd>{readableValue(metadata.selectedAlgorithm)}</dd>
            </div>
          ) : null}
          {metadata.responsePath ? (
            <div>
              <dt>Response path</dt>
              <dd>{readableValue(metadata.responsePath)}</dd>
            </div>
          ) : null}
          <div>
            <dt>Latency</dt>
            <dd>{formatDuration(metadata.latencyMilliseconds)}</dd>
          </div>
          {metadata.ttftMilliseconds !== undefined ? (
            <div>
              <dt>First token</dt>
              <dd>{formatDuration(metadata.ttftMilliseconds)}</dd>
            </div>
          ) : null}
          {metadata.usage ? (
            <div>
              <dt>Tokens</dt>
              <dd>
                {formatTokens(metadata.usage.totalTokens)} total ·{' '}
                {formatTokens(metadata.usage.inputTokens)} in ·{' '}
                {formatTokens(metadata.usage.outputTokens)} out
              </dd>
            </div>
          ) : null}
          {usageBreakdown.map(([label, value]) => (
            <div key={label}>
              <dt>{label}</dt>
              <dd>{formatTokens(value)}</dd>
            </div>
          ))}
          <div>
            <dt>Request ID</dt>
            <dd className={styles.routerMetadataRequestId}>{metadata.requestId}</dd>
          </div>
        </dl>
        {canReadRequestLogs ? (
          <a href={`/logs?q=${encodeURIComponent(metadata.requestId)}`}>
            Open request log
            <ProductIcon name="arrow-right" />
          </a>
        ) : null}
      </div>
    </details>
  )
}
