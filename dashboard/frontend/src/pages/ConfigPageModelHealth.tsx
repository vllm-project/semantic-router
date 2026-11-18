import { Skeleton, Sparkline, StatusLabel, Tooltip } from '../ui'
import type { ModelHealth, ModelHealthIssue } from '../utils/modelHealthApi'
import {
  checkedAgo,
  formatLatency,
  MODEL_HEALTH_META,
  modelHealthIssueText,
  modelHealthStatus,
  visibleP95,
} from './configPageModelHealthPresentation'
import type { ModelHealthPhase } from './useBackendModelHealth'

function HealthLabel({ health, checking }: { health?: ModelHealth; checking: boolean }) {
  const meta = MODEL_HEALTH_META[modelHealthStatus(health)]
  return checking ? (
    <StatusLabel tone="info" pulse>
      Checking…
    </StatusLabel>
  ) : (
    <StatusLabel tone={meta.tone} pulse={health?.status === 'degraded'}>
      {meta.label}
    </StatusLabel>
  )
}

/** The Health cell: a status, and the p95 answer time under it while the model answers. */
export function ModelHealthCell({
  health,
  phase,
  issue,
  checking,
}: {
  health?: ModelHealth
  phase: ModelHealthPhase
  /** Why the last check failed, as the health hook reports it. */
  issue?: ModelHealthIssue
  checking: boolean
}) {
  if (phase === 'loading' && !health) {
    return (
      <span className="flex flex-col gap-1.5" aria-label="Checking health">
        <Skeleton className="h-3.5 w-20" />
        <Skeleton className="ml-3.5 h-2.5 w-14" />
      </span>
    )
  }
  const p95 = visibleP95(health)
  const reason = health ? health.reason : modelHealthIssueText(phase, issue)
  const ago = checkedAgo(health?.checkedAt)
  return (
    <Tooltip
      side="left"
      content={
        <span className="flex max-w-72 flex-col gap-0.5 py-0.5 text-left leading-snug">
          <span>{reason ?? MODEL_HEALTH_META[modelHealthStatus(health)].label}</span>
          {health?.latencyMs !== undefined && p95 !== undefined ? (
            <span className="opacity-70">
              Latest {formatLatency(health.latencyMs)} · p95 of {health.recentMs.length}{' '}
              {health.recentMs.length === 1 ? 'check' : 'checks'}
            </span>
          ) : null}
          {ago ? <span className="opacity-70">Checked {ago}</span> : null}
        </span>
      }
    >
      <span className="inline-flex flex-col gap-0.5">
        <HealthLabel health={health} checking={checking} />
        {p95 !== undefined ? (
          <span className="pl-3.5 font-mono text-[11px] tabular text-subtle-foreground">
            p95 {formatLatency(p95)}
          </span>
        ) : null}
      </span>
    </Tooltip>
  )
}

/** The health block of the model sheet: why, how fast, and how recent. */
export function ModelHealthSummary({
  health,
  phase,
  issue,
  checking,
}: {
  health?: ModelHealth
  phase: ModelHealthPhase
  /** Why the last check failed, as the health hook reports it. */
  issue?: ModelHealthIssue
  checking: boolean
}) {
  if (phase === 'loading' && !health) {
    return <Skeleton className="h-[92px] w-full rounded-lg" />
  }
  const status = modelHealthStatus(health)
  const p95 = visibleP95(health)
  const issueText = modelHealthIssueText(phase, issue)
  const ago = checkedAgo(health?.checkedAt)
  const samples = health?.recentMs ?? []
  return (
    <section
      aria-label="Health"
      className="flex flex-col gap-2 rounded-lg border border-border px-3.5 pb-1 pt-3"
    >
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <HealthLabel health={health} checking={checking} />
        {p95 !== undefined ? (
          <span className="font-mono text-xs tabular text-muted-foreground">
            p95 {formatLatency(p95)}
            {health?.latencyMs !== undefined ? ` · latest ${formatLatency(health.latencyMs)}` : ''}
          </span>
        ) : null}
        <span className="ml-auto text-xs text-subtle-foreground">
          {ago
            ? `Checked ${ago}`
            : phase === 'unavailable'
              ? 'Not available'
              : phase === 'retrying'
                ? 'Retrying'
                : 'Not checked yet'}
        </span>
      </div>
      {health?.reason ? (
        <p className="text-xs leading-relaxed text-muted-foreground">{health.reason}</p>
      ) : status === 'healthy' && !issueText ? (
        <p className="text-xs text-muted-foreground">
          Every backend answered its last health check.
        </p>
      ) : null}
      {issueText ? (
        <p className="text-xs leading-relaxed text-muted-foreground">{issueText}</p>
      ) : null}
      {samples.length > 1 && p95 !== undefined ? (
        <div className="flex flex-col">
          <div className="flex items-center justify-between text-[11px] text-subtle-foreground">
            <span>Answer time</span>
            <span className="font-mono">last {samples.length} checks</span>
          </div>
          <Sparkline
            data={samples}
            color={status === 'degraded' ? 'var(--warning)' : 'var(--chart-6)'}
            height={48}
          />
        </div>
      ) : (
        <span className="pb-2" />
      )}
    </section>
  )
}
