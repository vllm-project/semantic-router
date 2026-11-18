import type { StatusTone } from '../ui'
import type { ModelHealth, ModelHealthStatus } from '../utils/modelHealthApi'

export const MODEL_HEALTH_META: Record<
  ModelHealthStatus,
  { tone: StatusTone; label: string; rank: number }
> = {
  healthy: { tone: 'success', label: 'Healthy', rank: 0 },
  degraded: { tone: 'warning', label: 'Degraded', rank: 1 },
  down: { tone: 'danger', label: 'Down', rank: 2 },
  unchecked: { tone: 'neutral', label: 'Not checked', rank: 3 },
}

export const MODEL_HEALTH_ORDER: ModelHealthStatus[] = ['healthy', 'degraded', 'down', 'unchecked']

export function formatLatency(milliseconds: number): string {
  return milliseconds >= 1000
    ? `${(milliseconds / 1000).toFixed(2)}s`
    : `${Math.round(milliseconds)}ms`
}

/** The status a model shows: its latest check, or not checked before one exists. */
export function modelHealthStatus(health: ModelHealth | undefined): ModelHealthStatus {
  return health?.status ?? 'unchecked'
}

/** The p95 a model shows: only while it answers, since a failed check has no answer time. */
export function visibleP95(health: ModelHealth | undefined): number | undefined {
  return health && (health.status === 'healthy' || health.status === 'degraded')
    ? health.p95Ms
    : undefined
}

export function healthSortValue(health: ModelHealth | undefined): number {
  return (
    MODEL_HEALTH_META[modelHealthStatus(health)].rank * 1_000_000 +
    Math.min(visibleP95(health) ?? 999_999, 999_999)
  )
}

export function median(values: number[]): number | undefined {
  if (values.length === 0) return undefined
  const sorted = [...values].sort((a, b) => a - b)
  const middle = Math.floor(sorted.length / 2)
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2
}

export type ModelHealthTotals = Record<ModelHealthStatus, number>

export function modelHealthTotals(
  models: string[],
  byModel: ReadonlyMap<string, ModelHealth>,
): ModelHealthTotals {
  const totals: ModelHealthTotals = { healthy: 0, degraded: 0, down: 0, unchecked: 0 }
  for (const model of models) totals[modelHealthStatus(byModel.get(model))] += 1
  return totals
}

/** "1 degraded · 2 not checked", or a plain line when every model answers. */
export function modelHealthHint(totals: ModelHealthTotals): string {
  const parts = [
    totals.degraded ? `${totals.degraded} degraded` : '',
    totals.down ? `${totals.down} down` : '',
    totals.unchecked ? `${totals.unchecked} not checked` : '',
  ].filter(Boolean)
  return parts.length ? parts.join(' · ') : 'Every model answers'
}

/** "just now", "12 s ago", "4 min ago": how old a check is. */
export function checkedAgo(checkedAt: string | undefined, now = Date.now()): string | null {
  const time = checkedAt ? Date.parse(checkedAt) : Number.NaN
  if (Number.isNaN(time)) return null
  const seconds = Math.max(0, Math.round((now - time) / 1000))
  if (seconds < 5) return 'just now'
  if (seconds < 60) return `${seconds} s ago`
  const minutes = Math.round(seconds / 60)
  return minutes < 60 ? `${minutes} min ago` : new Date(time).toLocaleTimeString()
}
