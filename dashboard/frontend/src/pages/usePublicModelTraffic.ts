import { useEffect, useState } from 'react'

export interface PublicModelTrafficEntry {
  requests: number
  /** p95 of the completed requests' durations over the window. */
  p95Ms?: number
  cost: { amount: number; currency: string } | null
  /** Requests per hour, oldest first, starting at `hours[0]`. */
  hourly: number[]
  hours: string[]
}

export interface PublicModelTraffic {
  phase: 'loading' | 'ready' | 'unavailable'
  byModel: ReadonlyMap<string, PublicModelTrafficEntry>
}

const WINDOW_MS = 24 * 60 * 60 * 1000
const REFRESH_MS = 120_000
// Each public model is one aggregate query; past this many the cards keep their config facts.
const MAX_MODELS = 12

export function formatRequestCount(value: number): string {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`
  if (value >= 10_000) return `${(value / 1000).toFixed(1)}k`
  return Math.round(value).toLocaleString()
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

const finite = (value: unknown) =>
  typeof value === 'number' && Number.isFinite(value) ? value : undefined

/** A public model's last 24 h from a Router Replay aggregate with an hourly traffic series. */
export function parsePublicModelTraffic(body: unknown): PublicModelTrafficEntry | null {
  if (!isRecord(body) || !isRecord(body.traffic) || !Array.isArray(body.traffic.buckets))
    return null
  const buckets = body.traffic.buckets.filter(isRecord)
  const summary = isRecord(body.summary) ? body.summary : {}
  const currency = typeof summary.currency === 'string' ? summary.currency.trim() : ''
  const total = finite(summary.total_spend)
  return {
    requests: finite(body.record_count) ?? 0,
    p95Ms: finite(body.traffic.latency_p95_ms),
    cost: currency && total !== undefined ? { amount: total, currency } : null,
    hourly: buckets.map((bucket) => finite(bucket.requests) ?? 0),
    hours: buckets.map((bucket) => (typeof bucket.start === 'string' ? bucket.start : '')),
  }
}

/** Requests, p95 and cost of each public model over the last 24 hours, from Router Replay. */
export function usePublicModelTraffic(names: string[]): PublicModelTraffic {
  const [traffic, setTraffic] = useState<PublicModelTraffic>({
    phase: 'loading',
    byModel: new Map(),
  })
  const key = names.slice(0, MAX_MODELS).join('\n')
  useEffect(() => {
    const models = key ? key.split('\n') : []
    const controller = new AbortController()
    let timeout: number | undefined
    const load = async () => {
      if (document.visibilityState !== 'hidden' && models.length) {
        const since = new Date(Date.now() - WINDOW_MS).toISOString()
        const answers = await Promise.all(
          models.map(async (model) => {
            try {
              const response = await fetch(
                `/api/router/api/v1/observability/replays/aggregate?${new URLSearchParams({ model, since, interval: '1h' })}`,
                { signal: controller.signal },
              )
              if (!response.ok) return [model, null] as const
              return [model, parsePublicModelTraffic(await response.json())] as const
            } catch {
              return [model, null] as const
            }
          }),
        )
        if (controller.signal.aborted) return
        const byModel = new Map(
          answers.filter(
            (answer): answer is readonly [string, PublicModelTrafficEntry] => answer[1] !== null,
          ),
        )
        setTraffic((current) =>
          byModel.size
            ? { phase: 'ready', byModel }
            : current.phase === 'ready'
              ? current
              : { phase: 'unavailable', byModel },
        )
      } else if (!models.length) {
        setTraffic({ phase: 'unavailable', byModel: new Map() })
      }
      if (!controller.signal.aborted) timeout = window.setTimeout(() => void load(), REFRESH_MS)
    }
    void load()
    return () => {
      controller.abort()
      window.clearTimeout(timeout)
    }
  }, [key])
  return traffic
}
