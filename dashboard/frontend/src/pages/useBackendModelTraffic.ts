import { useEffect, useState } from 'react'

/** Why Router Replay has no traffic to show: off, not readable, or the Router is not answering. */
export type BackendTrafficUnavailable = 'replay_off' | 'forbidden' | 'router_unreachable'

export interface BackendModelTraffic {
  phase: 'loading' | 'ready' | 'unavailable'
  reason?: BackendTrafficUnavailable
  /** Requests Router Replay recorded in the window. */
  requests: number
  /** Percent of the window's requests each model answered. */
  shares: ReadonlyMap<string, number>
  /** Whether `shares` names every model that answered; Replay lists the busiest ten. */
  complete: boolean
  spend: { amount: number; currency: string; currencies: number } | null
  savedPercent: number | null
}

const WINDOW_MS = 24 * 60 * 60 * 1000
const REFRESH_MS = 60_000
const UNREACHABLE_REFRESH_MS = 120_000
const MODEL_SELECTION_LIMIT = 10

const INITIAL_TRAFFIC: BackendModelTraffic = {
  phase: 'loading',
  requests: 0,
  shares: new Map(),
  complete: false,
  spend: null,
  savedPercent: null,
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

const finite = (value: unknown) =>
  typeof value === 'number' && Number.isFinite(value) ? value : undefined

export function trafficUnavailableReason(status: number): BackendTrafficUnavailable {
  if (status === 404) return 'replay_off'
  if (status === 401 || status === 403) return 'forbidden'
  return 'router_unreachable'
}

/** The 24 h traffic facts of a Router Replay aggregate, or null when it is not one. */
export function parseBackendTraffic(
  body: unknown,
): Omit<BackendModelTraffic, 'phase' | 'reason'> | null {
  if (!isRecord(body)) return null
  const requests = finite(body.record_count)
  if (requests === undefined || !Array.isArray(body.model_selection)) return null
  const selection = body.model_selection.filter(
    (entry): entry is { name: string; value: number } =>
      isRecord(entry) && typeof entry.name === 'string' && finite(entry.value) !== undefined,
  )
  const shares = new Map(
    requests > 0 ? selection.map((entry) => [entry.name, (entry.value / requests) * 100]) : [],
  )
  const summary = isRecord(body.summary) ? body.summary : {}
  const byCurrency = Array.isArray(summary.by_currency)
    ? summary.by_currency.filter(
        (entry): entry is { currency: string; total_spend: number } =>
          isRecord(entry) &&
          typeof entry.currency === 'string' &&
          finite(entry.total_spend) !== undefined,
      )
    : []
  const currency = typeof summary.currency === 'string' ? summary.currency.trim() : ''
  const total = finite(summary.total_spend)
  let spend: BackendModelTraffic['spend'] = null
  if (currency && total !== undefined) {
    spend = { amount: total, currency, currencies: 1 }
  } else if (byCurrency.length) {
    const [largest] = [...byCurrency].sort((a, b) => b.total_spend - a.total_spend)
    spend = {
      amount: largest.total_spend,
      currency: largest.currency,
      currencies: byCurrency.length,
    }
  } else if (total !== undefined && requests > 0) {
    spend = { amount: total, currency: 'USD', currencies: 1 }
  }
  const baseline = finite(summary.baseline_spend) ?? 0
  const saved = finite(summary.total_saved) ?? 0
  return {
    requests,
    shares,
    complete: selection.length < MODEL_SELECTION_LIMIT,
    spend,
    savedPercent: baseline > 0 && saved > 0 ? (saved / baseline) * 100 : null,
  }
}

/** Requests and spend per backend model over the last 24 hours, from Router Replay. */
export function useBackendModelTraffic(): BackendModelTraffic {
  const [traffic, setTraffic] = useState<BackendModelTraffic>(INITIAL_TRAFFIC)
  useEffect(() => {
    const controller = new AbortController()
    let timeout: number | undefined
    const load = async () => {
      let next = REFRESH_MS
      if (document.visibilityState !== 'hidden') {
        const since = new Date(Date.now() - WINDOW_MS).toISOString()
        try {
          const response = await fetch(
            `/api/router/api/v1/observability/replays/aggregate?${new URLSearchParams({ since })}`,
            { signal: controller.signal },
          )
          if (!response.ok) {
            const reason = trafficUnavailableReason(response.status)
            setTraffic((current) =>
              current.phase === 'ready' && reason === 'router_unreachable'
                ? current
                : { ...INITIAL_TRAFFIC, phase: 'unavailable', reason },
            )
            // Replay stays off and permissions stay as they are until the page reloads.
            if (reason !== 'router_unreachable') return
            next = UNREACHABLE_REFRESH_MS
          } else {
            const parsed = parseBackendTraffic(await response.json())
            setTraffic(
              parsed
                ? { ...parsed, phase: 'ready' }
                : { ...INITIAL_TRAFFIC, phase: 'unavailable', reason: 'router_unreachable' },
            )
          }
        } catch {
          if (controller.signal.aborted) return
          setTraffic((current) =>
            current.phase === 'ready'
              ? current
              : { ...INITIAL_TRAFFIC, phase: 'unavailable', reason: 'router_unreachable' },
          )
          next = UNREACHABLE_REFRESH_MS
        }
      }
      if (!controller.signal.aborted) timeout = window.setTimeout(() => void load(), next)
    }
    void load()
    return () => {
      controller.abort()
      window.clearTimeout(timeout)
    }
  }, [])
  return traffic
}
