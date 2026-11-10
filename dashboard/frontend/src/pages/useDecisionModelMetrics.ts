import { useEffect, useState } from 'react'
import {
  DECISION_MODEL_METRICS,
  decisionModelMetricQueries,
  decisionModelMetricRange,
  readDecisionModelMetricMatrix,
  type DecisionModelMetricSeries,
  type DecisionModelTimeWindow,
} from './decisionModelMetrics'

export function useDecisionModelMetrics(
  deployments: string[],
  refreshedAt: Date | null,
  enabled: boolean,
  timeWindow: DecisionModelTimeWindow,
) {
  const [series, setSeries] = useState<DecisionModelMetricSeries>({})
  const [unavailable, setUnavailable] = useState<string[]>([])
  const [loading, setLoading] = useState(false)
  const [updatedAt, setUpdatedAt] = useState<Date | null>(null)
  const deploymentKey = JSON.stringify([...deployments].sort())

  useEffect(() => {
    // A new deployment or window must not temporarily show a previous one.
    setSeries({})
    setUpdatedAt(null)
  }, [deploymentKey, timeWindow, enabled])

  useEffect(() => {
    const names = JSON.parse(deploymentKey) as string[]
    if (!enabled || !names.length) {
      setSeries({})
      setUnavailable([])
      setLoading(false)
      setUpdatedAt(null)
      return
    }
    // The parent performs visibility-aware polling and refreshes on resume.
    if (document.visibilityState === 'hidden') return
    const reads = new AbortController()
    let mounted = true
    const timeout = window.setTimeout(() => reads.abort(), 8_000)
    setLoading(true)
    const queries = decisionModelMetricQueries(names)
    const range = decisionModelMetricRange(timeWindow)
    void Promise.allSettled(
      DECISION_MODEL_METRICS.map(async ({ key }) => {
        const params = new URLSearchParams({
          query: queries[key],
          start: String(range.start),
          end: String(range.end),
          step: String(range.step),
        })
        const response = await fetch(`/embedded/prometheus/api/v1/query_range?${params}`, {
          headers: { Accept: 'application/json' },
          signal: reads.signal,
        })
        if (!response.ok) throw new Error('Model metric history is unavailable.')
        return readDecisionModelMetricMatrix(await response.json(), names, range)
      }),
    )
      .then((results) => {
        if (!mounted) return
        const next: DecisionModelMetricSeries = {}
        const failed: string[] = []
        results.forEach((result, index) => {
          const metric = DECISION_MODEL_METRICS[index]
          if (result.status === 'fulfilled') next[metric.key] = result.value
          else failed.push(metric.label)
        })
        // Failed observations clear their history; old success is not evidence.
        setSeries(next)
        setUnavailable(failed)
        setUpdatedAt(failed.length < DECISION_MODEL_METRICS.length ? new Date() : null)
        setLoading(false)
      })
      .finally(() => window.clearTimeout(timeout))
    return () => {
      mounted = false
      reads.abort()
      window.clearTimeout(timeout)
    }
  }, [deploymentKey, refreshedAt, enabled, timeWindow])

  return { series, unavailable, loading, updatedAt }
}
