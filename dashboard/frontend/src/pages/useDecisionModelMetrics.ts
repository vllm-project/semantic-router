import { useEffect, useState } from 'react'
import {
  DECISION_MODEL_METRICS,
  decisionModelMetricQueries,
  readDecisionModelMetricVector,
  type DecisionModelMetricValues,
} from './decisionModelMetrics'

export function useDecisionModelMetrics(
  deployments: string[],
  refreshedAt: Date | null,
  enabled: boolean,
) {
  const [values, setValues] = useState<DecisionModelMetricValues>({})
  const [unavailable, setUnavailable] = useState<string[]>([])
  const [loading, setLoading] = useState(false)
  const deploymentKey = JSON.stringify([...deployments].sort())

  useEffect(() => {
    const names = JSON.parse(deploymentKey) as string[]
    if (!enabled || !names.length) {
      setValues({})
      setUnavailable([])
      setLoading(false)
      return
    }
    const reads = new AbortController()
    let mounted = true
    const timeout = window.setTimeout(() => reads.abort(), 8_000)
    setLoading(true)
    const queries = decisionModelMetricQueries(names)
    void Promise.allSettled(
      DECISION_MODEL_METRICS.map(async ({ key }) => {
        const params = new URLSearchParams({ query: queries[key] })
        const response = await fetch(`/embedded/prometheus/api/v1/query?${params}`, {
          headers: { Accept: 'application/json' },
          signal: reads.signal,
        })
        if (!response.ok) throw new Error('Model metrics are unavailable.')
        return readDecisionModelMetricVector(await response.json(), names)
      }),
    )
      .then((results) => {
        if (!mounted) return
        const next: DecisionModelMetricValues = {}
        const failed: string[] = []
        results.forEach((result, index) => {
          const metric = DECISION_MODEL_METRICS[index]
          if (result.status === 'fulfilled') next[metric.key] = result.value
          else failed.push(metric.label)
        })
        setValues(next)
        setUnavailable(failed)
        setLoading(false)
      })
      .finally(() => window.clearTimeout(timeout))
    return () => {
      mounted = false
      reads.abort()
      window.clearTimeout(timeout)
    }
  }, [deploymentKey, refreshedAt, enabled])

  return { values, unavailable, loading }
}
