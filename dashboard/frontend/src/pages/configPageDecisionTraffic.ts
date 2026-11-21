import { useEffect, useState } from 'react'

interface AggregateValue {
  name: string
  value: number
}

/**
 * Requests Router Replay recorded per decision of one recipe, or null while
 * replay is off or the Router does not answer.
 */
export function useDecisionTraffic(recipe: string | undefined): ReadonlyMap<string, number> | null {
  const [traffic, setTraffic] = useState<ReadonlyMap<string, number> | null>(null)
  useEffect(() => {
    if (!recipe) return
    const controller = new AbortController()
    setTraffic(null)
    fetch(`/api/router/api/v1/observability/replays/aggregate?${new URLSearchParams({ recipe })}`, {
      signal: controller.signal,
    })
      .then(async (response) => {
        if (!response.ok) return null
        const body = (await response.json()) as {
          record_count?: number
          decision_distribution?: AggregateValue[]
        }
        if (!Array.isArray(body.decision_distribution) || !body.record_count) return null
        return new Map(body.decision_distribution.map((entry) => [entry.name, entry.value]))
      })
      .then((next) => {
        if (!controller.signal.aborted) setTraffic(next)
      })
      .catch(() => {
        if (!controller.signal.aborted) setTraffic(null)
      })
    return () => controller.abort()
  }, [recipe])
  return traffic
}
