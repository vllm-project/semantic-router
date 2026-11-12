import { createDashboardObservation, type ObservationOptions } from './dashboardObservation'
import { withRequestTimeout } from './boundedRequest'

type Resource = ReturnType<typeof createDashboardObservation<unknown>>
const observations = new Map<string, Resource>()

export function dashboardObservation<T>(
  path: string,
  options: ObservationOptions & { timeoutMs?: number } = {},
) {
  let resource = observations.get(path)
  if (!resource) {
    resource = createDashboardObservation<unknown>(
      (signal) =>
        withRequestTimeout(
          async (requestSignal) => {
            const response = await fetch(path, {
              headers: { Accept: 'application/json' },
              signal: requestSignal,
            })
            if (!response.ok) throw new Error(`Observation unavailable (HTTP ${response.status}).`)
            return response.json()
          },
          signal,
          options.timeoutMs ?? 10_000,
        ),
      { ...options, isHidden: () => typeof document !== 'undefined' && document.hidden },
    )
    observations.set(path, resource)
  }
  return resource as ReturnType<typeof createDashboardObservation<T>>
}

export function invalidateDashboardObservations(paths?: readonly string[]) {
  observations.forEach((resource, path) => {
    if (!paths || paths.includes(path)) resource.invalidate()
  })
}

export function clearDashboardObservations() {
  observations.forEach((resource) => resource.clear())
}
