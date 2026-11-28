import { useEffect, useState } from 'react'
import type { SystemStatus } from '../utils/routerRuntime'
import { fetchDashboardJson } from './dashboardPageRequests'

/**
 * Where the Router stands after setup activation, read from `/api/status`.
 *
 * Activation only saves the config and records it for `vllm-sr serve`; the
 * Router comes up later, or not at all when serve has exited. The wizard
 * claims the router is live only once the status reports routing access.
 */
export type SetupRouterReadinessState = 'checking' | 'starting' | 'ready' | 'stopped'

export interface SetupRouterReadiness {
  state: SetupRouterReadinessState
  /** What the status reports about a Router that is not ready yet. */
  detail?: string
}

export const SETUP_READINESS_POLL_MS = 2_000

const STOPPED_SERVICES = new Set(['Router', 'Envoy'])

export function describeSetupRouterReadiness(status: SystemStatus | null): SetupRouterReadiness {
  if (!status) return { state: 'checking' }
  const services = status.services ?? []
  const routingAccess = services.find((service) => service.component === 'gateway')
  if (routingAccess ? routingAccess.healthy : status.overall === 'healthy') {
    return { state: 'ready' }
  }

  const stopped = services.find(
    (service) => STOPPED_SERVICES.has(service.name) && service.status === 'not running',
  )
  if (stopped) {
    return { state: 'stopped', detail: stopped.message || `${stopped.name} is not running` }
  }

  const router = services.find((service) => service.name === 'Router')
  return { state: 'starting', detail: status.router_runtime?.message || router?.message }
}

/**
 * Polls `/api/status` while enabled and stops once the Router is ready. A
 * stopped Router keeps being polled, so a fixed `vllm-sr serve` still turns
 * the page live.
 */
export function useSetupRouterReadiness(
  enabled: boolean,
  fetcher: typeof fetch = fetch,
): SetupRouterReadiness {
  const [readiness, setReadiness] = useState<SetupRouterReadiness>({ state: 'checking' })

  useEffect(() => {
    if (!enabled) return
    let cancelled = false
    let timer: number | undefined
    const poll = async () => {
      let next: SetupRouterReadiness
      try {
        next = describeSetupRouterReadiness(
          await fetchDashboardJson<SystemStatus>('/api/status', 'System status', fetcher),
        )
      } catch {
        next = { state: 'checking' }
      }
      if (cancelled) return
      setReadiness(next)
      if (next.state !== 'ready')
        timer = window.setTimeout(() => void poll(), SETUP_READINESS_POLL_MS)
    }
    void poll()
    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [enabled, fetcher])

  return readiness
}
