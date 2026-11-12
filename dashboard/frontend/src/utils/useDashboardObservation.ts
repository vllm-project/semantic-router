import { useCallback, useEffect, useMemo, useSyncExternalStore } from 'react'
import type { ObservationOptions } from './dashboardObservation'
import { dashboardObservation } from './dashboardObservationCache'

const disabled = {
  data: null,
  error: null,
  updatedAt: null,
  loading: false,
  refreshing: false,
  stale: false,
}
const noSubscribe = () => () => {}

export function useDashboardObservation<T>(
  path: string,
  options: ObservationOptions & { timeoutMs?: number; enabled?: boolean } = {},
) {
  const { freshMs, pollMs, timeoutMs, enabled = true } = options
  const resource = useMemo(
    () => dashboardObservation<T>(path, { freshMs, pollMs, timeoutMs }),
    [path, freshMs, pollMs, timeoutMs],
  )
  const observation = useSyncExternalStore(
    enabled ? resource.subscribe : noSubscribe,
    enabled ? resource.getSnapshot : () => disabled,
    enabled ? resource.getSnapshot : () => disabled,
  )
  useEffect(() => {
    if (!enabled) return
    const visible = () => {
      if (!document.hidden) void resource.refresh(false)
    }
    document.addEventListener('visibilitychange', visible)
    return () => document.removeEventListener('visibilitychange', visible)
  }, [resource, enabled])
  const refresh = useCallback(() => resource.refresh(), [resource])
  return { ...observation, refresh }
}
