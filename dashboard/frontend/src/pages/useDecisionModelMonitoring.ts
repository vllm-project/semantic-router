import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { SystemStatus } from '../utils/routerRuntime'
import { responseErrorMessage } from './configPageRequestErrors'
import type { ModelRuntimeInventory } from './decisionModelManagement'
import { createVisibilityAwareRequest } from './visibilityAwareRequest'

// Monitoring only reads operational state. Configuration and catalog loading
// belong to their own pages and cannot delay the first runtime observation.
export function useDecisionModelMonitoring() {
  const [status, setStatus] = useState<SystemStatus | null>(null)
  const [inventory, setInventory] = useState<ModelRuntimeInventory | null>(null)
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [updatedAt, setUpdatedAt] = useState<Date | null>(null)
  const lifetime = useRef<AbortController | null>(null)

  const refreshSnapshot = useCallback(async () => {
    const mounted = lifetime.current
    if (!mounted || mounted.signal.aborted) return
    setRefreshing(true)
    const reads = new AbortController()
    const cancel = () => reads.abort()
    mounted.signal.addEventListener('abort', cancel, { once: true })
    const timeout = window.setTimeout(() => {
      reads.abort(new Error('Runtime observation timed out after 10 seconds.'))
    }, 10_000)
    const observe = async <T>(path: string, label: string, update: (value: T | null) => void) => {
      try {
        const response = await fetch(path, {
          headers: { Accept: 'application/json' },
          signal: reads.signal,
        })
        if (!response.ok) throw new Error(await responseErrorMessage(response))
        const value = (await response.json()) as T
        if (mounted.signal.aborted) return
        update(value)
        setErrors((current) => ({ ...current, [label]: '' }))
      } catch (cause) {
        if (mounted.signal.aborted) return
        update(null)
        setErrors((current) => ({
          ...current,
          [label]: `${label}: ${cause instanceof Error ? cause.message : 'Unavailable'}`,
        }))
      }
    }
    await Promise.all([
      observe<SystemStatus>('/api/status', 'Service status', setStatus),
      observe<ModelRuntimeInventory>(
        '/api/router/api/v1/inventory/model-runtime',
        'Runtime deployments',
        setInventory,
      ).finally(() => {
        if (!mounted.signal.aborted) setLoading(false)
      }),
    ]).finally(() => {
      window.clearTimeout(timeout)
      mounted.signal.removeEventListener('abort', cancel)
    })
    if (mounted.signal.aborted) return
    setUpdatedAt(new Date())
    setRefreshing(false)
  }, [])
  const request = useMemo(() => createVisibilityAwareRequest(refreshSnapshot), [refreshSnapshot])

  useEffect(() => {
    const mounted = new AbortController()
    lifetime.current = mounted
    void request.run({ allowHidden: true })
    const refreshVisible = () => {
      void request.run()
    }
    document.addEventListener('visibilitychange', refreshVisible)
    const interval = window.setInterval(refreshVisible, 10_000)
    return () => {
      mounted.abort()
      if (lifetime.current === mounted) lifetime.current = null
      window.clearInterval(interval)
      document.removeEventListener('visibilitychange', refreshVisible)
    }
  }, [request])

  return {
    status,
    inventory,
    errors: Object.values(errors).filter(Boolean),
    loading,
    refreshing,
    updatedAt,
    refresh: () => request.run({ allowHidden: true }),
  }
}
