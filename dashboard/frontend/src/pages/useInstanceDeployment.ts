import { useCallback, useEffect, useRef, useState } from 'react'
import { responseErrorMessage } from './configPageRequestErrors'
import { withRequestTimeout } from '../utils/boundedRequest'

export type InstanceMode = 'router' | 'engine'
export interface InstanceDeployment {
  ownership: 'managed' | 'external' | 'kubernetes'
  controller_available: boolean
  desired_mode?: InstanceMode
  observed_mode?: InstanceMode | 'unknown'
  active_deployment?: string
  deployment?: string
  model?: string
  can_switch: boolean
  unavailable_reason?: string
  operation?: {
    id: string
    phase: string
    target_mode: InstanceMode
    started_at: string
    finished_at?: string
    error?: string
    rolled_back?: boolean
  } | null
}

export function useInstanceDeployment() {
  const [status, setStatus] = useState<InstanceDeployment | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const lifetime = useRef<AbortController | null>(null)
  const reading = useRef(false)
  const refresh = useCallback(async () => {
    const signal = lifetime.current?.signal
    if (!signal || signal.aborted || reading.current) return
    reading.current = true
    try {
      const next = await withRequestTimeout(
        async (requestSignal) => {
          const response = await fetch('/api/instance', { signal: requestSignal })
          if (!response.ok) throw new Error(await responseErrorMessage(response))
          return response.json() as Promise<InstanceDeployment>
        },
        signal,
        8000,
      )
      if (signal.aborted) return
      setStatus(next)
      setError(null)
    } catch (cause) {
      if (!signal.aborted) {
        setStatus(null)
        setError(cause instanceof Error ? cause.message : 'Instance status is unavailable.')
      }
    } finally {
      reading.current = false
    }
  }, [])
  useEffect(() => {
    const controller = new AbortController()
    lifetime.current = controller
    void refresh()
    const visibleRefresh = () => {
      if (document.visibilityState !== 'hidden') void refresh()
    }
    const timer = window.setInterval(visibleRefresh, 3000)
    document.addEventListener('visibilitychange', visibleRefresh)
    return () => {
      controller.abort()
      window.clearInterval(timer)
      document.removeEventListener('visibilitychange', visibleRefresh)
    }
  }, [refresh])
  const deploy = async (mode: InstanceMode, deployment?: string) => {
    if (submitting) return
    setSubmitting(true)
    setError(null)
    try {
      const next = await withRequestTimeout(async (signal) => {
        const response = await fetch('/api/instance/deploy', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          signal,
          body: JSON.stringify({ mode, deployment, request_id: crypto.randomUUID() }),
        })
        if (!response.ok) throw new Error(await responseErrorMessage(response))
        return response.json() as Promise<InstanceDeployment>
      }, lifetime.current?.signal)
      setStatus(next)
      window.dispatchEvent(new Event('instance-deployed'))
      return next
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Deployment failed.')
      throw cause
    } finally {
      setSubmitting(false)
    }
  }
  const busy =
    submitting ||
    Boolean(
      status?.operation &&
        !status.operation.finished_at &&
        !['ready', 'completed', 'failed', 'rolled_back'].includes(status.operation.phase),
    )
  return { status, error, busy, refresh, deploy }
}
