import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  systemOneError,
  type SystemOneRequest,
  type SystemOneResponse,
} from './systemOnePlayground'

import {
  isSystemOneRoutes,
  systemOneTargets,
  systemOneTargetRequest,
  systemOneTargetTimeout,
  type SystemOneDeployment,
  type SystemOneRoutes,
} from './systemOneTargets'
export type { SystemOneDeployment } from './systemOneTargets'

interface Capabilities {
  serving_mode: 'router' | 'engine'
  timeout_ms: number
  deployments: SystemOneDeployment[]
}
export interface SystemOneRun {
  request: SystemOneRequest
  response: SystemOneResponse
  elapsed: number
  completedAt: Date
  deployment: string
  targetKind?: 'route' | 'deployment'
  targetKey?: string
}

async function readJSON(response: Response): Promise<unknown> {
  const text = await response.text()
  try {
    return JSON.parse(text)
  } catch {
    throw new Error(
      response.ok
        ? 'The service returned an invalid response.'
        : `The service returned HTTP ${response.status}.`,
    )
  }
}

function isCapabilities(value: unknown): value is Capabilities {
  if (!value || typeof value !== 'object') return false
  const body = value as Capabilities
  return (
    ['router', 'engine'].includes(body.serving_mode) &&
    Number.isFinite(body.timeout_ms) &&
    body.timeout_ms >= 0 &&
    Array.isArray(body.deployments) &&
    body.deployments.every(
      (item) =>
        typeof item.id === 'string' &&
        typeof item.model === 'string' &&
        typeof item.ready === 'boolean' &&
        Array.isArray(item.question_types) &&
        item.question_types.every((kind) => typeof kind === 'string') &&
        Array.isArray(item.surfaces),
    )
  )
}

function isDecisionResponse(value: unknown): value is SystemOneResponse {
  if (!value || typeof value !== 'object') return false
  const body = value as SystemOneResponse
  return (
    typeof body.model === 'string' &&
    !!body.answers &&
    typeof body.answers === 'object' &&
    !Array.isArray(body.answers) &&
    !!body.usage &&
    typeof body.usage.input_tokens === 'number' &&
    typeof body.usage.output_tokens === 'number' &&
    (!body.routing ||
      (['recipe', 'decision', 'algorithm', 'stage', 'selected_model', 'quality'].every(
        (key) => typeof (body.routing as unknown as Record<string, unknown>)[key] === 'string',
      ) &&
        Number.isSafeInteger(body.routing.model_calls) &&
        body.routing.model_calls >= 0))
  )
}

export function useSystemOnePlayground() {
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null)
  const [loading, setLoading] = useState(true)
  const [capabilityError, setCapabilityError] = useState<string | null>(null)
  const [selectedId, setSelectedId] = useState('')
  const [routes, setRoutes] = useState<SystemOneRoutes | null>(null)
  const [routesLoading, setRoutesLoading] = useState(true)
  const [routeError, setRouteError] = useState<string | null>(null)
  const [revision, setRevision] = useState(0)
  const [running, setRunning] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [result, setResult] = useState<SystemOneRun | null>(null)
  const active = useRef<AbortController | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    let mounted = true
    const timer = window.setTimeout(() => controller.abort(), 10000)
    setLoading(true)
    setCapabilityError(null)
    void (async () => {
      try {
        const response = await fetch('/api/decision-model/capabilities', {
          signal: controller.signal,
        })
        const body = await readJSON(response)
        if (!response.ok)
          throw new Error(systemOneError(body, `Model discovery failed (HTTP ${response.status}).`))
        if (!isCapabilities(body)) throw new Error('Model discovery returned an invalid response.')
        if (!mounted) return
        setCapabilities(body)
      } catch (cause) {
        if (!mounted) return
        setCapabilities(null)
        setCapabilityError(
          controller.signal.aborted
            ? 'Model discovery timed out. Try refreshing the runtime list.'
            : cause instanceof Error
              ? cause.message
              : 'Model discovery failed.',
        )
      } finally {
        window.clearTimeout(timer)
        if (mounted) setLoading(false)
      }
    })()
    return () => {
      mounted = false
      window.clearTimeout(timer)
      controller.abort()
    }
  }, [revision])

  useEffect(() => {
    const controller = new AbortController()
    let mounted = true
    const timer = window.setTimeout(() => controller.abort(), 10000)
    setRoutesLoading(true)
    setRouteError(null)
    void (async () => {
      try {
        const response = await fetch('/api/decision-model/routes', { signal: controller.signal })
        const body = await readJSON(response)
        if (!response.ok)
          throw new Error(systemOneError(body, `Route discovery failed (HTTP ${response.status}).`))
        if (!isSystemOneRoutes(body))
          throw new Error('Route discovery returned an invalid response.')
        if (mounted) setRoutes(body)
      } catch (cause) {
        if (!mounted) return
        setRoutes(null)
        setRouteError(
          controller.signal.aborted
            ? 'Route discovery timed out. Direct models remain available.'
            : cause instanceof Error
              ? cause.message
              : 'Route discovery failed.',
        )
      } finally {
        window.clearTimeout(timer)
        if (mounted) setRoutesLoading(false)
      }
    })()
    return () => {
      mounted = false
      window.clearTimeout(timer)
      controller.abort()
    }
  }, [revision])

  const targets = useMemo(
    () => systemOneTargets(capabilities?.deployments ?? [], routes, capabilities?.timeout_ms),
    [capabilities, routes],
  )
  const selected =
    targets.find((item) => item.key === selectedId) ??
    targets.find((item) => item.ready && item.surfaces.includes('decisions')) ??
    targets[0]
  useEffect(() => {
    if (selected) setSelectedId(selected.key)
  }, [selected])

  useEffect(
    () => () => {
      active.current?.abort()
      active.current = null
    },
    [],
  )

  const run = useCallback(
    async (request: SystemOneRequest) => {
      if (active.current || !selected) return
      const submission = systemOneTargetRequest(selected, request)
      const controller = new AbortController()
      active.current = controller
      const started = performance.now()
      let timedOut = false
      const timer = window.setTimeout(() => {
        timedOut = true
        controller.abort()
      }, systemOneTargetTimeout(selected))
      const ticker = window.setInterval(() => setElapsed(performance.now() - started), 100)
      setRunning(true)
      setElapsed(0)
      setError(null)
      setNotice(null)
      setResult(null)
      try {
        const response = await fetch(submission.path, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(submission.body),
          signal: controller.signal,
        })
        const body = await readJSON(response)
        if (!response.ok)
          throw new Error(systemOneError(body, `The test failed (HTTP ${response.status}).`))
        if (!isDecisionResponse(body))
          throw new Error('The runtime returned an invalid System One response.')
        if (active.current !== controller) return
        setResult({
          request,
          response: body,
          elapsed: performance.now() - started,
          completedAt: new Date(),
          deployment: selected.id,
          targetKind: selected.kind,
          targetKey: selected.key,
        })
      } catch (cause) {
        if (active.current !== controller) return
        if (controller.signal.aborted) {
          if (timedOut)
            setError('The request timed out. The runtime may still be finishing its work.')
          else
            setNotice(
              'Stopped waiting for this request. The runtime may still be finishing its work.',
            )
        } else
          setError(cause instanceof Error ? cause.message : 'The test could not reach the runtime.')
      } finally {
        window.clearTimeout(timer)
        window.clearInterval(ticker)
        if (active.current === controller) {
          active.current = null
          setRunning(false)
        }
      }
    },
    [selected],
  )

  return {
    capabilities,
    loading: loading && routesLoading,
    refreshing: loading || routesLoading,
    capabilityError,
    routeError,
    routes,
    targets,
    selectedId: selected?.key ?? selectedId,
    setSelectedId,
    selected,
    refresh: () => setRevision((value) => value + 1),
    running,
    elapsed,
    error,
    notice,
    result,
    run,
    cancel: () => active.current?.abort(),
  }
}
