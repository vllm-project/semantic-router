import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { SystemStatus } from '../utils/routerRuntime'
import type { RouterConfig } from './dashboardPageTypes'
import type { CanonicalGlobalConfig } from './configPageSupport'
import { responseErrorMessage } from './configPageRequestErrors'
import {
  decisionModelPatch,
  readDecisionModelApplyResult,
  type DecisionModelActivation,
  type DecisionModelApplyResult,
  type ModelRuntimeInventory,
} from './decisionModelManagement'
import { configuredDecisionModel, type DecisionModelName } from './decisionModelSupport'
import { createVisibilityAwareRequest } from './visibilityAwareRequest'

async function fetchSnapshot<T>(path: string, signal: AbortSignal): Promise<T> {
  const response = await fetch(path, { headers: { Accept: 'application/json' }, signal })
  if (!response.ok) throw new Error(await responseErrorMessage(response))
  return response.json() as Promise<T>
}

export function useDecisionModelManagement() {
  const [config, setConfig] = useState<RouterConfig | null>(null)
  const [global, setGlobal] = useState<CanonicalGlobalConfig | null>(null)
  const [status, setStatus] = useState<SystemStatus | null>(null)
  const [inventory, setInventory] = useState<ModelRuntimeInventory | null>(null)
  const [activation, setActivation] = useState<DecisionModelActivation | null>(null)
  const [choice, setChoice] = useState<DecisionModelName | null>(null)
  const [errors, setErrors] = useState<string[]>([])
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [deploying, setDeploying] = useState(false)
  const [applyResult, setApplyResult] = useState<DecisionModelApplyResult | null>(null)
  const [applyError, setApplyError] = useState<string | null>(null)
  const [updatedAt, setUpdatedAt] = useState<Date | null>(null)
  const mutationInProgress = useRef(false)
  const lifetime = useRef<AbortController | null>(null)

  const refreshSnapshot = useCallback(async () => {
    const mounted = lifetime.current
    if (mutationInProgress.current || !mounted || mounted.signal.aborted) return
    setRefreshing(true)
    const reads = new AbortController()
    const cancel = () => reads.abort()
    mounted.signal.addEventListener('abort', cancel, { once: true })
    const timeout = window.setTimeout(() => {
      reads.abort(new Error('Status request timed out after 10 seconds.'))
    }, 10_000)
    const results = await Promise.allSettled([
      fetchSnapshot<RouterConfig>('/api/router/config/all', reads.signal),
      fetchSnapshot<CanonicalGlobalConfig>('/api/router/config/global', reads.signal),
      fetchSnapshot<SystemStatus>('/api/status', reads.signal),
      fetchSnapshot<ModelRuntimeInventory>(
        '/api/router/api/v1/inventory/model-runtime',
        reads.signal,
      ),
      fetchSnapshot<DecisionModelActivation>('/api/router/api/v1/config/hash', reads.signal),
    ]).finally(() => {
      window.clearTimeout(timeout)
      mounted.signal.removeEventListener('abort', cancel)
    })
    if (mounted.signal.aborted) return
    const [configResult, globalResult, statusResult, inventoryResult, activationResult] = results
    // Clear failed observations: a previous ready result is not current evidence.
    setConfig(configResult.status === 'fulfilled' ? configResult.value : null)
    setGlobal(globalResult.status === 'fulfilled' ? globalResult.value : null)
    setStatus(statusResult.status === 'fulfilled' ? statusResult.value : null)
    setInventory(inventoryResult.status === 'fulfilled' ? inventoryResult.value : null)
    setActivation(activationResult.status === 'fulfilled' ? activationResult.value : null)
    const labels = [
      'Routing configuration',
      'Saved decision model',
      'Service status',
      'Runtime deployments',
      'Configuration activation',
    ]
    setErrors(
      results.flatMap((result, index) =>
        result.status === 'rejected'
          ? [
              `${labels[index]}: ${result.reason instanceof Error ? result.reason.message : 'Unavailable'}`,
            ]
          : [],
      ),
    )
    setUpdatedAt(new Date())
    setLoading(false)
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

  const savedModel = global ? configuredDecisionModel({ global }) : null
  const selectedModel = choice ?? savedModel
  const deploy = async () => {
    if (!selectedModel || !global || mutationInProgress.current) return
    mutationInProgress.current = true
    setDeploying(true)
    setApplyResult(null)
    setApplyError(null)
    // Finish an earlier poll before the write, so it cannot replace the result
    // of the post-write refresh with an older saved configuration.
    await request.run({ allowHidden: true })
    try {
      const response = await fetch('/api/router/config/global/update', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(decisionModelPatch(selectedModel)),
      })
      setApplyResult(await readDecisionModelApplyResult(response))
      window.dispatchEvent(new Event('config-deployed'))
    } catch (cause) {
      setApplyError(cause instanceof Error ? cause.message : 'The deployment request failed.')
    } finally {
      mutationInProgress.current = false
      await request.run({ allowHidden: true })
      setDeploying(false)
    }
  }

  return {
    config,
    global,
    status,
    inventory,
    activation,
    savedModel,
    selectedModel,
    selectModel: setChoice,
    errors,
    loading,
    refreshing,
    deploying,
    applyResult,
    applyError,
    updatedAt,
    deploy,
    refresh: () => request.run({ allowHidden: true }),
  }
}
