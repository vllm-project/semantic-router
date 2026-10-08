import { engineModelInventory } from './decisionRuntimeInventory'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { SystemStatus } from '../utils/routerRuntime'
import type { RouterConfig } from './dashboardPageTypes'
import type { CanonicalGlobalConfig } from './configPageSupport'
import { responseErrorMessage } from './configPageRequestErrors'
import {
  readDecisionModelApplyResult,
  type DecisionModelActivation,
  type DecisionModelApplyResult,
  type ModelRuntimeInventory,
} from './decisionModelManagement'
import {
  configuredDecisionModel,
  withDecisionModel,
  type DecisionModelName,
} from './decisionModelSupport'
import { createVisibilityAwareRequest } from './visibilityAwareRequest'
import { withRequestTimeout } from '../utils/boundedRequest'
import {
  withDecisionRuntimeDeployment,
  type DecisionRuntimeDeploymentRequest,
} from './decisionRuntimeDeployment'

async function fetchSnapshot<T>(path: string, signal: AbortSignal): Promise<T> {
  const response = await fetch(path, { headers: { Accept: 'application/json' }, signal })
  if (!response.ok) throw new Error(await responseErrorMessage(response))
  return response.json() as Promise<T>
}

export function useDecisionModelManagement(engine = false) {
  const [config, setConfig] = useState<RouterConfig | null>(null)
  const [global, setGlobal] = useState<CanonicalGlobalConfig | null>(null)
  const [status, setStatus] = useState<SystemStatus | null>(null)
  const [inventory, setInventory] = useState<ModelRuntimeInventory | null>(null)
  const [activation, setActivation] = useState<DecisionModelActivation | null>(null)
  const [choice, setChoice] = useState<DecisionModelName | null>(null)
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [deploying, setDeploying] = useState(false)
  const [applyResult, setApplyResult] = useState<DecisionModelApplyResult | null>(null)
  const [applyError, setApplyError] = useState<string | null>(null)
  const [updatedAt, setUpdatedAt] = useState<Date | null>(null)
  const mutationInProgress = useRef(false)
  const lifetime = useRef<AbortController | null>(null)
  const configurationNeeded = useRef(true)
  const observedConfiguration = useRef<string | null>(null)

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
    const readConfiguration = configurationNeeded.current
    configurationNeeded.current = false
    const observe = async <T>(path: string, label: string, update: (value: T | null) => void) => {
      try {
        const value = await fetchSnapshot<T>(path, reads.signal)
        if (mounted.signal.aborted) return
        update(value)
        setErrors((current) => ({ ...current, [label]: '' }))
        return value
      } catch (cause) {
        if (mounted.signal.aborted) return
        // A failed observation cannot keep an earlier ready state alive.
        update(null)
        setErrors((current) => ({
          ...current,
          [label]: `${label}: ${cause instanceof Error ? cause.message : 'Unavailable'}`,
        }))
        if (path === '/api/router/config/all' || path === '/api/router/config/global') {
          configurationNeeded.current = true
        }
      }
    }
    const readSavedConfiguration = () =>
      Promise.all([
        observe<RouterConfig>('/api/router/config/all', 'Routing configuration', setConfig),
        observe<CanonicalGlobalConfig>(
          '/api/router/config/global',
          'Saved decision model',
          setGlobal,
        ).finally(() => {
          if (!mounted.signal.aborted) setLoading(false)
        }),
      ])
    const readActivation = async () => {
      const value = await observe<DecisionModelActivation>(
        '/api/router/api/v1/config/hash',
        'Configuration activation',
        setActivation,
      )
      const revision = value?.generated_runtime_hash
      if (!revision || mounted.signal.aborted) return
      const changed = observedConfiguration.current !== revision
      observedConfiguration.current = revision
      // The revision only invalidates a cached observation. It never gates
      // deployment or demands equality with the active runtime. Read within
      // this poll so external changes appear promptly without polling config.
      if (changed && !readConfiguration) await readSavedConfiguration()
    }
    // Render each observation when it arrives. A slow inventory endpoint must
    // not hide an already available saved selection or deployment controls.
    await Promise.all([
      ...(readConfiguration ? [readSavedConfiguration()] : []),
      observe<SystemStatus>('/api/status', 'Service status', setStatus),
      observe<ModelRuntimeInventory>(
        engine ? '/api/instance/models' : '/api/router/api/v1/inventory/model-runtime',
        'Runtime deployments',
        (value) => setInventory(value && engine ? engineModelInventory(value) : value),
      ),
      ...(engine ? [] : [readActivation()]),
    ]).finally(() => {
      window.clearTimeout(timeout)
      mounted.signal.removeEventListener('abort', cancel)
    })
    if (mounted.signal.aborted) return
    setUpdatedAt(new Date())
    setLoading(false)
    setRefreshing(false)
  }, [engine])
  const request = useMemo(() => createVisibilityAwareRequest(refreshSnapshot), [refreshSnapshot])

  useEffect(() => {
    const mounted = new AbortController()
    lifetime.current = mounted
    configurationNeeded.current = true
    observedConfiguration.current = null
    void request.run({ allowHidden: true })
    const refreshVisible = () => {
      void request.run()
    }
    const configurationChanged = () => {
      configurationNeeded.current = true
      void request.run()
    }
    window.addEventListener('config-deployed', configurationChanged)
    document.addEventListener('visibilitychange', refreshVisible)
    const interval = window.setInterval(refreshVisible, 10_000)
    return () => {
      mounted.abort()
      if (lifetime.current === mounted) lifetime.current = null
      window.clearInterval(interval)
      document.removeEventListener('visibilitychange', refreshVisible)
      window.removeEventListener('config-deployed', configurationChanged)
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
      const current = await withRequestTimeout((signal) =>
        fetchSnapshot<RouterConfig>('/api/router/config/all', signal),
      )
      const next = withDecisionModel(current, selectedModel)
      const response = await fetch('/api/router/config/update', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(next),
      })
      setApplyResult(await readDecisionModelApplyResult(response))
      window.dispatchEvent(new Event('config-deployed'))
      return next
    } catch (cause) {
      setApplyError(cause instanceof Error ? cause.message : 'The deployment request failed.')
      throw cause
    } finally {
      mutationInProgress.current = false
      configurationNeeded.current = true
      await request.run({ allowHidden: true })
      setDeploying(false)
    }
  }

  const deployRuntime = async (deployment: DecisionRuntimeDeploymentRequest) => {
    if (mutationInProgress.current) throw new Error('Another configuration request is in progress.')
    mutationInProgress.current = true
    setDeploying(true)
    setApplyResult(null)
    setApplyError(null)
    await request.run({ allowHidden: true })
    try {
      // Read immediately before writing: the manager's static snapshot may be
      // older than edits made in another configuration page or browser.
      const current = await withRequestTimeout((signal) =>
        fetchSnapshot<RouterConfig>('/api/router/config/all', signal),
      )
      const next = withDecisionRuntimeDeployment(current, deployment)
      const response = await fetch('/api/router/config/update', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(next),
      })
      const result = await readDecisionModelApplyResult(response)
      window.dispatchEvent(new Event('config-deployed'))
      return result
    } finally {
      mutationInProgress.current = false
      configurationNeeded.current = true
      await request.run({ allowHidden: true })
      setDeploying(false)
    }
  }

  const updateConfig = async (mutate: (current: RouterConfig) => RouterConfig) => {
    if (mutationInProgress.current) throw new Error('Another configuration request is in progress.')
    mutationInProgress.current = true
    setDeploying(true)
    try {
      await request.run({ allowHidden: true })
      const current = await withRequestTimeout((signal) =>
        fetchSnapshot<RouterConfig>('/api/router/config/all', signal),
      )
      const response = await fetch('/api/router/config/update', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(mutate(current)),
      })
      const result = await readDecisionModelApplyResult(response)
      setApplyResult(result)
      window.dispatchEvent(new Event('config-deployed'))
      return result
    } finally {
      mutationInProgress.current = false
      configurationNeeded.current = true
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
    errors: Object.values(errors).filter(Boolean),
    loading,
    refreshing,
    deploying,
    applyResult,
    applyError,
    updatedAt,
    deploy,
    deployRuntime,
    updateConfig,
    refresh: () => {
      configurationNeeded.current = true
      return request.run({ allowHidden: true })
    },
  }
}
