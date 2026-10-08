import { useCallback, useEffect, useRef, useState } from 'react'
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
import { useDashboardObservation } from '../utils/useDashboardObservation'
import { invalidateDashboardObservations } from '../utils/dashboardObservationCache'
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

let observedConfigurationRevision: string | null = null
const configPath = '/api/router/config/all'
const globalPath = '/api/router/config/global'

export function useDecisionModelManagement() {
  const configObservation = useDashboardObservation<RouterConfig>(configPath, {
    freshMs: 60_000,
    pollMs: 0,
  })
  const globalObservation = useDashboardObservation<CanonicalGlobalConfig>(globalPath, {
    freshMs: 60_000,
    pollMs: 0,
  })
  const statusObservation = useDashboardObservation<SystemStatus>('/api/status')
  const inventoryObservation = useDashboardObservation<ModelRuntimeInventory>(
    '/api/router/api/v1/inventory/model-runtime',
  )
  const activationObservation = useDashboardObservation<DecisionModelActivation>(
    '/api/router/api/v1/config/hash',
  )
  const config = configObservation.data
  // Authored config and effective defaults are different contracts. Cache both,
  // but never make runtime observations wait for either configuration view.
  const global = globalObservation.data
  const status = statusObservation.data
  const inventory = inventoryObservation.data
  const activation = activationObservation.data
  const [choice, setChoice] = useState<DecisionModelName | null>(null)
  const [deploying, setDeploying] = useState(false)
  const [applyResult, setApplyResult] = useState<DecisionModelApplyResult | null>(null)
  const [applyError, setApplyError] = useState<string | null>(null)
  const mutationInProgress = useRef(false)
  const { refresh: refreshConfig } = configObservation
  const { refresh: refreshGlobal } = globalObservation
  const { refresh: refreshStatus } = statusObservation
  const { refresh: refreshInventory } = inventoryObservation
  const { refresh: refreshActivation } = activationObservation
  const refreshSnapshot = useCallback(
    () =>
      Promise.all([
        refreshConfig(),
        refreshGlobal(),
        refreshStatus(),
        refreshInventory(),
        refreshActivation(),
      ]),
    [refreshConfig, refreshGlobal, refreshStatus, refreshInventory, refreshActivation],
  )
  useEffect(() => {
    const revision = activation?.generated_runtime_hash
    if (!revision) return
    if (observedConfigurationRevision && observedConfigurationRevision !== revision) {
      invalidateDashboardObservations([configPath, globalPath])
    }
    observedConfigurationRevision = revision
  }, [activation?.generated_runtime_hash])
  const observationState = {
    config: configObservation,
    global: globalObservation,
    status: statusObservation,
    inventory: inventoryObservation,
    activation: activationObservation,
  }
  const observations = Object.entries(observationState)
  const errors = observations.flatMap(([label, observation]) =>
    observation.error ? [`${label}: ${observation.error}`] : [],
  )
  const latest = Math.max(...observations.map(([, value]) => value.updatedAt ?? 0))
  const updatedAt = latest ? new Date(latest) : null
  const loading = globalObservation.loading
  const refreshing = observations.some(([, value]) => value.refreshing)
  const savedModel = global ? configuredDecisionModel({ global }) : null
  const selectedModel = choice ?? savedModel
  const deploy = async () => {
    if (!selectedModel || !global || mutationInProgress.current) return
    mutationInProgress.current = true
    setDeploying(true)
    setApplyResult(null)
    setApplyError(null)
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
      setDeploying(false)
      void refreshSnapshot()
    }
  }

  const deployRuntime = async (deployment: DecisionRuntimeDeploymentRequest) => {
    if (mutationInProgress.current) throw new Error('Another configuration request is in progress.')
    mutationInProgress.current = true
    setDeploying(true)
    setApplyResult(null)
    setApplyError(null)
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
      setDeploying(false)
      void refreshSnapshot()
    }
  }

  const updateConfig = async (mutate: (current: RouterConfig) => RouterConfig) => {
    if (mutationInProgress.current) throw new Error('Another configuration request is in progress.')
    mutationInProgress.current = true
    setDeploying(true)
    try {
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
      setDeploying(false)
      void refreshSnapshot()
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
    observationState,
    loading,
    refreshing,
    deploying,
    applyResult,
    applyError,
    updatedAt,
    deploy,
    deployRuntime,
    updateConfig,
    refresh: refreshSnapshot,
  }
}
