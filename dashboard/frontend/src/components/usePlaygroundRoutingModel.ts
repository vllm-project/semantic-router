import { useCallback, useEffect, useState } from 'react'

import {
  getRouterModelsEndpoint,
  listConfiguredBackendModels,
  listRouterModels,
  type RouterModelOption,
  selectRouterAutoModel,
} from '../utils/routerModelSelection'

export type PlaygroundRoutingModelStatus = 'discovering' | 'ready' | 'error'

interface PlaygroundRoutingModelSelection {
  model: string
  models: RouterModelOption[]
  status: PlaygroundRoutingModelStatus
}

interface PlaygroundRoutingModelState extends PlaygroundRoutingModelSelection {
  retry: () => void
  setModel: (model: string) => void
}

async function fetchJSON(endpoint: string, signal: AbortSignal): Promise<unknown> {
  const response = await fetch(endpoint, {
    cache: 'no-store',
    headers: { Accept: 'application/json' },
    signal,
  })
  if (!response.ok) {
    throw new Error(`Model discovery failed with status ${response.status}`)
  }
  return response.json() as Promise<unknown>
}

export function usePlaygroundRoutingModel(
  endpoint: string,
  includeIndividualModels = false,
): PlaygroundRoutingModelState {
  const [attempt, setAttempt] = useState(0)
  const [selection, setSelection] = useState<PlaygroundRoutingModelSelection>({
    model: '',
    models: [],
    status: 'discovering',
  })
  const retry = useCallback(() => setAttempt((current) => current + 1), [])
  const setModel = useCallback((model: string) => {
    setSelection((current) =>
      current.models.some((option) => option.id === model) ? { ...current, model } : current,
    )
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    setSelection((current) => ({ ...current, status: 'discovering' }))

    const routerModelsRequest = fetchJSON(getRouterModelsEndpoint(endpoint), controller.signal)
    const configuredModelsRequest = includeIndividualModels
      ? fetchJSON('/api/router/config/all', controller.signal).catch((error: unknown) => {
          if (!controller.signal.aborted) {
            console.warn('Individual model discovery failed.', error)
          }
          return null
        })
      : Promise.resolve(null)

    void Promise.all([routerModelsRequest, configuredModelsRequest])
      .then(([routerPayload, configPayload]) => {
        const routingModels = listRouterModels(routerPayload)
        const individualModels = listConfiguredBackendModels(configPayload)
        const routingModelIds = new Set(routingModels.map((option) => option.id))
        const models = [
          ...routingModels,
          ...individualModels.filter((option) => !routingModelIds.has(option.id)),
        ]
        const automaticModel = selectRouterAutoModel(routerPayload)
        const defaultModel = models.some((option) => option.id === automaticModel)
          ? automaticModel
          : routingModels[0]?.id
        if (!defaultModel || routingModels.length === 0) {
          throw new Error('The router did not advertise a selectable routing model.')
        }
        setSelection((current) => ({
          model: models.some((option) => option.id === current.model)
            ? current.model
            : defaultModel,
          models,
          status: 'ready',
        }))
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) {
          return
        }
        console.warn('Playground model discovery failed.', error)
        setSelection((current) => ({ ...current, status: 'error' }))
      })

    return () => controller.abort()
  }, [attempt, endpoint, includeIndividualModels])

  useEffect(() => {
    window.addEventListener('config-deployed', retry)
    return () => window.removeEventListener('config-deployed', retry)
  }, [retry])

  return { ...selection, retry, setModel }
}
