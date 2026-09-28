import { useCallback, useEffect, useState } from 'react'

import {
  getRouterModelsEndpoint,
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

export function usePlaygroundRoutingModel(endpoint: string): PlaygroundRoutingModelState {
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

    void fetch(getRouterModelsEndpoint(endpoint), {
      cache: 'no-store',
      headers: { Accept: 'application/json' },
      signal: controller.signal,
    })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`Model discovery failed with status ${response.status}`)
        }
        return response.json() as Promise<unknown>
      })
      .then((payload) => {
        const models = listRouterModels(payload)
        const automaticModel = selectRouterAutoModel(payload)
        const defaultModel = models.some((option) => option.id === automaticModel)
          ? automaticModel
          : models[0]?.id
        if (!defaultModel || models.length === 0) {
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
  }, [attempt, endpoint])

  useEffect(() => {
    window.addEventListener('config-deployed', retry)
    return () => window.removeEventListener('config-deployed', retry)
  }, [retry])

  return { ...selection, retry, setModel }
}
