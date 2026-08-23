import { useCallback, useEffect, useRef, useState } from 'react'

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
  refresh: (options?: PlaygroundModelRefreshOptions) => Promise<RouterModelOption[]>
  setModel: (model: string) => void
}

interface PlaygroundModelRefreshOptions {
  expectedModelIds?: readonly string[]
  timeoutMilliseconds?: number
}

function waitWithAbort(milliseconds: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(new DOMException('Aborted', 'AbortError'))
      return
    }
    const onAbort = () => {
      window.clearTimeout(timer)
      reject(new DOMException('Aborted', 'AbortError'))
    }
    const timer = window.setTimeout(() => {
      signal.removeEventListener('abort', onAbort)
      resolve()
    }, milliseconds)
    signal.addEventListener('abort', onAbort, { once: true })
  })
}

async function fetchJSON(
  endpoint: string,
  signal: AbortSignal,
  getAccessToken: () => Promise<string>,
): Promise<unknown> {
  const response = await fetch(endpoint, {
    cache: 'no-store',
    credentials: 'omit',
    headers: {
      Accept: 'application/json',
      Authorization: `Bearer ${await getAccessToken()}`,
    },
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
  getAccessToken: () => Promise<string>,
  enabled = true,
): PlaygroundRoutingModelState {
  const [selection, setSelection] = useState<PlaygroundRoutingModelSelection>({
    model: '',
    models: [],
    status: 'discovering',
  })
  const requestRef = useRef<AbortController | null>(null)
  const setModel = useCallback((model: string) => {
    setSelection((current) =>
      current.models.some((option) => option.id === model) ? { ...current, model } : current,
    )
  }, [])

  const refresh = useCallback(
    async (options: PlaygroundModelRefreshOptions = {}): Promise<RouterModelOption[]> => {
      requestRef.current?.abort()
      if (!enabled) {
        setSelection({ model: '', models: [], status: 'discovering' })
        return []
      }
      const controller = new AbortController()
      requestRef.current = controller
      const expected = new Set(options.expectedModelIds?.filter(Boolean) ?? [])
      const deadline = Date.now() + Math.max(0, options.timeoutMilliseconds ?? 0)
      let delay = 200
      setSelection((current) => ({ ...current, status: 'discovering' }))

      try {
        while (true) {
          const routerPayload = await fetchJSON(
            getRouterModelsEndpoint(endpoint),
            controller.signal,
            getAccessToken,
          )
          const authorizedModels = listRouterModels(routerPayload)
          const models = includeIndividualModels
            ? authorizedModels
            : authorizedModels.filter((option) => option.kind !== 'individual')
          const routingModels = authorizedModels.filter((option) => option.kind !== 'individual')
          const automaticModel = selectRouterAutoModel(routerPayload)
          const defaultModel = models.some((option) => option.id === automaticModel)
            ? automaticModel
            : routingModels[0]?.id
          if (!defaultModel || routingModels.length === 0) {
            throw new Error('The router did not advertise a selectable routing model.')
          }
          const expectedVisible =
            expected.size === 0 || models.some((model) => expected.has(model.id))
          if (!expectedVisible && Date.now() < deadline) {
            await waitWithAbort(delay, controller.signal)
            delay = Math.min(Math.round(delay * 1.6), 1_200)
            continue
          }
          if (!expectedVisible) {
            throw new Error('The published model is not visible from the Router yet.')
          }
          setSelection((current) => ({
            model: models.some((option) => option.id === current.model)
              ? current.model
              : defaultModel,
            models,
            status: 'ready',
          }))
          return models
        }
      } catch (error) {
        if (!controller.signal.aborted) {
          setSelection((current) => ({ ...current, status: 'error' }))
        }
        throw error
      } finally {
        if (requestRef.current === controller) requestRef.current = null
      }
    },
    [enabled, endpoint, getAccessToken, includeIndividualModels],
  )

  useEffect(() => {
    void refresh().catch((error: unknown) => {
      if (!(error instanceof DOMException && error.name === 'AbortError')) {
        console.warn('Playground model discovery failed.', error)
      }
    })
    return () => requestRef.current?.abort()
  }, [refresh])

  return { ...selection, refresh, setModel }
}
