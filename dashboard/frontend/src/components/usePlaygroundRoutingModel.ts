import { useCallback, useEffect, useRef, useState } from 'react'

import {
  getRouterModelsEndpoint,
  listRouterModels,
  type RouterModelOption,
  selectRouterAutoModel,
} from '../utils/routerModelSelection'
import { fetchPlaygroundModelPayload } from './playgroundModelDiscovery'

export type PlaygroundRoutingModelStatus = 'discovering' | 'ready' | 'error'

interface PlaygroundRoutingModelSelection {
  error: string | null
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

export function usePlaygroundRoutingModel(
  endpoint: string,
  getAccessToken: () => Promise<string>,
  enabled = true,
): PlaygroundRoutingModelState {
  const [selection, setSelection] = useState<PlaygroundRoutingModelSelection>({
    error: null,
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
        setSelection({ error: null, model: '', models: [], status: 'discovering' })
        return []
      }
      const controller = new AbortController()
      requestRef.current = controller
      const expected = new Set(options.expectedModelIds?.filter(Boolean) ?? [])
      const deadline = Date.now() + Math.max(0, options.timeoutMilliseconds ?? 0)
      let delay = 200
      setSelection((current) => ({ ...current, error: null, status: 'discovering' }))

      try {
        while (true) {
          const routerPayload = await fetchPlaygroundModelPayload(
            getRouterModelsEndpoint(endpoint),
            controller.signal,
            getAccessToken,
          )
          // /v1/models is already the key-scoped authorization projection. A
          // passthrough record is therefore safe to show to every caller that
          // received it; Dashboard role must not hide a Router-authorized model.
          const models = listRouterModels(routerPayload, { includeIndividualModels: true })
          const routingModels = models.filter((option) => option.kind !== 'individual')
          const automaticModel = selectRouterAutoModel(routerPayload)
          const defaultModel = models.some((option) => option.id === automaticModel)
            ? automaticModel
            : (routingModels[0]?.id ?? models[0]?.id)
          if (!defaultModel || models.length === 0) {
            throw new Error('The router did not advertise a selectable model.')
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
            error: null,
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
          setSelection((current) => ({
            ...current,
            error: error instanceof Error ? error.message : 'Model discovery failed.',
            status: 'error',
          }))
        }
        throw error
      } finally {
        if (requestRef.current === controller) requestRef.current = null
      }
    },
    [enabled, endpoint, getAccessToken],
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
