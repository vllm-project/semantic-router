import { useCallback, useEffect, useState } from 'react'

import {
  CANONICAL_AUTO_MODEL,
  getRouterModelsEndpoint,
  selectRouterAutoModel,
} from '../utils/routerModelSelection'

export type PlaygroundRoutingModelStatus = 'discovering' | 'ready' | 'error'

interface PlaygroundRoutingModelSelection {
  model: string
  status: PlaygroundRoutingModelStatus
}

interface PlaygroundRoutingModelState extends PlaygroundRoutingModelSelection {
  retry: () => void
}

export function usePlaygroundRoutingModel(endpoint: string): PlaygroundRoutingModelState {
  const [attempt, setAttempt] = useState(0)
  const [selection, setSelection] = useState<PlaygroundRoutingModelSelection>({
    model: CANONICAL_AUTO_MODEL,
    status: 'discovering',
  })
  const retry = useCallback(() => setAttempt((current) => current + 1), [])

  useEffect(() => {
    const controller = new AbortController()
    setSelection({ model: CANONICAL_AUTO_MODEL, status: 'discovering' })

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
        const model = selectRouterAutoModel(payload)
        if (!model) {
          throw new Error('The router did not advertise an automatic-routing model.')
        }
        setSelection({ model, status: 'ready' })
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) {
          return
        }
        console.warn('Playground model discovery failed.', error)
        setSelection({ model: CANONICAL_AUTO_MODEL, status: 'error' })
      })

    return () => controller.abort()
  }, [attempt, endpoint])

  useEffect(() => {
    window.addEventListener('config-deployed', retry)
    return () => window.removeEventListener('config-deployed', retry)
  }, [retry])

  return { ...selection, retry }
}
