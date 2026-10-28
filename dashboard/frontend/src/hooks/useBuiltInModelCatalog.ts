import { useCallback, useEffect, useState } from 'react'

import modelCatalogMetadata from '../modelCatalogMetadata'
import { modelCatalogResource } from '../utils/modelCatalogResource'

export default function useBuiltInModelCatalog() {
  const cached = modelCatalogResource.peek()
  const [catalog, setCatalog] = useState(cached?.catalog ?? modelCatalogMetadata)
  const [error, setError] = useState<string | null>(cached?.error ?? null)
  const [loading, setLoading] = useState(!cached)
  const [ready, setReady] = useState(Boolean(cached))
  const [source, setSource] = useState<'bundled' | 'server'>(cached?.source ?? 'bundled')
  const [attempt, setAttempt] = useState(0)
  const retry = useCallback(() => setAttempt((value) => value + 1), [])

  useEffect(() => {
    const controller = new AbortController()
    setLoading(true)
    void modelCatalogResource
      .read(controller.signal, attempt > 0)
      .then((result) => {
        if (controller.signal.aborted) return
        setCatalog(result.catalog)
        setSource(result.source)
        setError(result.error)
        setReady(true)
      })
      .catch((cause: unknown) => {
        if (controller.signal.aborted) return
        setError(cause instanceof Error ? cause.message : 'The model catalog is unavailable.')
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false)
      })
    return () => controller.abort()
  }, [attempt])

  return { catalog, error, loading, ready, source, retry }
}
