import { useCallback, useEffect, useState } from 'react'

import bundledCatalog from '../modelCatalogDocument'
import type { BuiltInModelCatalog } from '../types/modelCatalog'
import { getBuiltInModelCatalog } from '../utils/modelCatalogApi'

const fallbackCatalog = bundledCatalog as unknown as BuiltInModelCatalog

export default function useBuiltInModelCatalog() {
  const [catalog, setCatalog] = useState<BuiltInModelCatalog>(fallbackCatalog)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [source, setSource] = useState<'bundled' | 'server'>('bundled')
  const [attempt, setAttempt] = useState(0)
  const retry = useCallback(() => setAttempt((value) => value + 1), [])

  useEffect(() => {
    const controller = new AbortController()
    setLoading(true)
    void getBuiltInModelCatalog(controller.signal)
      .then((nextCatalog) => {
        if (controller.signal.aborted) return
        setCatalog(nextCatalog)
        setSource('server')
        setError(null)
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

  return { catalog, error, loading, source, retry }
}
