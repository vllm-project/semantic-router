import { useCallback, useEffect, useState } from 'react'

import type { BuiltInModelCatalog } from '../types/modelCatalog'
import { getBuiltInModelCatalog } from '../utils/modelCatalogApi'

export function useBuiltInModelCatalog() {
  const [catalog, setCatalog] = useState<BuiltInModelCatalog | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [revision, setRevision] = useState(0)
  const retry = useCallback(() => setRevision((current) => current + 1), [])

  useEffect(() => {
    const controller = new AbortController()
    setLoading(true)
    setError(null)
    void getBuiltInModelCatalog(controller.signal)
      .then((nextCatalog) => {
        if (!controller.signal.aborted) setCatalog(nextCatalog)
      })
      .catch((reason: unknown) => {
        if (controller.signal.aborted) return
        setCatalog(null)
        setError(reason instanceof Error ? reason.message : 'Failed to load built-in models.')
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false)
      })
    return () => controller.abort()
  }, [revision])

  return { catalog, loading, error, retry }
}
