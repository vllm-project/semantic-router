import { useEffect, useState } from 'react'

import bundledCatalog from '../generated/modelCatalog.json'
import type { BuiltInModelCatalog } from '../types/modelCatalog'
import { getBuiltInModelCatalog } from '../utils/modelCatalogApi'

const fallbackCatalog = bundledCatalog as unknown as BuiltInModelCatalog

export default function useBuiltInModelCatalog() {
  const [catalog, setCatalog] = useState<BuiltInModelCatalog>(fallbackCatalog)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    void getBuiltInModelCatalog(controller.signal)
      .then((nextCatalog) => {
        setCatalog(nextCatalog)
        setError(null)
      })
      .catch((cause: unknown) => {
        if (controller.signal.aborted) return
        setError(cause instanceof Error ? cause.message : 'The model catalog is unavailable.')
      })
    return () => controller.abort()
  }, [])

  return { catalog, error }
}
