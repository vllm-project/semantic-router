import { useEffect, useState } from 'react'

import { listProviderCatalog, type ProviderCatalogDisplay } from '../utils/providerCatalogApi'

export function useProviderCatalogDisplayMap(): ReadonlyMap<string, ProviderCatalogDisplay> {
  const [catalog, setCatalog] = useState<ReadonlyMap<string, ProviderCatalogDisplay>>(new Map())

  useEffect(() => {
    const controller = new AbortController()
    void (async () => {
      const displays = new Map<string, ProviderCatalogDisplay>()
      let cursor: string | undefined
      do {
        const page = await listProviderCatalog({ cursor, pageSize: 200 }, controller.signal)
        for (const provider of page.data) displays.set(provider.providerId, provider.display)
        cursor = page.page.nextCursor
      } while (cursor && !controller.signal.aborted)
      if (!controller.signal.aborted) setCatalog(displays)
    })().catch(() => {
      if (!controller.signal.aborted) setCatalog(new Map())
    })
    return () => controller.abort()
  }, [])

  return catalog
}
