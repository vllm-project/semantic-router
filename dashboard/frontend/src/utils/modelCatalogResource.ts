import catalogURL from '../../../../website/static/model-catalog/catalog.json?url&no-inline'
import { withRequestTimeout } from './boundedRequest'
import type { BuiltInModelCatalog } from '../types/modelCatalog'
import { decodeBuiltInModelCatalog, getBuiltInModelCatalog } from './modelCatalogApi'

export interface LoadedModelCatalog {
  catalog: BuiltInModelCatalog
  source: 'bundled' | 'server'
  error: string | null
}

async function loadModelCatalog(signal: AbortSignal): Promise<LoadedModelCatalog> {
  try {
    const catalog = await withRequestTimeout(getBuiltInModelCatalog, signal)
    return { catalog, source: 'server', error: null }
  } catch (error) {
    signal.throwIfAborted()
    // Keep the full fallback out of every route's JavaScript dependency graph.
    // The generated JSON is emitted as a hashed, cacheable static asset.
    const catalog = await withRequestTimeout(async (requestSignal) => {
      const response = await fetch(catalogURL, { signal: requestSignal })
      if (!response.ok) throw new Error('The bundled model catalog is unavailable.')
      return decodeBuiltInModelCatalog(await response.json())
    }, signal)
    return {
      catalog,
      source: 'bundled',
      error:
        error instanceof Error && error.name !== 'AbortError'
          ? error.message
          : 'The server model catalog request timed out.',
    }
  }
}

// A route transition reuses the public catalog already fetched in this tab.
// Concurrent consumers share one request, but leaving the last consumer aborts
// unfinished work so it cannot contend with the next page's requests.
export function createModelCatalogResource(load = loadModelCatalog) {
  let cached: LoadedModelCatalog | undefined
  let pending:
    | {
        controller: AbortController
        promise: Promise<LoadedModelCatalog>
        consumers: number
      }
    | undefined

  return {
    peek: () => cached,
    read(signal: AbortSignal, refresh = false): Promise<LoadedModelCatalog> {
      if (signal.aborted) return Promise.reject(signal.reason)
      if (cached && !refresh) return Promise.resolve(cached)
      if (!pending || pending.controller.signal.aborted) {
        const controller = new AbortController()
        const promise = load(controller.signal)
          .then((result) => {
            controller.signal.throwIfAborted()
            cached = result
            return result
          })
          .finally(() => {
            if (pending?.controller === controller) pending = undefined
          })
        pending = { controller, consumers: 0, promise }
      }
      const request = pending
      request.consumers += 1
      return new Promise<LoadedModelCatalog>((resolve, reject) => {
        let finished = false
        const finish = () => {
          if (finished) return false
          finished = true
          signal.removeEventListener('abort', abort)
          request.consumers -= 1
          if (request.consumers === 0 && pending === request) request.controller.abort()
          return true
        }
        const abort = () => {
          if (finish()) reject(signal.reason)
        }
        signal.addEventListener('abort', abort, { once: true })
        request.promise.then(
          (result) => {
            if (finish()) resolve(result)
          },
          (error: unknown) => {
            if (finish()) reject(error)
          },
        )
      })
    },
  }
}

export const modelCatalogResource = createModelCatalogResource()
