import { afterEach, describe, expect, it, vi } from 'vitest'
import catalogDocument from '../modelCatalogDocument'
import type { BuiltInModelCatalog } from '../types/modelCatalog'
import { createModelCatalogResource, type LoadedModelCatalog } from './modelCatalogResource'
const catalog = catalogDocument as unknown as BuiltInModelCatalog
const loaded: LoadedModelCatalog = { catalog, source: 'server', error: null }
const caller = () => new AbortController()
afterEach(() => {
  vi.unstubAllGlobals()
  vi.useRealTimers()
})
describe('shared model catalog resource', () => {
  it('does no work before a consumer, deduplicates reads and reuses loaded data across routes', async () => {
    let resolve!: (value: LoadedModelCatalog) => void
    const load = vi.fn(
      () =>
        new Promise<LoadedModelCatalog>((done) => {
          resolve = done
        }),
    )
    const resource = createModelCatalogResource(load)
    expect(load).not.toHaveBeenCalled()
    const first = resource.read(caller().signal)
    const second = resource.read(caller().signal)
    expect(load).toHaveBeenCalledTimes(1)
    resolve(loaded)
    await expect(first).resolves.toBe(loaded)
    await expect(second).resolves.toBe(loaded)
    await expect(resource.read(caller().signal)).resolves.toBe(loaded)
    expect(load).toHaveBeenCalledTimes(1)
    expect(resource.peek()).toBe(loaded)
  })
  it('keeps shared work for remaining consumers and aborts it when the last route leaves', async () => {
    let child!: AbortSignal
    const resource = createModelCatalogResource((signal) => {
      child = signal
      return new Promise((_resolve, reject) =>
        signal.addEventListener('abort', () => reject(signal.reason)),
      )
    })
    const one = caller(),
      two = caller()
    const first = expect(resource.read(one.signal)).rejects.toMatchObject({ name: 'AbortError' })
    const second = expect(resource.read(two.signal)).rejects.toMatchObject({ name: 'AbortError' })
    one.abort()
    await first
    expect(child.aborted).toBe(false)
    two.abort()
    await second
    expect(child.aborted).toBe(true)
    expect(resource.peek()).toBeUndefined()
  })
  it('allows explicit server retry after a cached bundled fallback', async () => {
    const fallback: LoadedModelCatalog = { ...loaded, source: 'bundled', error: 'HTTP 503' }
    const load = vi.fn().mockResolvedValueOnce(fallback).mockResolvedValueOnce(loaded)
    const resource = createModelCatalogResource(load)
    await expect(resource.read(caller().signal)).resolves.toBe(fallback)
    await expect(resource.read(caller().signal, true)).resolves.toBe(loaded)
    expect(load).toHaveBeenCalledTimes(2)
  })
  it('fetches the generated fallback as a separate asset only if the API fails', async () => {
    const fetcher = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(new Response('{}', { status: 503 }))
      .mockResolvedValueOnce(Response.json(catalog))
    vi.stubGlobal('fetch', fetcher)
    const result = await createModelCatalogResource().read(caller().signal)
    expect(result.source).toBe('bundled')
    expect(result.error).toContain('HTTP 503')
    expect(result.catalog).toEqual(catalog)
    expect(fetcher).toHaveBeenCalledTimes(2)
    expect(fetcher.mock.calls[0][0]).toBe('/api/models/catalog')
    expect(fetcher.mock.calls[1][0]).toMatch(/catalog.*\.json/)
  })
  it('does not download a second catalog when the server responds successfully', async () => {
    const fetcher = vi.fn(async () => Response.json(catalog))
    vi.stubGlobal('fetch', fetcher)
    expect((await createModelCatalogResource().read(caller().signal)).source).toBe('server')
    expect(fetcher).toHaveBeenCalledTimes(1)
  })
  it('recovers a stalled API request using a bounded static fallback', async () => {
    vi.useFakeTimers()
    const fetcher = vi
      .fn<typeof fetch>()
      .mockImplementationOnce(
        (_url, init) =>
          new Promise((_resolve, reject) => {
            init?.signal?.addEventListener('abort', () => reject(init.signal?.reason))
          }),
      )
      .mockResolvedValueOnce(Response.json(catalog))
    vi.stubGlobal('fetch', fetcher)
    const pending = createModelCatalogResource().read(caller().signal)
    await vi.advanceTimersByTimeAsync(15_000)
    expect((await pending).source).toBe('bundled')
    expect(vi.getTimerCount()).toBe(0)
  })
})
