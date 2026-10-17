import { afterEach, describe, expect, it, vi } from 'vitest'
import { dslCompiler } from './dslCompiler'

describe('page-owned compiler transport', () => {
  afterEach(() => {
    dslCompiler.cancelPending()
    vi.unstubAllGlobals()
  })

  it('sends enclosing config and keeps distinct rule budgets out of the same cache entry', async () => {
    const resolve: Array<(value: Response) => void> = []
    const fetchMock = vi.fn((_path, _options: RequestInit) => new Promise<Response>((yes) => resolve.push(yes)))
    vi.stubGlobal('fetch', fetchMock)
    const baseYaml = 'global:\n  router:\n    decision_rule_limits:\n      max_depth: 32\n'
    const fragment = dslCompiler.compile('draft')
    const document = dslCompiler.compile('draft', baseYaml)
    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(JSON.parse(fetchMock.mock.calls[1][1].body as string)).toEqual({ source: 'draft', baseYaml })
    resolve.forEach((yes) => yes(new Response(JSON.stringify({ yaml: 'routing: {}', diagnostics: [] }))))
    await Promise.all([fragment, document])
  })

  it('aborts requests on departure and preserves deduplication for the next visit', async () => {
    const signals: AbortSignal[] = []
    const resolve: Array<(value: Response) => void> = []
    vi.stubGlobal(
      'fetch',
      vi.fn(
        (_path, options: RequestInit) =>
          new Promise<Response>((yes, no) => {
            const signal = options.signal as AbortSignal
            signals.push(signal)
            resolve.push(yes)
            signal.addEventListener('abort', () => no(new DOMException('Cancelled', 'AbortError')))
          }),
      ),
    )
    const old = dslCompiler.compile('draft')
    const aborted = expect(old).rejects.toMatchObject({ name: 'AbortError' })
    dslCompiler.cancelPending()
    expect(signals[0].aborted).toBe(true)
    const current = dslCompiler.compile('draft')
    await aborted
    expect(dslCompiler.compile('draft')).toBe(current)
    expect(fetch).toHaveBeenCalledTimes(2)
    resolve[1](new Response(JSON.stringify({ yaml: 'current', diagnostics: [] })))
    await expect(current).resolves.toMatchObject({ yaml: 'current' })
  })
})
