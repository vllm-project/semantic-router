import { afterEach, describe, expect, it, vi } from 'vitest'
import { dslCompiler } from './dslCompiler'

describe('page-owned compiler transport', () => {
  afterEach(() => {
    dslCompiler.cancelPending()
    vi.unstubAllGlobals()
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
