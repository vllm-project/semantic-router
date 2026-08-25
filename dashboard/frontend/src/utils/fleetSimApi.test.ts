import { afterEach, describe, expect, it, vi } from 'vitest'

import { listJobs } from './fleetSimApi'

describe('fleet simulator client', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('accepts list responses', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response('[]', { status: 200, headers: { 'Content-Type': 'application/json' } }),
      ),
    )

    await expect(listJobs()).resolves.toEqual([])
  })

  it('fails malformed list responses without crashing the route', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response('{}', { status: 200, headers: { 'Content-Type': 'application/json' } }),
      ),
    )

    await expect(listJobs()).rejects.toThrow('invalid list response')
  })
})
