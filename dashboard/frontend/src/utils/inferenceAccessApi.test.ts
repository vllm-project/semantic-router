import { afterEach, describe, expect, it, vi } from 'vitest'
import { inferenceAccessApi } from './inferenceAccessApi'

afterEach(() => vi.unstubAllGlobals())

describe('inference access API contract', () => {
  it('uses user resources, pagination, and scoped usage query fields', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ items: [], series: [] }),
    })
    vi.stubGlobal('fetch', fetchMock)

    await inferenceAccessApi.users({ limit: 20, offset: 20 })
    await inferenceAccessApi.usage({ userId: 'user-a', teamId: 'team-a', keyId: 'key-a' })

    expect(fetchMock.mock.calls[0][0]).toContain('/api/v1/access-control/users')
    expect(fetchMock.mock.calls[0][0]).toContain('offset=20')
    expect(fetchMock.mock.calls[1][0]).toContain('userId=user-a')
    expect(fetchMock.mock.calls[1][0]).toContain('teamId=team-a')
    expect(fetchMock.mock.calls[1][0]).toContain('keyId=key-a')
  })

  it('exposes deep key controls and request log details', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({}),
    })
    vi.stubGlobal('fetch', fetchMock)

    await inferenceAccessApi.key('key/a')
    await inferenceAccessApi.keySecret('key/a')
    await inferenceAccessApi.rotateKey('key/a')
    await inferenceAccessApi.deleteKey('key/a')
    await inferenceAccessApi.requestLog('log/a')

    expect(fetchMock.mock.calls.map(([path]) => path)).toEqual([
      '/api/v1/access-control/api-keys/key%2Fa',
      '/api/v1/access-control/api-keys/key%2Fa/secret',
      '/api/v1/access-control/api-keys/key%2Fa/rotate',
      '/api/v1/access-control/api-keys/key%2Fa',
      '/api/v1/access-control/request-logs/log%2Fa',
    ])
    expect(fetchMock.mock.calls[2][1]).toMatchObject({ method: 'POST' })
    expect(fetchMock.mock.calls[3][1]).toMatchObject({ method: 'DELETE' })
  })

  it('sends linked budgets and direct key limits through create and edit', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) })
    vi.stubGlobal('fetch', fetchMock)
    const policy = {
      name: 'Production',
      budgetId: 'shared-budget',
      budget: { rpm: 30, tpm: 12_000, dailyTokens: 1_000_000 },
    }

    await inferenceAccessApi.createKey(policy)
    await inferenceAccessApi.saveKey({ id: 'key-1', ...policy })

    expect(fetchMock.mock.calls[0][1]).toMatchObject({
      method: 'POST',
      body: JSON.stringify(policy),
    })
    expect(fetchMock.mock.calls[1][0]).toBe('/api/v1/access-control/api-keys/key-1')
    expect(fetchMock.mock.calls[1][1]).toMatchObject({
      method: 'PUT',
      body: JSON.stringify({ id: 'key-1', ...policy }),
    })
  })
})
