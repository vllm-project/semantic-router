import { afterEach, describe, expect, it, vi } from 'vitest'

import { ModelVerificationApiError, verifyProviderModel } from './modelVerificationApi'

const verifiedPayload = {
  verified: true,
  model: 'logical-model',
  providerModel: 'provider/real-model',
  backend: 'logical-model/primary',
  provider: 'openai',
  summary: 'OK',
  verifiedAt: '2026-08-15T05:06:07Z',
  latencyMs: 18,
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('live provider model verification API', () => {
  it('posts only the selected logical model and accepts generated-text evidence', async () => {
    const fetchMock = vi.fn(
      async () =>
        new Response(JSON.stringify(verifiedPayload), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }),
    )
    vi.stubGlobal('fetch', fetchMock)

    await expect(verifyProviderModel('logical-model')).resolves.toEqual(verifiedPayload)
    expect(fetchMock).toHaveBeenCalledWith('/api/models/verify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: 'logical-model' }),
      signal: undefined,
    })
  })

  it('preserves a safe server error for retry UX', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(
        async () =>
          new Response(
            JSON.stringify({
              error: 'provider_rejected',
              message: 'Provider inference returned HTTP 401.',
            }),
            { status: 502 },
          ),
      ),
    )

    await expect(verifyProviderModel('logical-model')).rejects.toMatchObject({
      name: 'ModelVerificationApiError',
      code: 'provider_rejected',
      status: 502,
      message: 'Provider inference returned HTTP 401.',
    } satisfies Partial<ModelVerificationApiError>)
  })

  it('fails closed when success evidence is malformed or names another model', async () => {
    for (const payload of [
      { ...verifiedPayload, verified: false },
      { ...verifiedPayload, model: 'different-model' },
      { ...verifiedPayload, summary: '' },
      { ...verifiedPayload, summary: '   ' },
      { ...verifiedPayload, verifiedAt: 'not-a-date' },
    ]) {
      vi.stubGlobal(
        'fetch',
        vi.fn(async () => new Response(JSON.stringify(payload), { status: 200 })),
      )
      await expect(verifyProviderModel('logical-model')).rejects.toMatchObject({
        code: 'invalid_verification_contract',
      })
    }
  })
})
