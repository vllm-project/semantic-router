import { describe, expect, it } from 'vitest'

import {
  buildProviderModelPayload,
  normalizeModelBackendRefs,
  normalizeModelEvaluations,
} from './configPageModelFormSupport'

describe('model form backend targets', () => {
  it('preserves every canonical backend target field', () => {
    const backend = {
      name: 'hosted-primary',
      endpoint: 'provider.internal:8443',
      protocol: 'https',
      weight: 75,
      base_url: 'https://provider.example/v1',
      provider: 'openai',
      auth_header: 'Authorization',
      auth_prefix: 'Bearer',
      extra_headers: { 'X-Tenant': 'production' },
      api_version: '2026-09-01',
      chat_path: '/chat/completions',
      api_key: 'test-only-key',
      api_key_env: 'PROVIDER_API_KEY',
    } as const

    expect(normalizeModelBackendRefs([backend])).toEqual([backend])
  })

  it('emits catalog identity and leaves built-in reasoning to the catalog', () => {
    expect(
      buildProviderModelPayload('frontier', {
        catalog: 'vendor/frontier-v1',
        reasoning_family: 'gpt',
        provider_model_id: 'frontier-v1',
        backend_refs: [
          { name: 'primary', base_url: 'https://api.example/v1', provider: 'vendor' },
        ],
      }),
    ).toMatchObject({
      name: 'frontier',
      catalog: 'vendor/frontier-v1',
    })
    expect(
      buildProviderModelPayload('frontier', {
        catalog: 'vendor/frontier-v1',
        reasoning_family: 'gpt',
      }).reasoning,
    ).toBeUndefined()
  })

  it('emits custom reasoning only when catalog is absent', () => {
    expect(
      buildProviderModelPayload('private', {
        reasoning_family: 'qwen3',
      }),
    ).toMatchObject({ name: 'private', reasoning: { family: 'qwen3' } })
  })

  it('normalizes open benchmark metrics without imposing a fixed benchmark schema', () => {
    expect(
      normalizeModelEvaluations([
        {
          benchmark: 'acme/support@1',
          metrics: { resolution_rate: '0.82', invalid: 'not-a-number' },
          metadata: { runtime: 'vllm', tensor_parallel: 2 },
        },
      ]),
    ).toEqual([
      {
        benchmark: 'acme/support@1',
        metrics: { resolution_rate: 0.82 },
        metadata: { runtime: 'vllm', tensor_parallel: 2 },
      },
    ])
  })
})
