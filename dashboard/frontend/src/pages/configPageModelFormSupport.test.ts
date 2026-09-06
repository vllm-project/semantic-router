import { describe, expect, expectTypeOf, it } from 'vitest'

import type { ProviderModel } from '../types/config'

import {
  buildProviderModelPayload,
  modelReasoningFormData,
  normalizeModelBackendRefs,
  normalizeModelEvaluations,
  normalizeModelPricing,
} from './configPageModelFormSupport'
import { newModelFormData } from './configPageModelsSectionSupport'

describe('model form backend targets', () => {
  it('keeps every canonical API format in the public dashboard config type', () => {
    expectTypeOf<NonNullable<ProviderModel['api_format']>>().toEqualTypeOf<
      'openai' | 'responses' | 'anthropic'
    >()

    const models = [
      { name: 'chat', api_format: 'openai' },
      { name: 'responses', api_format: 'responses' },
      { name: 'messages', api_format: 'anthropic' },
    ] satisfies ProviderModel[]
    expect(models.map((model) => model.api_format)).toEqual(['openai', 'responses', 'anthropic'])
  })

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

  it('preserves an explicitly empty authentication prefix through the model payload', () => {
    const payload = buildProviderModelPayload('private', {
      backend_refs: [
        {
          name: 'raw-key',
          base_url: 'https://provider.example/v1',
          provider: 'custom-provider',
          auth_prefix: '',
        },
      ],
    })

    expect(payload.backend_refs).toEqual([
      {
        name: 'raw-key',
        base_url: 'https://provider.example/v1',
        provider: 'custom-provider',
        auth_prefix: '',
      },
    ])
    expect(Object.prototype.hasOwnProperty.call(payload.backend_refs[0], 'auth_prefix')).toBe(true)
  })

  it('leaves new-model pricing unset until the operator enters a value', () => {
    const form = newModelFormData()

    expect(form).not.toHaveProperty('pricing')
    expect(buildProviderModelPayload('private', form).pricing).toBeUndefined()
  })

  it('preserves explicitly configured zero pricing', () => {
    const pricing = {
      currency: 'USD',
      prompt_per_1m: 0,
      cached_input_per_1m: 0,
      completion_per_1m: 0,
    }

    expect(normalizeModelPricing(pricing)).toEqual(pricing)
    expect(buildProviderModelPayload('private', { pricing }).pricing).toEqual(pricing)
  })

  it('defaults the currency when the operator explicitly enters a zero rate', () => {
    expect(normalizeModelPricing({ prompt_per_1m: 0 })).toEqual({
      prompt_per_1m: 0,
      currency: 'USD',
    })
  })
})

describe('model form catalog and reasoning payloads', () => {
  it('emits catalog identity and leaves built-in reasoning to the catalog', () => {
    expect(
      buildProviderModelPayload('frontier', {
        catalog: 'vendor/frontier-v1',
        reasoning_family: 'gpt',
        provider_model_id: 'frontier-v1',
        backend_refs: [{ name: 'primary', base_url: 'https://api.example/v1', provider: 'vendor' }],
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

  it('does not synthesize an alias as the upstream ID for a catalog model', () => {
    const model = buildProviderModelPayload('production-alias', {
      catalog: 'openai/gpt-5.6-sol',
      provider_model_id: '',
    })

    expect(model.provider_model_id).toBeUndefined()
  })

  it('preserves an explicit upstream override for a catalog model', () => {
    expect(
      buildProviderModelPayload('production-alias', {
        catalog: 'vendor/frontier-v1',
        provider_model_id: 'frontier-custom-endpoint',
      }).provider_model_id,
    ).toBe('frontier-custom-endpoint')
  })

  it('emits custom reasoning only when catalog is absent', () => {
    expect(
      buildProviderModelPayload('private', {
        reasoning_family: 'qwen3',
      }),
    ).toMatchObject({ name: 'private', reasoning: { family: 'qwen3' } })
  })

  it('round-trips complete inline reasoning fields for custom models', () => {
    const reasoning = {
      type: 'reasoning_effort',
      parameter: 'reasoning_effort',
      activation_parameter: 'enable_thinking',
      levels: ['disabled', 'low', 'high'],
      default: 'low',
      disabled: 'disabled',
    }

    const payload = buildProviderModelPayload('private', modelReasoningFormData(reasoning))

    expect(payload.reasoning).toEqual(reasoning)
  })
})

describe('model form evaluations', () => {
  it('normalizes open benchmark metrics without imposing a fixed benchmark schema', () => {
    expect(
      normalizeModelEvaluations([
        {
          benchmark: 'acme/support@1',
          benchmark_profile: ' production-standard ',
          reasoning_effort: ' high ',
          metrics: { resolution_rate: '0.82', invalid: 'not-a-number' },
          metadata: { runtime: 'vllm', tensor_parallel: 2 },
        },
      ]),
    ).toEqual([
      {
        benchmark: 'acme/support@1',
        benchmark_profile: 'production-standard',
        reasoning_effort: 'high',
        metrics: { resolution_rate: 0.82 },
        metadata: { runtime: 'vllm', tensor_parallel: 2 },
      },
    ])
  })
})
