import { describe, expect, it } from 'vitest'

import { normalizeModelBackendRefs } from './configPageModelFormSupport'

describe('model form backend targets', () => {
  it('preserves every canonical backend target field', () => {
    const backend = {
      name: 'hosted-primary',
      endpoint: 'provider.internal:8443',
      protocol: 'https',
      weight: 75,
      type: 'openai',
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
})
