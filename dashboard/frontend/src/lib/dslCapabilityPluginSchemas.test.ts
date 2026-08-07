import { describe, expect, it } from 'vitest'

import { getCapabilityPluginFieldSchema } from './dslCapabilityPluginSchemas'

describe('capability plugin field schemas', () => {
  it('exposes the complete context compression contract', () => {
    const fields = getCapabilityPluginFieldSchema('context_compression')
    expect(fields?.map((field) => field.key)).toEqual([
      'enabled',
      'min_tokens',
      'target_tokens',
      'compress_rag',
      'bypass_header',
    ])
    expect(fields?.find((field) => field.key === 'compress_rag')?.type).toBe('boolean')
  })

  it('exposes canonical response cache fields', () => {
    const fields = getCapabilityPluginFieldSchema('response_cache')
    expect(fields?.map((field) => field.key)).toEqual([
      'enabled',
      'mode',
      'scope',
      'semantic',
      'ttl_seconds',
      'request_controls',
      'personalized',
      'revision',
    ])
  })
})
