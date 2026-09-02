import { describe, expect, it } from 'vitest'

import { getCapabilityPluginFieldSchema } from './dslCapabilityPluginSchemas'

describe('capability plugin field schemas', () => {
  it('exposes the complete context compression contract', () => {
    const fields = getCapabilityPluginFieldSchema('context_compression')
    expect(fields?.map((field) => field.key)).toEqual([
      'enabled',
      'mode',
      'budget',
      'targets',
      'scoring',
      'recovery',
      'request_controls',
      'failure_mode',
    ])
    expect(fields?.find((field) => field.key === 'targets')?.type).toBe('object')
  })

  it('exposes the shadow dispatch bounds', () => {
    const fields = getCapabilityPluginFieldSchema('shadow_dispatch')
    expect(fields?.map((field) => field.key)).toEqual([
      'enabled',
      'model',
      'sample_rate',
      'max_concurrency',
      'max_queue_depth',
      'timeout_seconds',
      'max_response_bytes',
      'max_retries',
      'capture_response_body',
      'max_capture_bytes',
      'tls_skip_verify',
    ])
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
