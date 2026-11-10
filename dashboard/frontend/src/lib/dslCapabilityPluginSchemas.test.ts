import { describe, expect, it } from 'vitest'

import { resolveCapabilityPluginFieldSchema } from './dslCapabilityPluginSchemas'
import { getCuratedPluginFieldSchema } from './dslPluginSchemas'

describe('capability plugin field schemas', () => {
  it('exposes the complete context contract', () => {
    const fields = resolveCapabilityPluginFieldSchema('context')
    expect(fields?.map((field) => field.key)).toEqual([
      'enabled',
      'sources',
      'compress',
      'budget',
      'injection',
    ])
    const sources = fields?.find((field) => field.key === 'sources')
    expect(sources?.type).toBe('object[]')
    expect(sources?.fields?.map((field) => field.key)).toContain('match')
    const compress = fields?.find((field) => field.key === 'compress')
    expect(compress?.fields?.map((field) => field.key)).toEqual([
      'mode',
      'targets',
      'match',
      'recovery',
      'request_controls',
      'failure_mode',
      'signals',
    ])
    expect(resolveCapabilityPluginFieldSchema('context_compression')).toBeNull()
  })

  it('exposes the shadow bounds under router_replay', () => {
    const shadow = getCuratedPluginFieldSchema('router_replay').find(
      (field) => field.key === 'shadow',
    )
    expect(resolveCapabilityPluginFieldSchema('shadow_dispatch')).toBeNull()
    expect(shadow?.fields?.map((field) => field.key)).toEqual([
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
      'forward_headers',
    ])
  })

  it('exposes canonical response cache fields', () => {
    const fields = resolveCapabilityPluginFieldSchema('response_cache')
    expect(fields?.map((field) => field.key)).toEqual([
      'enabled',
      'mode',
      'scope',
      'ttl_seconds',
      'semantic',
      'request_controls',
      'personalized',
      'revision',
    ])
    const semantic = fields?.find((field) => field.key === 'semantic')
    expect(semantic?.fields?.map((field) => field.key)).toEqual(['match'])
  })

  it('exposes the prompt cache marker contract', () => {
    const fields = resolveCapabilityPluginFieldSchema('prompt_cache')
    expect(fields?.map((field) => field.key)).toEqual([
      'enabled',
      'ttl',
      'targets',
      'on_unsupported',
    ])
  })

  it.each(['constructor', 'toString', '__proto__'])('rejects prototype key %s', (pluginType) => {
    expect(resolveCapabilityPluginFieldSchema(pluginType)).toBeNull()
  })
})
