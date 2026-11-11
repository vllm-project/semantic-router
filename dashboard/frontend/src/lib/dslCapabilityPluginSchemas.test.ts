import { describe, expect, it } from 'vitest'

import { resolveCapabilityPluginFieldSchema } from './dslCapabilityPluginSchemas'
import { getPluginFieldSchema } from './dslSchemas'
import { pluginFieldsFromRouterSchema } from './routerConfigSchema'

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

  it('uses the generated Replay schema for the decision editor', () => {
    expect(resolveCapabilityPluginFieldSchema('shadow_dispatch')).toBeNull()
    const fields = getPluginFieldSchema('router_replay')
    expect(fields).toEqual(pluginFieldsFromRouterSchema('router_replay'))
    const shadow = fields.find((field) => field.key === 'shadow')?.fields
    expect(shadow?.map(({ key, type }) => [key, type])).toEqual([
      ['models', 'string[]'],
      ['sample_rate', 'number'],
      ['execution', 'object'],
      ['budget', 'object'],
      ['capture', 'object'],
      ['forward_headers', 'string[]'],
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
