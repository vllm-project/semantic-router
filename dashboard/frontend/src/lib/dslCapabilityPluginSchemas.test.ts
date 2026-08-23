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

  it.each(['semantic-cache', 'semantic_cache', 'response-cache'])(
    'does not expose a schema for removed alias %s',
    (pluginType) => {
      expect(getCapabilityPluginFieldSchema(pluginType)).toBeNull()
    },
  )
})
