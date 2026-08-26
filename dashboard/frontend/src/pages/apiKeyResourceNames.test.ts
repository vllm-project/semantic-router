import { describe, expect, it } from 'vitest'
import type { RoutingCatalog } from '../generated/managementApiContract'
import type { AccessResourceRef } from '../utils/inferenceAccessApi'
import {
  API_KEY_MODEL_PLACEHOLDER,
  apiKeyQuickstartModel,
  apiKeyResourceResolutions,
  apiKeyVisibleResourceName,
} from './apiKeyResourceNames'

const resources: AccessResourceRef[] = [
  { resourceType: 'model', resourceId: 'mdl_internal_123' },
  { resourceType: 'entrypoint', resourceId: 'ep_internal_456' },
]

const catalog = {
  keyId: 'key-1',
  policyRevision: 1,
  policyDigest: 'policy-digest',
  routingRevision: 1,
  routingDigest: 'routing-digest',
  models: [
    {
      aliases: [],
      capabilities: [],
      id: 'mdl_internal_123',
      loras: [],
      name: 'local/fast',
      pricing: {},
      revision: 1,
      tags: [],
    },
  ],
  recipes: [],
  entrypoints: [
    {
      aliases: [],
      id: 'ep_internal_456',
      name: 'vllm-sr/blend',
      revision: 1,
      rules: [],
    },
  ],
} satisfies RoutingCatalog

describe('API key visible resource names', () => {
  it('uses the key-scoped routing catalog without global routing access', () => {
    const resolutions = apiKeyResourceResolutions(resources, catalog)

    expect(apiKeyVisibleResourceName(resources[0], resolutions)).toBe('local/fast')
    expect(apiKeyVisibleResourceName(resources[1], resolutions)).toBe('vllm-sr/blend')
    expect(apiKeyQuickstartModel(resources, resolutions)).toBe('vllm-sr/blend')
  })

  it('never falls back to an internal resource id', () => {
    const resolutions = apiKeyResourceResolutions(resources, null)

    expect(apiKeyVisibleResourceName(resources[0], resolutions)).toBe('Unavailable model')
    expect(apiKeyVisibleResourceName(resources[1], resolutions)).toBe(
      'Unavailable Mixture-of-Model',
    )
    expect(apiKeyQuickstartModel(resources, resolutions)).toBe(API_KEY_MODEL_PLACEHOLDER)
    const visibleNames = resources.map((resource) =>
      apiKeyVisibleResourceName(resource, resolutions),
    )
    expect(visibleNames).not.toContain('mdl_internal_123')
    expect(visibleNames).not.toContain('ep_internal_456')
  })

  it('shows a neutral loading label until the catalog arrives', () => {
    expect(apiKeyVisibleResourceName(resources[0], {})).toBe('Loading…')
    expect(apiKeyQuickstartModel(resources, {})).toBe(API_KEY_MODEL_PLACEHOLDER)
  })
})
