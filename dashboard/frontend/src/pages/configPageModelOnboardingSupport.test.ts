import { describe, expect, it } from 'vitest'

import type { ProviderCatalogItem } from '../utils/providerCatalogApi'
import {
  buildModelExecutionOverrides,
  buildModelPricingOverrides,
  buildRoutingBulkImportRequest,
  initialProviderConnectionFields,
  validatedProviderConnectionFields,
} from './configPageModelOnboardingSupport'

const provider: ProviderCatalogItem = {
  providerId: 'example',
  revision: `sha256:${'a'.repeat(64)}`,
  display: {
    name: 'Example',
    description: 'Example models.',
    category: 'Model APIs',
    icon: { source: 'lobe', value: 'example', color: false },
  },
  credential: { mode: 'required' },
  origin: { mode: 'fixed', defaultUrl: 'https://example.test', baseUrlRequired: false },
  discoverySupported: true,
  capabilities: [],
  interfaces: [
    { id: 'chat', label: 'Chat Completions', default: true, capabilities: ['text'] },
    { id: 'responses', label: 'Responses API', default: false, capabilities: ['text'] },
  ],
  connectionFields: [
    {
      name: 'region',
      label: 'Region',
      kind: 'select',
      required: true,
      advanced: false,
      options: [{ value: 'global', label: 'Global' }],
    },
    {
      name: 'replicas',
      label: 'Replicas',
      kind: 'integer',
      required: false,
      advanced: true,
    },
    {
      name: 'private',
      label: 'Private',
      kind: 'boolean',
      required: false,
      advanced: true,
      default: 'true',
    },
  ],
}

describe('model onboarding form contract', () => {
  it('initializes and validates only catalog-declared typed fields', () => {
    const initial = initialProviderConnectionFields(provider)
    expect(initial).toEqual({ region: 'global', replicas: '', private: true })
    expect(
      validatedProviderConnectionFields(provider, {
        ...initial,
        replicas: '3',
        undeclared: 'ignored',
      }),
    ).toEqual({ region: 'global', replicas: 3, private: true })
  })

  it('validates execution durations and preserves decimal pricing strings', () => {
    expect(
      buildModelExecutionOverrides({
        maxRetries: '2',
        requestTimeout: '30s',
        streamTimeout: '5m',
      }),
    ).toEqual({ maxRetries: 2, requestTimeout: '30s', streamTimeout: '5m' })
    expect(
      buildModelPricingOverrides({
        inputCost: '0.25',
        outputCost: '1.00',
        cacheReadCost: '',
        cacheWriteCost: '0.30',
      }),
    ).toEqual({
      inputCostPerMillionTokens: '0.25',
      outputCostPerMillionTokens: '1.00',
      cacheWriteCostPerMillionTokens: '0.30',
    })
    expect(() =>
      buildModelExecutionOverrides({
        maxRetries: '2',
        requestTimeout: '30 seconds',
        streamTimeout: '',
      }),
    ).toThrow('Request timeout')
    expect(() =>
      buildModelExecutionOverrides({
        maxRetries: '6',
        requestTimeout: '',
        streamTimeout: '',
      }),
    ).toThrow('0 to 5')
    expect(() =>
      buildModelExecutionOverrides({
        maxRetries: '',
        requestTimeout: '500ms',
        streamTimeout: '',
      }),
    ).toThrow('1s to 24h')
  })

  it('builds the canonical bulk-import command in signed discovery order', () => {
    expect(
      buildRoutingBulkImportRequest({
        provider,
        interfaceId: ' responses ',
        catalogRevision: `sha256:${'b'.repeat(64)}`,
        discoveryClaim: 'signed-discovery-claim',
        credentialId: '  credential-id  ',
        baseUrl: 'https://ignored-for-fixed-provider.test',
        connectionFields: { region: 'global' },
        models: [
          {
            catalogItemId: 'catalog-b',
            providerModelId: 'vendor/model-b',
            displayName: 'Model B',
            capabilities: ['tools'],
          },
          {
            catalogItemId: 'catalog-a',
            providerModelId: 'vendor/model-a',
            displayName: 'Model A',
            capabilities: ['text'],
          },
        ],
        selectedCatalogItemIds: new Set(['catalog-a', 'catalog-b']),
        namePrefix: 'team',
        execution: { maxRetries: 2, streamTimeout: '5m' },
        pricing: {
          inputCostPerMillionTokens: '0.25',
          outputCostPerMillionTokens: '1.00',
        },
      }),
    ).toEqual({
      providerId: 'example',
      interfaceId: 'responses',
      catalogRevision: `sha256:${'b'.repeat(64)}`,
      discoveryClaim: 'signed-discovery-claim',
      credentialId: 'credential-id',
      connectionFields: { region: 'global' },
      weight: '1',
      selections: [
        {
          catalogItemId: 'catalog-b',
          name: 'team/vendor/model-b',
          aliases: [],
          capabilities: ['tools'],
          loras: [],
          execution: { maxRetries: 2, requestTimeout: '300s', streamTimeout: '5m' },
          pricing: {
            inputCostPerMillionTokens: '0.25',
            outputCostPerMillionTokens: '1.00',
            cacheReadCostPerMillionTokens: null,
            cacheWriteCostPerMillionTokens: null,
          },
        },
        {
          catalogItemId: 'catalog-a',
          name: 'team/vendor/model-a',
          aliases: [],
          capabilities: ['text'],
          loras: [],
          execution: { maxRetries: 2, requestTimeout: '300s', streamTimeout: '5m' },
          pricing: {
            inputCostPerMillionTokens: '0.25',
            outputCostPerMillionTokens: '1.00',
            cacheReadCostPerMillionTokens: null,
            cacheWriteCostPerMillionTokens: null,
          },
        },
      ],
    })
  })
})
