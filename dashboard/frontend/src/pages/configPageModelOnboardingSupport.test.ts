import { describe, expect, it } from 'vitest'

import type { ProviderCatalogItem } from '../utils/providerCatalogApi'
import {
  buildModelControlOverrides,
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

  it('validates model control and preserves decimal pricing strings', () => {
    expect(
      buildModelControlOverrides({
        maxRetries: '2',
        retryOn: ['unavailable', 'timeout'],
        requestTimeout: '30s',
        streamTimeout: '5m',
      }),
    ).toEqual({
      retry: { count: 2, on: ['unavailable', 'timeout'] },
      timeout: { request: '30s', stream: '5m' },
    })
    expect(
      buildModelControlOverrides({
        maxRetries: '',
        retryOn: [],
        requestTimeout: '1h30m',
        streamTimeout: '+.5h',
      }),
    ).toEqual({ timeout: { request: '1h30m', stream: '+.5h' } })
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
      buildModelControlOverrides({
        maxRetries: '2',
        retryOn: [],
        requestTimeout: '30 seconds',
        streamTimeout: '',
      }),
    ).toThrow('Request timeout')
    expect(() =>
      buildModelControlOverrides({
        maxRetries: '6',
        retryOn: [],
        requestTimeout: '',
        streamTimeout: '',
      }),
    ).toThrow('0 to 5')
    expect(() =>
      buildModelControlOverrides({
        maxRetries: '',
        retryOn: [],
        requestTimeout: '500ms',
        streamTimeout: '',
      }),
    ).toThrow('1s to 24h')
    expect(() =>
      buildModelPricingOverrides({
        inputCost: '1000000.000000001',
        outputCost: '',
        cacheReadCost: '',
        cacheWriteCost: '',
      }),
    ).toThrow('0 to 1,000,000')
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
          },
        ],
        selectedCatalogItemIds: new Set(['catalog-a', 'catalog-b']),
        namePrefix: 'team',
        control: {
          retry: { count: 2, on: ['unavailable'] },
          timeout: { stream: '5m' },
        },
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
          control: {
            retry: { count: 2, on: ['unavailable'] },
            timeout: { stream: '5m' },
          },
          pricing: {
            inputCostPerMillionTokens: '0.25',
            outputCostPerMillionTokens: '1.00',
          },
        },
        {
          catalogItemId: 'catalog-a',
          name: 'team/vendor/model-a',
          aliases: [],
          loras: [],
          control: {
            retry: { count: 2, on: ['unavailable'] },
            timeout: { stream: '5m' },
          },
          pricing: {
            inputCostPerMillionTokens: '0.25',
            outputCostPerMillionTokens: '1.00',
          },
        },
      ],
    })
  })

  it('omits untouched advanced settings from bulk imports', () => {
    const request = buildRoutingBulkImportRequest({
      provider,
      catalogRevision: `sha256:${'c'.repeat(64)}`,
      discoveryClaim: 'signed-discovery-claim',
      connectionFields: { region: 'global' },
      models: [
        {
          catalogItemId: 'catalog-a',
          providerModelId: 'vendor/model-a',
          displayName: 'Model A',
        },
      ],
      selectedCatalogItemIds: new Set(['catalog-a']),
      namePrefix: '',
    })

    expect(request.selections[0]).not.toHaveProperty('control')
    expect(request.selections[0]).not.toHaveProperty('pricing')
  })
})
