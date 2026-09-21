import { describe, expect, it } from 'vitest'

import { modelProviderCatalog } from './modelProviderCatalog'
import {
  DEFAULT_SETUP_PROVIDER_ID,
  getSetupProviderOption,
  isSetupRuntimeProvider,
  SETUP_PROVIDER_IDS,
  SETUP_PROVIDER_OPTIONS,
} from './setupWizardProviderCatalog'

describe('setup wizard provider catalog', () => {
  it('projects its onboarding subset from the shared generated provider catalog', () => {
    expect(SETUP_PROVIDER_OPTIONS.map((provider) => provider.id)).toEqual(SETUP_PROVIDER_IDS)

    for (const option of SETUP_PROVIDER_OPTIONS) {
      const catalogProvider = modelProviderCatalog.find((provider) => provider.id === option.id)
      expect(catalogProvider).toBeDefined()
      expect(option).toMatchObject({
        label: catalogProvider?.name,
        description: catalogProvider?.description,
        apiFormat: catalogProvider?.apiFormat,
        supportTier: catalogProvider?.supportTier,
      })
    }
  })

  it('derives runtime and wire behavior instead of maintaining provider conditionals', () => {
    expect(DEFAULT_SETUP_PROVIDER_ID).toBe('vllm')
    expect(isSetupRuntimeProvider('vllm')).toBe(true)
    expect(isSetupRuntimeProvider('openai-compatible')).toBe(false)
    expect(getSetupProviderOption('anthropic').apiFormat).toBe('anthropic')
    expect(getSetupProviderOption('anthropic').initialBaseUrl).toBe('https://api.anthropic.com')
  })
})
