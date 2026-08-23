import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const readSource = (name: string) => readFileSync(new URL(name, import.meta.url), 'utf8')

describe('provider catalog model onboarding', () => {
  it('renders provider identity and fields from the Management catalog', () => {
    const dialog = readSource('./ConfigPageAddModelsDialog.tsx')
    const picker = readSource('./ConfigPageModelProviderPicker.tsx')

    expect(dialog).toContain('listProviderCatalog')
    expect(dialog).toContain('getProviderCatalogDetail')
    expect(dialog).toContain('connectionFields.filter')
    expect(picker).toContain('ProviderCatalogItem')
    expect(picker).toContain('provider.display.icon')
  })

  it('keeps raw credentials out of discovery and signed selection out of connection overrides', () => {
    const dialog = readSource('./ConfigPageAddModelsDialog.tsx')
    const discoveryClient = readSource('../utils/providerCatalogApi.ts')
    const onboarding = readSource('./configPageModelOnboardingSupport.ts')

    expect(dialog).toContain('createProviderCredential')
    expect(dialog).toContain('credentialId: resolvedCredentialId')
    expect(discoveryClient).not.toContain('apiKey')
    expect(onboarding).toContain('discoveryClaim')
    expect(onboarding).toContain('selections')
    expect(onboarding).toContain('RoutingBulkImportRequest')
    for (const forbidden of [
      'authHeader',
      'authPrefix',
      'chatPath',
      'modelsPath',
      'extraHeaders',
      'protocolAdapterId',
    ]) {
      expect(dialog).not.toContain(forbidden)
    }
  })

  it('offers the complete safe execution and pricing overrides', () => {
    const advanced = readSource('./ConfigPageModelAdvancedOptions.tsx')
    expect(advanced).toContain('API style')
    for (const label of [
      'Max retries',
      'Request timeout',
      'Stream timeout',
      'Input cost',
      'Output cost',
      'Cache read cost',
      'Cache write cost',
    ]) {
      expect(advanced).toContain(label)
    }
  })

  it('keeps both the decision list and each model picker independently scrollable', () => {
    const styles = readSource('./ConfigPageMixtureDialog.module.css')

    expect(styles).toMatch(/\.assignments\s*{[^}]*max-height:[^}]*overflow-y: auto;/s)
    expect(styles).toMatch(/\.modelPicker\s*{[^}]*max-height:[^}]*overflow-y: auto;/s)
  })

  it('keeps discovery actions explicit, semantic, and accessible', () => {
    const results = readSource('./ConfigPageModelDiscoveryResults.tsx')

    expect(results).toContain('<ProductIcon name="search"')
    expect(results).toContain('aria-pressed={allSelected}')
    expect(results).toContain("name={allSelected ? 'close' : 'check'}")
    expect(results).toContain('<ProductIcon name="chevron-down"')
    expect(results).not.toContain('Next models')
  })
})
