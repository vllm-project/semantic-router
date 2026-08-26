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

  it('offers complete, structured model control and pricing overrides', () => {
    const advanced = readSource('./ConfigPageModelAdvancedOptions.tsx')
    const modelEditor = readSource('./ConfigPageModelsSection.tsx')
    const styles = readSource('./ConfigPageAddModelsDialog.module.css')
    expect(advanced).toContain('API style')
    for (const label of [
      'Max retries',
      'Retry on',
      'Request timeout',
      'Stream timeout',
      'Input cost',
      'Output cost',
      'Cache read cost',
      'Cache write cost',
    ]) {
      expect(advanced).toContain(label)
    }
    expect(advanced).toContain('MODEL_RETRY_TRIGGERS.map')
    expect(advanced).toContain('type="checkbox"')
    expect(advanced).not.toContain('placeholder="unavailable"')
    expect(modelEditor).toMatch(
      /name: 'retryOn',[\s\S]*?type: 'multiselect',[\s\S]*?options: \['unavailable', 'timeout'\]/,
    )
    for (const field of [
      'paramSize',
      'contextWindowSize',
      'description',
      'qualityScore',
      'modality',
      'tags',
    ]) {
      expect(modelEditor).toContain(`name: '${field}'`)
    }
    expect(styles).toMatch(
      /@media \(max-width: 760px\)[\s\S]*?\.retryChoices\s*{[\s\S]*?grid-template-columns: 1fr;/,
    )
  })

  it('keeps the dialog body as the single model-assignment scroll owner', () => {
    const styles = readSource('./ConfigPageMixtureDialog.module.css')

    expect(styles).toMatch(/\.body\s*{[^}]*overflow-y: auto;/s)
    expect(styles).not.toMatch(/\.assignments\s*{[^}]*overflow-y:/s)
    expect(styles).not.toMatch(/\.modelPicker\s*{[^}]*overflow-y:/s)
  })

  it('keeps discovery actions explicit, semantic, and accessible', () => {
    const results = readSource('./ConfigPageModelDiscoveryResults.tsx')

    expect(results).toContain('<ProductIcon name="search"')
    expect(results).toContain('aria-pressed={allSelected}')
    expect(results).toContain("name={allSelected ? 'close' : 'check'}")
    expect(results).toContain('<ProductIcon name="chevron-down"')
    expect(results).not.toContain('Next models')
  })

  it('keeps model onboarding inside the dynamic mobile viewport', () => {
    const styles = readSource('./ConfigPageAddModelsDialog.module.css')

    expect(styles).toContain('width: min(var(--product-dialog-content-width), 100%);')
    expect(styles).toContain('max-height: min(900px, calc(100dvh - 48px));')
    expect(styles).toContain('max-height: calc(100dvh - 0.75rem);')
    expect(styles).toContain('padding-bottom: max(14px, env(safe-area-inset-bottom));')
    expect(styles).toMatch(
      /@media \(max-width: 760px\)[\s\S]*?\.footer\s*{[\s\S]*?flex-direction: column;/,
    )
    expect(styles).toMatch(/\.footer > div > button\s*{[\s\S]*?min-width: 0;/)
  })
})
