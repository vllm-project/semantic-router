import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

import type { RecipeSourceHealth } from '../types/recipe'

const readSource = (name: string) => readFileSync(new URL(name, import.meta.url), 'utf8')

describe('Mixture-of-Models workspace contracts', () => {
  it('keeps the existing route section as a narrow accessible three-view orchestrator', () => {
    const source = readSource('./ConfigPageEntrypointsRecipesSection.tsx')

    expect(source).toContain("type MoMView = 'overview' | 'routing' | 'probes'")
    expect(source).toContain('role="tablist"')
    expect(source).toContain('role="tab"')
    expect(source).toContain('role="tabpanel"')
    expect(source).toContain('<ConfigPageMoMRoutingPanel')
    expect(source).toContain('<ConfigPageMoMOverviewPanel')
    expect(source).toContain('<ConfigPageMoMProbesPanel')
    expect(source).toContain('useBuiltInModelCatalog()')
    expect(source).toContain('recipeRevision')
    expect(source).toContain("window.addEventListener('config-deployed'")
    expect(source).not.toContain('AMD models')
  })

  it('uses server paging, lazy detail, strict validation, and fresh run plans', () => {
    const source = readSource('./ConfigPageMoMProbesPanel.tsx')

    expect(source).toContain('RECIPE_PROBE_PAGE_SIZE')
    expect(source).toContain('listRecipeProbes(')
    expect(source).toContain('getRecipeProbe(')
    expect(source).toContain('validateRecipeProbe(')
    expect(source).toContain('createRecipeProbeRunPlan(')
    expect(source).not.toContain('/api/v1/eval')
    expect(source).not.toContain('simulate')
  })

  it('does not load package-controlled images from Recipe documentation', () => {
    const source = readSource('./ConfigPageMoMActiveRecipePanel.tsx')
    const types = readSource('../types/recipe.ts')

    expect(source).toContain('<MarkdownRenderer content={recipe.readme} allowImages={false} />')
    expect(types).toContain("'ready' | 'unmanaged' | 'invalid' | 'recovering' | 'inconsistent'")
    const recoveryStates: RecipeSourceHealth['status'][] = ['recovering', 'inconsistent']
    expect(recoveryStates).toEqual(['recovering', 'inconsistent'])
  })

  it('uses the installed CLI catalog as the primary distribution surface without ZIP import UI', () => {
    const overview = readSource('./ConfigPageMoMOverviewPanel.tsx')
    const catalog = readSource('./ConfigPageMoMCatalogPanel.tsx')
    const compatibilityLifecycle = readSource('./ConfigPageMoMPackagesPanel.tsx')
    const packageSupport = readSource('./configPageMoMPackagesSupport.ts')
    const packageStyles = readSource('./ConfigPageMoMPackagesPanel.module.css')
    const api = readSource('../utils/modelCatalogApi.ts')
    const recipeApi = readSource('../utils/recipeApi.ts')
    const recipeTypes = readSource('../types/recipe.ts')

    expect(overview).toContain('<ConfigPageMoMCatalogPanel')
    expect(overview).toContain('<ConfigPageMoMPackagesPanel')
    expect(overview.indexOf('<ConfigPageMoMCatalogPanel')).toBeLessThan(
      overview.indexOf('<ConfigPageMoMPackagesPanel'),
    )
    expect(catalog).toContain('Built-in model catalogs')
    expect(catalog).toContain('version packages')
    expect(catalog).toContain('model.roles.map')
    expect(catalog).toContain('recommended_pool.map')
    expect(catalog).toContain('vllm-sr model fork')
    expect(catalog).not.toMatch(/ZIP|importRecipePackage|Recipe ZIP/i)
    expect(compatibilityLifecycle).toContain('Custom package lifecycle')
    expect(compatibilityLifecycle).toContain('previewRecipePackageActivation(')
    expect(compatibilityLifecycle).toContain('previewRecipePackageDeactivation(')
    expect(compatibilityLifecycle).toContain('deactivateRecipePackage(')
    expect(compatibilityLifecycle).not.toMatch(
      /importRecipePackage|validateRecipePackageImport|Recipe ZIP|SHA-256|HTTPS URL/i,
    )
    expect(recipeApi).not.toMatch(/importRecipePackage|\/api\/recipe\/import/)
    expect(recipeTypes).not.toMatch(/RecipePackageImport(?:Request|Result)/)
    expect(packageSupport).not.toMatch(/canImport|importBlocker|validateRecipePackageImport/)
    expect(packageStyles).not.toMatch(/\.import(?:Form|Heading)|\.urlRow|\.advanced|\.fieldError/)
    expect(api).toContain("fetch('/api/models/catalog'")
  })

  it('keeps catalog reads and custom configuration mutations on separate fail-closed seams', () => {
    const configPage = readSource('./ConfigPage.tsx')
    const api = readSource('../utils/modelCatalogApi.ts')

    expect(configPage).toContain('isReadonly || !canWriteConfig(user)')
    expect(configPage).toContain('configReadonly || managedRecipeProtection !== null')
    expect(configPage).toContain('resolveRecipePackageCapabilities(')
    expect(api).toContain('if (!response.ok)')
    expect(api).toContain('if (!isBuiltInModelCatalog(payload))')
  })

  it('shows catalog reference metadata only on unified-model disclosures', () => {
    const lists = readSource('./ConfigPageMoMRoutingLists.tsx')
    const metadata = readSource('./ConfigPageMoMEntrypointCatalogMetadata.tsx')

    expect(lists).toContain('<ConfigPageMoMEntrypointCatalogMetadata')
    expect(lists).not.toContain('RecipeModelPool')
    expect(lists.match(/className=\{pageStyles\.disclosureButton\}/g)).toHaveLength(1)
    expect(metadata).toContain('Mixture-of-Models metadata')
    expect(metadata).toContain('catalog ${model.verification.status}')
    expect(metadata).toContain('model.verification.authority')
    expect(metadata).toContain('model.verification.asset_sha256')
    expect(metadata).toMatch(/currently\s+deployed YAML is independently custom \/ unverified/)
  })

  it('keeps custom unified-model creation outside the reserved built-in namespace', () => {
    const routing = readSource('./ConfigPageMoMRoutingPanel.tsx')

    expect(routing).toContain("modelNames: entrypoint?.model_names.join('\\n') ?? ''")
    expect(routing).toContain("placeholder: 'team/custom-model'")
    expect(routing).not.toMatch(/vllm-sr\/mom-/)
  })

  it('locks every ordinary ConfigPage editor while a package manages runtime config', () => {
    const configPage = readSource('./ConfigPage.tsx')
    const banner = readSource('./ConfigPageManagedRecipeBanner.tsx')

    expect(configPage).toContain('resolveManagedRecipeProtection(recipeDescriptor)')
    expect(configPage).toContain('configReadonly || managedRecipeProtection !== null')
    expect(configPage).toContain('isReadonly={configEditorReadonly}')
    expect(configPage).not.toContain('isReadonly={configReadonly}')
    expect(configPage).toContain('<ConfigPageManagedRecipeBanner')
    expect(configPage).toContain('refreshAfterRecipeLifecycleChange')
    expect(configPage).toContain('if (managedRecipeProtection)')
    expect(banner).toContain('Ordinary configuration editors are read-only')
    expect(banner).toContain('Custom package lifecycle')
  })
})
