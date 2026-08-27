import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const readSource = (name: string) => readFileSync(new URL(name, import.meta.url), 'utf8')

describe('Mixture-of-Models workspace contracts', () => {
  it('uses accessible Recipes, Models, and Probes views without the old built-in-model page', () => {
    const source = readSource('./ConfigPageEntrypointsRecipesSection.tsx')

    expect(source).toContain("type MixtureWorkspaceView = 'recipes' | 'models' | 'probes'")
    expect(source).toContain('role="tablist"')
    expect(source).toContain('role="tab"')
    expect(source).toContain('role="tabpanel"')
    expect(source).toContain('<ConfigPageMoMRoutingPanel')
    expect(source).toContain('<ConfigPageMoMProbesPanel')
    expect(source).not.toContain('useBuiltInModelCatalog()')
    expect(source).not.toContain('Built-in Models')
    expect(source).not.toContain('Models & Routing')
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

  it('keeps catalog reads fail closed and separate from configuration writes', () => {
    const configPage = readSource('./ConfigPage.tsx')
    const api = readSource('../utils/modelCatalogApi.ts')

    expect(configPage).toContain('isReadonly || !canWriteConfig(user)')
    expect(configPage).toContain('const configEditorReadonly = configReadonly')
    expect(api).toContain('if (!response.ok)')
    expect(api).toContain('if (!isBuiltInModelCatalog(payload))')
  })

  it('opens direct usage examples instead of a catalog disclosure', () => {
    const lists = readSource('./ConfigPageMoMRoutingLists.tsx')
    const usage = readSource('./ConfigPageModelUsageDialog.tsx')

    expect(lists).toContain('onUsage(entrypoint)')
    expect(lists).not.toContain('Catalog metadata is unavailable')
    expect(lists).not.toContain('RecipeModelPool')
    expect(usage).toContain('/v1/chat/completions')
    expect(usage).toContain('from openai import OpenAI')
    expect(usage).toContain('import OpenAI from "openai"')
  })

  it('creates a model by choosing a recipe and assigning every decision', () => {
    const routing = readSource('./ConfigPageMoMRoutingPanel.tsx')
    const dialog = readSource('./ConfigPageMixtureDialog.tsx')
    const support = readSource('./configPageMixtureSupport.ts')

    expect(routing).toContain('<ConfigPageMixtureDialog')
    expect(dialog).toContain('Choose a recipe. Assign the right models.')
    expect(dialog).toContain('Choose at least one model')
    expect(dialog).toContain('applyRecipeAssignments')
    expect(support).toContain('assignDecisionModels')
  })

  it('uses the dashboard role as the single authoring gate', () => {
    const configPage = readSource('./ConfigPage.tsx')

    expect(configPage).toContain('const configEditorReadonly = configReadonly')
    expect(configPage).toContain('isReadonly={configEditorReadonly}')
    expect(configPage).not.toContain('isReadonly={configReadonly}')
    expect(configPage).not.toContain('managedRecipeProtection')
  })
})
