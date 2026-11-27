import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const readSource = (name: string) => readFileSync(new URL(name, import.meta.url), 'utf8')

describe('Recipes page contracts', () => {
  it('shows public models first, then recipes and probes, without a built-in model catalog', () => {
    const source = readSource('./ConfigPageEntrypointsRecipesSection.tsx')

    expect(source).toContain("type RecipesPageTab = 'models' | 'recipes' | 'probes'")
    expect(source).toContain("const TABS: RecipesPageTab[] = ['models', 'recipes', 'probes']")
    expect(source).toContain('<TabsTrigger value="models">')
    expect(source).toContain('<ConfigPageRecipesPublicModels')
    expect(source).toContain('<ConfigPageRecipesTable')
    expect(source).toContain('<ConfigPageMoMProbesPanel')
    expect(source).not.toContain('useBuiltInModelCatalog()')
    expect(source).not.toContain('Built-in Models')
  })

  it('uses server paging, lazy detail, strict validation, and fresh run plans', () => {
    const source = readSource('./ConfigPageMoMProbesPanel.tsx')

    expect(source).toContain('RECIPE_PROBE_PAGE_SIZE')
    expect(source).toContain('listRecipeProbes(')
    expect(source).toContain('getRecipeProbe(')
    expect(source).toContain('validateRecipeProbe(')
    expect(source).toContain('createRecipeProbeRunPlan(')
    expect(source).not.toContain('/api/v1/routing/preview')
    expect(source).not.toContain('simulate')
  })

  it('activates and deactivates recipe packages through a previewed, confirmed plan', () => {
    const dialog = readSource('./ConfigPageRecipeActivationDialog.tsx')

    expect(dialog).toContain('previewRecipePackageActivation(')
    expect(dialog).toContain('previewRecipePackageDeactivation(')
    expect(dialog).toContain('activateRecipePackage(')
    expect(dialog).toContain('deactivateRecipePackage(')
    expect(dialog).toContain('expected_plan_digest: plan.plan_digest')
    expect(dialog).toContain('Restart required: run vllm-sr serve to apply')
  })

  it('keeps catalog reads fail closed and separate from configuration writes', () => {
    const configPage = readSource('./ConfigPage.tsx')
    const api = readSource('../utils/modelCatalogApi.ts')

    expect(configPage).toContain('isReadonly || !canWriteConfig(user)')
    expect(configPage).toContain('const configEditorReadonly = configReadonly')
    expect(api).toContain('if (!response.ok)')
    expect(api).toContain('if (!isBuiltInModelCatalog(payload))')
  })

  it('opens direct usage examples for every chat public model', () => {
    const cards = readSource('./ConfigPageRecipesPublicModels.tsx')
    const sheet = readSource('./ConfigPageEntrypointSheet.tsx')
    const support = readSource('./configPageRecipesPageSupport.ts')

    expect(cards).toContain("onOpen(card, 'connect')")
    expect(cards).toContain('Evaluate')
    expect(sheet).toContain('usageExamples(origin, name)')
    expect(sheet).toContain('No API key required')
    expect(support).toContain('/v1/chat/completions')
    expect(support).toContain('from openai import OpenAI')
    expect(support).toContain('import OpenAI from "openai"')
    expect(support).not.toContain('YOUR_API_KEY')
  })

  it('creates a model by choosing a recipe and assigning every decision', () => {
    const page = readSource('./ConfigPageEntrypointsRecipesSection.tsx')
    const dialog = readSource('./ConfigPageMixtureDialog.tsx')
    const editor = readSource('./useMixtureEditor.ts')
    const support = readSource('./configPageMixtureSupport.ts')

    expect(page).toContain('<ConfigPageMixtureDialog')
    expect(dialog).toContain('Choose a recipe. Assign the right models.')
    expect(editor).toContain('resolveMixtureRecipe')
    expect(editor).toContain('applyRecipeAssignments')
    expect(support).toContain('minimumCandidatesForDecision(decision)')
    expect(support).toContain('assignDecisionModels')
  })

  it('publishes built-in recipes through the packaged CLI composition', () => {
    const page = readSource('./ConfigPageEntrypointsRecipesSection.tsx')
    const editor = readSource('./useMixtureEditor.ts')

    expect(page).toContain('useBuiltInRecipes()')
    expect(editor).toContain('composeBuiltInRecipe(')
    expect(editor).toContain('builtInRecipeBindings(decisions, assignments)')
  })

  it('uses the dashboard role as the single authoring gate', () => {
    const configPage = readSource('./ConfigPage.tsx')

    expect(configPage).toContain('const configEditorReadonly = configReadonly')
    expect(configPage).toContain('isReadonly={configEditorReadonly}')
    expect(configPage).not.toContain('isReadonly={configReadonly}')
    expect(configPage).not.toContain('managedRecipeProtection')
  })
})
