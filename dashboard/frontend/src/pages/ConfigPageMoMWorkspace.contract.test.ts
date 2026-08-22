import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const readSource = (name: string) => readFileSync(new URL(name, import.meta.url), 'utf8')

describe('Mixture-of-Models product workspace', () => {
  it('presents only the Recipes and Models product views', () => {
    const source = readSource('./ConfigPageEntrypointsRecipesSection.tsx')

    expect(source).toContain("export type MixtureWorkspaceView = 'recipes' | 'models'")
    expect(source).toContain("{ id: 'recipes', label: 'Recipes' }")
    expect(source).toContain("{ id: 'models', label: 'Models' }")
    expect(source).toContain('role="tablist"')
    expect(source).toContain('role="tabpanel"')
    expect(source).not.toMatch(/Built-in Models|Models & Routing|Probes/)
  })

  it('builds a callable model by assigning physical models to recipe decisions', () => {
    const dialog = readSource('./ConfigPageMixtureDialog.tsx')
    const support = readSource('./configPageSupport.ts')

    expect(dialog).toContain('Model assignments')
    expect(dialog).toContain('entrypoint.model_bindings')
    expect(dialog).toContain('model_bindings: modelBindings')
    expect(dialog).toContain('Choose at least one model')
    expect(support).toContain('model_bindings?: Record<string, DecisionModelRef[]>')
  })

  it('creates recipe identity first and builds its reusable path in scoped editors', () => {
    const dialog = readSource('./ConfigPageRecipeDialog.tsx')
    const scope = readSource('./configPageRoutingScopeSupport.ts')
    const signals = readSource('./ConfigPageSignalsSection.tsx')
    const projections = readSource('./ConfigPageProjectionsSection.tsx')
    const decisions = readSource('./ConfigPageDecisionsSection.tsx')
    const routing = readSource('./ConfigPageMoMRoutingPanel.tsx')

    expect(dialog).toContain('<span>Name</span>')
    expect(dialog).toContain('<span>Description</span>')
    expect(dialog).toContain('Build its path')
    expect(dialog).not.toContain('Model assignments')
    expect(scope).toContain('requestedRecipeScope')
    expect(scope).toContain('recipeDraftApi')
    for (const editor of [signals, projections, decisions]) {
      expect(editor).toContain('useRoutingScopeManager')
      expect(editor).toContain('<RoutingScopeSelector')
    }
    expect(routing).toContain(
      'navigate(`/config/signals?recipe=${encodeURIComponent(recipe.name)}`)',
    )
  })

  it('supports paged model and recipe portfolios', () => {
    const lists = readSource('./ConfigPageMoMRoutingLists.tsx')

    expect(lists).toContain('const PAGE_SIZE = 8')
    expect(lists).toContain('<Pager page={page}')
    expect(lists).toContain('Create model')
    expect(lists).toContain('Create recipe')
  })

  it('keeps package-managed configuration fail closed with concise copy', () => {
    const configPage = readSource('./ConfigPage.tsx')
    const banner = readSource('./ConfigPageManagedRecipeBanner.tsx')

    expect(configPage).toContain('configReadonly || managedRecipeProtection !== null')
    expect(configPage).toContain('isReadonly={configEditorReadonly}')
    expect(banner).toContain('Recipe package active')
    expect(banner).not.toMatch(/Built-in Models|Custom package lifecycle/)
  })
})
