import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const readSource = (name: string) => readFileSync(new URL(name, import.meta.url), 'utf8')

describe('Mixture-of-Models Management API workspace', () => {
  it('presents only Recipes and Models', () => {
    const source = readSource('./ConfigPageEntrypointsRecipesSection.tsx')
    expect(source).toContain("export type MixtureWorkspaceView = 'recipes' | 'models'")
    expect(source).toContain("{ id: 'recipes', label: 'Recipes' }")
    expect(source).toContain("{ id: 'models', label: 'Models' }")
    expect(source).not.toMatch(/Built-in Models|Models & Routing|Probes/)
  })

  it('uses Router permissions as the only UI capability signal', () => {
    const source = readSource('./ConfigPageEntrypointsRecipesSection.tsx')
    expect(source).toContain('canReadRoutingCatalog(user)')
    expect(source).toContain('canManageRouting(user)')
    expect(source).not.toMatch(/configEditorReadonly|canWriteConfig|isReadonly/)
  })

  it('assigns stable Model ids to every stable decision id', () => {
    const dialog = readSource('./ConfigPageMixtureDialog.tsx')
    expect(dialog).toContain('Model assignments')
    expect(dialog).toContain('decision.id')
    expect(dialog).toContain('model.id')
    expect(dialog).toContain('Choose at least one model')
    expect(dialog).toContain("strategy: 'priority'")
    expect(dialog).toContain('Try the next priority when this model can’t start')
  })

  it('uses the Router Management API as its only persistence boundary', () => {
    const sources = [
      './ConfigPageEntrypointsRecipesSection.tsx',
      './ConfigPageMoMRoutingPanel.tsx',
      './ConfigPageMoMRoutingLists.tsx',
      './ConfigPageMixtureDialog.tsx',
      './ConfigPageRecipeDialog.tsx',
      './ConfigPageModelsSection.tsx',
      './configPageRoutingScopeSupport.ts',
    ]
      .map(readSource)
      .join('\n')
    expect(sources).not.toMatch(/cloneConfigData|recipeDraftApi|\/api\/router\/config/)
    expect(sources).toContain('routingManagementApi')
  })

  it('keeps Model edits sparse and physical assignment in the Entrypoint only', () => {
    const models = readSource('./ConfigPageModelsSection.tsx')
    const scopes = readSource('./configPageRoutingScopeSupport.ts')
    const recipeTypes = readSource('./configPageSupport.ts')
    expect(models).toContain('RoutingModelPatch')
    expect(models).not.toMatch(/patch\s*=\s*\{[\s\S]*?backends:/)
    expect(scopes).not.toContain('modelRefs')
    expect(recipeTypes).toContain('export type ConfigData = RoutingConfig')
    expect(recipeTypes).not.toMatch(
      /CanonicalGlobalConfig|config_source|auto_model_names?|include_config_models_in_list|semantic_cache|ratelimit|modelRefs/,
    )
  })

  it('keeps topology visible while mutations require routing.manage', () => {
    const panel = readSource('./ConfigPageMoMRoutingPanel.tsx')
    const lists = readSource('./ConfigPageMoMRoutingLists.tsx')
    const topology = readSource('./ConfigPageMoMTopologyDialog.tsx')
    expect(panel).toContain('getEntrypointTopology')
    expect(panel).toContain('canManage={canManage}')
    expect(lists).toContain('onView(entrypoint)')
    expect(topology).toContain('{canManage ?')
  })

  it('keeps the topology dialog inside a 320px mobile viewport', () => {
    const styles = readSource('./ConfigPageMoMTopologyDialog.module.css')

    expect(styles).toMatch(
      /@media \(max-width: 640px\)[\s\S]*?\.overlay\s*{[\s\S]*?padding: 0\.375rem;/,
    )
    expect(styles).toMatch(
      /@media \(max-width: 640px\)[\s\S]*?\.dialog\s*{[\s\S]*?width: 100%;[\s\S]*?max-height: calc\(100dvh - 0\.75rem\);/,
    )
    expect(styles).toMatch(/\.actions > div\s*{[\s\S]*?minmax\(0, 1fr\)/)
  })

  it('duplicates immutable Recipes into a clean custom create flow', () => {
    const panel = readSource('./ConfigPageMoMRoutingPanel.tsx')
    const dialog = readSource('./ConfigPageRecipeDialog.tsx')
    expect(panel).toContain('recipeEditor?.immutable && canManage')
    expect(panel).toContain('duplicateFrom={recipeTemplate ?? undefined}')
    expect(dialog).toContain('Duplicate recipe')
    expect(dialog).not.toMatch(/JSON\.parse|Recipe document|<textarea/)
  })
})
