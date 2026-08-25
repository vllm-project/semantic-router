import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const readSource = (name: string) => readFileSync(new URL(name, import.meta.url), 'utf8')

describe('Recipe Builder boundary', () => {
  it('loads and saves Router-managed Recipes with routing permissions', () => {
    const page = readSource('./BuilderPage.tsx')
    const client = readSource('./useBuilderRecipeClient.ts')

    expect(page).toContain('canReadRouting(user)')
    expect(page).toContain('canManageRouting(user)')
    expect(client).toContain('routingManagementApi.listRecipes()')
    expect(client).toContain('routingManagementApi.updateRecipe(recipe.id, recipe.revision')
    expect(client).toContain('routingManagementApi.createRecipe')
  })

  it('projects compiler Recipe documents and never authors deployment configuration', () => {
    const page = readSource('./BuilderPage.tsx')
    const projection = readSource('./builderRecipeClient.ts')
    const store = readSource('../stores/dslStore.ts')

    expect(projection).toContain('result.recipeDocuments')
    expect(projection).toContain('documents.length !== 1')
    expect(projection).toContain('routing,')
    expect(projection).not.toContain("version: 'v0.4'")
    expect(projection).not.toContain('id: target.id')
    expect(projection).not.toContain('revision: target.recipeRevision')
    expect(page).not.toMatch(/canWriteConfig|canDeployConfig|loadFromRouter|requestDeploy/)
    expect(`${page}\n${store}`).not.toMatch(
      /\/api\/router\/config\/(?:yaml|deploy|preview|rollback|versions)/,
    )
  })

  it('keeps built-in Recipes immutable and moves physical assignment out of Builder', () => {
    const page = readSource('./BuilderPage.tsx')
    const visual = readSource('./builderPageVisualShell.tsx')
    const route = readSource('./builderPageAddRouteForm.tsx')

    expect(page).toContain('selectedRecipe.immutable')
    expect(page).toContain('Duplicate it to make your own.')
    expect(visual).not.toContain('AddModelForm')
    expect(route).toContain('models: []')
    expect(route).toContain('Assign models after this Recipe is used by a Mixture-of-Model.')
  })

  it('removes every mutation entry point for routing readers', () => {
    const page = readSource('./BuilderPage.tsx')
    const toolbar = readSource('./builderPageToolbar.tsx')

    expect(toolbar).toContain('const editable = !readOnly && !immutable')
    expect(toolbar).toContain('{editable ? (')
    expect(toolbar).toContain('{!readOnly ? (')
    expect(page).toContain('readOnly={!editable}')
    expect(page).toContain('open={editable && showImportModal}')
    expect(page).toContain('open={editable && guideOpen}')
    expect(page).toContain('{editable ? (')
  })
})
