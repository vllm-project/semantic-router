import type { RoutingRecipe, RoutingRecipeWrite } from '../utils/routingManagementApi'

export const EMPTY_RECIPE_DOCUMENT: Record<string, unknown> = {
  signals: {},
  projections: {},
  decisions: [],
}

const cloneDocument = (document: Record<string, unknown>): Record<string, unknown> =>
  JSON.parse(JSON.stringify(document)) as Record<string, unknown>

const collectionSize = (value: unknown): number => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return 0
  return Object.values(value).reduce<number>(
    (count, family) => count + (Array.isArray(family) ? family.length : 0),
    0,
  )
}

export function recipeDocumentSummary(recipe: RoutingRecipe) {
  return {
    signals: collectionSize(recipe.document.signals),
    projections: collectionSize(recipe.document.projections),
    decisions: recipe.decisions.length,
  }
}

export function suggestedRecipeCopyName(recipe: RoutingRecipe): string {
  return `${recipe.name} copy`.slice(0, 256)
}

export function recipeWrite(
  name: string,
  description: string,
  sourceDocument: Record<string, unknown> = EMPTY_RECIPE_DOCUMENT,
): RoutingRecipeWrite {
  return {
    name: name.trim(),
    ...(description.trim() ? { description: description.trim() } : {}),
    document: cloneDocument(sourceDocument),
  }
}
