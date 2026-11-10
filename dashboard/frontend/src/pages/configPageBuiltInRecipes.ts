import { useDashboardObservation } from '../utils/useDashboardObservation'
import { responseErrorMessage } from './configPageRequestErrors'
import type { ConfigData, DecisionModelRef } from './configPageSupport'

export interface BuiltInRecipeDecision {
  name: string
  description: string
  algorithm: string | null
  minimum_candidates: number
}

export interface BuiltInRecipe {
  name: string
  description: string
  decisions: BuiltInRecipeDecision[]
}

export interface BuiltInRecipeBundle {
  name: string
  sha256: string
  recipes: BuiltInRecipe[]
}

export interface BuiltInRecipeCatalog {
  catalog_version: string
  bundles: BuiltInRecipeBundle[]
}

export interface BuiltInRecipeComposeRequest {
  bundle: string
  recipe: string
  bindings: Record<string, DecisionModelRef[]>
  model_names: string[]
}

// Outlasts the Dashboard's 60-second budget for running the packaged CLI.
const BUILT_IN_RECIPES_TIMEOUT_MS = 90_000

export function useBuiltInRecipes() {
  return useDashboardObservation<BuiltInRecipeCatalog>('/api/recipes/builtin', {
    freshMs: Number.POSITIVE_INFINITY,
    pollMs: 0,
    timeoutMs: BUILT_IN_RECIPES_TIMEOUT_MS,
  })
}

export async function composeBuiltInRecipe(
  request: BuiltInRecipeComposeRequest,
): Promise<ConfigData> {
  const response = await fetch('/api/recipes/builtin/compose', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })
  if (!response.ok) throw new Error(await responseErrorMessage(response))
  return (await response.json()) as ConfigData
}
