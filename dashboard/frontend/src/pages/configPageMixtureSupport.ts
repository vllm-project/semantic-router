import { cloneConfigData } from './configPageCanonicalization'
import { DEFAULT_RECIPE_NAME } from './configPageEntrypointsRecipesSupport'
import type { ConfigData, DecisionConfig } from './configPageSupport'

export type ModelAssignments = Record<string, string[]>

export function assignmentState(decisions: DecisionConfig[]): ModelAssignments {
  return Object.fromEntries(
    decisions.map((decision) => [
      decision.name,
      (decision.modelRefs ?? []).map((reference) => reference.model).filter(Boolean),
    ]),
  )
}

export function assignDecisionModels(
  decisions: DecisionConfig[],
  assignments: ModelAssignments,
): DecisionConfig[] {
  return decisions.map((decision) => {
    const existing = new Map(
      (decision.modelRefs ?? []).map((reference) => [reference.model, reference]),
    )
    return {
      ...decision,
      modelRefs: (assignments[decision.name] ?? []).map((model) => ({
        ...(existing.get(model) ?? { model, use_reasoning: false }),
        model,
      })),
    }
  })
}

export function applyRecipeAssignments(
  config: ConfigData,
  recipeName: string,
  assignments: ModelAssignments,
): ConfigData {
  const next = cloneConfigData(config)
  if (recipeName === DEFAULT_RECIPE_NAME) {
    const explicitDefaultIndex = (next.recipes ?? []).findIndex(
      (recipe) => recipe.name === DEFAULT_RECIPE_NAME,
    )
    if (explicitDefaultIndex >= 0) {
      const recipes = [...(next.recipes ?? [])]
      const recipe = recipes[explicitDefaultIndex]
      recipes[explicitDefaultIndex] = {
        ...recipe,
        routing: {
          ...recipe.routing,
          decisions: assignDecisionModels(recipe.routing.decisions ?? [], assignments),
        },
      }
      next.recipes = recipes
      return next
    }
    next.routing = {
      ...(next.routing ?? {}),
      decisions: assignDecisionModels(next.routing?.decisions ?? next.decisions ?? [], assignments),
    }
    delete next.decisions
    return next
  }

  next.recipes = (next.recipes ?? []).map((recipe) =>
    recipe.name === recipeName
      ? {
          ...recipe,
          routing: {
            ...recipe.routing,
            decisions: assignDecisionModels(recipe.routing.decisions ?? [], assignments),
          },
        }
      : recipe,
  )
  return next
}
