import type { BuiltInRecipeCatalog } from './configPageBuiltInRecipes'
import { cloneConfigData } from './configPageCanonicalization'
import { DEFAULT_RECIPE_NAME, getRecipeByName } from './configPageEntrypointsRecipesSupport'
import type { ConfigData, DecisionConfig, DecisionModelRef } from './configPageSupport'

export type ModelAssignments = Record<string, string[]>

export interface MixtureRecipeChoice {
  name: string
  /** Installed bundle of a built-in recipe; absent for a recipe in the config. */
  bundle?: string
}

/** A decision to assign. A minimum of 0 means it responds without calling a model. */
export interface MixtureDecision {
  name: string
  description?: string
  minimumCandidates: number
}

export function minimumCandidatesForDecision(decision: DecisionConfig): number {
  const value = decision.algorithm?.minimum_candidates
  return typeof value === 'number' && Number.isInteger(value) && value > 0 ? value : 1
}

export const mixtureRecipeKey = ({ bundle, name }: MixtureRecipeChoice): string =>
  JSON.stringify(bundle === undefined ? [name] : [bundle, name])

export function builtInRecipeChoices(catalog: BuiltInRecipeCatalog | null): MixtureRecipeChoice[] {
  return (catalog?.bundles ?? []).flatMap((bundle) =>
    bundle.recipes.map((recipe) => ({ bundle: bundle.name, name: recipe.name })),
  )
}

export function resolveMixtureRecipe(
  config: ConfigData,
  catalog: BuiltInRecipeCatalog | null,
  choice: MixtureRecipeChoice | null,
): { description?: string; decisions: MixtureDecision[] } {
  if (!choice) return { decisions: [] }
  if (choice.bundle !== undefined) {
    const recipe = catalog?.bundles
      .find((bundle) => bundle.name === choice.bundle)
      ?.recipes.find((candidate) => candidate.name === choice.name)
    return {
      description: recipe?.description,
      decisions: (recipe?.decisions ?? []).map((decision) => ({
        name: decision.name,
        description: decision.description,
        minimumCandidates: decision.minimum_candidates,
      })),
    }
  }
  const recipe = getRecipeByName(config, choice.name)
  return {
    description: recipe?.description,
    decisions: (recipe?.routing.decisions ?? []).map((decision) => ({
      name: decision.name,
      description: decision.description,
      minimumCandidates: minimumCandidatesForDecision(decision),
    })),
  }
}

const newModelRef = (model: string): DecisionModelRef => ({ model, use_reasoning: false })

export function builtInRecipeBindings(
  decisions: MixtureDecision[],
  assignments: ModelAssignments,
): Record<string, DecisionModelRef[]> {
  return Object.fromEntries(
    decisions
      .filter((decision) => decision.minimumCandidates > 0)
      .map((decision) => [decision.name, (assignments[decision.name] ?? []).map(newModelRef)]),
  )
}

export function assignmentState(decisions: DecisionConfig[]): ModelAssignments {
  return Object.fromEntries(
    decisions.map((decision) => [
      decision.name,
      (decision.modelRefs ?? []).map((reference) => reference.model).filter(Boolean),
    ]),
  )
}

/** Models a config recipe already assigns; a built-in recipe starts unassigned. */
export function recipeAssignments(
  config: ConfigData,
  choice: MixtureRecipeChoice | null,
): ModelAssignments {
  return choice && choice.bundle === undefined
    ? assignmentState(getRecipeByName(config, choice.name)?.routing.decisions ?? [])
    : {}
}

export function assignDecisionModels(
  decisions: DecisionConfig[],
  assignments: ModelAssignments,
): DecisionConfig[] {
  return decisions.map((decision) => {
    const existing = new Map(
      (decision.modelRefs ?? []).map((reference) => [reference.model, reference]),
    )
    const modelRefs = (assignments[decision.name] ?? []).map((model) => ({
      ...(existing.get(model) ?? newModelRef(model)),
      model,
    }))
    return {
      ...decision,
      modelRefs,
      algorithm: materializeDynamicWorkflowPlanner(decision.algorithm, modelRefs),
    }
  })
}

function materializeDynamicWorkflowPlanner(
  algorithm: Record<string, unknown> | undefined,
  modelRefs: DecisionConfig['modelRefs'],
): Record<string, unknown> | undefined {
  if (algorithm?.type !== 'workflows') return algorithm
  const workflows = asRecord(algorithm.workflows)
  if (workflows?.mode !== 'dynamic') return algorithm
  const planner = asRecord(workflows.planner) ?? {}
  if (typeof planner.model === 'string' && planner.model.trim()) return algorithm
  const model = modelRefs.find((reference) => reference.model.trim())?.model
  if (!model) return algorithm
  return {
    ...algorithm,
    workflows: {
      ...workflows,
      planner: { ...planner, model },
    },
  }
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined
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
