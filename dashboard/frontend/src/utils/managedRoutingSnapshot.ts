import {
  routingManagementApi,
  type RoutingAssignmentSet,
  type RoutingEntrypoint,
  type RoutingEntrypointRule,
  type RoutingModelCardView,
  type RoutingRecipe,
} from './routingManagementApi'

export interface RoutingDocument {
  strategy?: unknown
  signals?: Record<string, unknown>
  projections?: Record<string, unknown>
  decisions?: Array<Record<string, unknown>>
  [key: string]: unknown
}

export interface ManagedRoutingScope {
  id: string
  recipeId: string
  label: string
  description?: string
  entrypointModelNames: string[]
  document: RoutingDocument
  source: 'recipe' | 'entrypoint'
  hydrated: boolean
}

export interface ManagedRoutingSummary {
  models: RoutingModelCardView[]
  recipes: RoutingRecipe[]
  entrypoints: RoutingEntrypoint[]
}

export interface ManagedRoutingSnapshot extends ManagedRoutingSummary {
  routingScopes: ManagedRoutingScope[]
}

const clone = <T>(value: T): T => JSON.parse(JSON.stringify(value)) as T

const record = (value: unknown): Record<string, unknown> =>
  value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {}

export function recipeDocument(recipe: RoutingRecipe): RoutingDocument {
  const document = record(recipe.document)
  return {
    ...(document.strategy !== undefined ? { strategy: clone(document.strategy) } : {}),
    ...(document.signals !== undefined ? { signals: clone(record(document.signals)) } : {}),
    ...(document.projections !== undefined
      ? { projections: clone(record(document.projections)) }
      : {}),
    decisions: Array.isArray(document.decisions)
      ? document.decisions.map((decision) => clone(record(decision)))
      : [],
  }
}

function assignmentForDecision(
  recipe: RoutingRecipe,
  rule: RoutingEntrypointRule,
  decision: Record<string, unknown>,
): RoutingAssignmentSet | undefined {
  const declaredID = typeof decision.id === 'string' ? decision.id : ''
  const declaredName = typeof decision.name === 'string' ? decision.name : ''
  const metadata = recipe.decisions.find(
    (candidate) => candidate.id === declaredID || candidate.name === declaredName,
  )
  return rule.assignments[declaredID] ?? rule.assignments[metadata?.id ?? '']
}

function assignedDocument(
  recipe: RoutingRecipe,
  rule: RoutingEntrypointRule,
  modelsByID: Map<string, RoutingModelCardView>,
): RoutingDocument {
  const document = recipeDocument(recipe)
  return {
    ...document,
    decisions: (document.decisions ?? []).map((decision) => {
      const assignment = assignmentForDecision(recipe, rule, decision)
      return {
        ...decision,
        modelRefs: (assignment?.models ?? [])
          .slice()
          .sort((left, right) => left.priority - right.priority)
          .map((reference) => {
            const model = modelsByID.get(reference.modelId)
            return {
              model: model?.name ?? reference.modelId,
              use_reasoning: reference.reasoning?.enabled ?? false,
              ...(reference.reasoning?.effort
                ? { reasoning_effort: reference.reasoning.effort }
                : {}),
              ...(model?.card.reasoning?.type
                ? { reasoning_family: model.card.reasoning.type }
                : {}),
              ...(reference.loraName ? { lora_name: reference.loraName } : {}),
            }
          }),
      }
    }),
  }
}

function publicNames(entrypoint: RoutingEntrypoint): string[] {
  return entrypoint.aliases.length > 0 ? entrypoint.aliases : [entrypoint.name]
}

function entrypointNamesForRecipe(entrypoints: RoutingEntrypoint[], recipeID: string): string[] {
  return [
    ...new Set(
      entrypoints.flatMap((entrypoint) =>
        (entrypoint.rules ?? []).some((rule) => rule.recipeId === recipeID) ||
        entrypoint.recipeIds.includes(recipeID)
          ? publicNames(entrypoint)
          : [],
      ),
    ),
  ]
}

export function listManagedRecipeScopes(
  catalog: Pick<ManagedRoutingSummary, 'recipes' | 'entrypoints'>,
): ManagedRoutingScope[] {
  return catalog.recipes.map((recipe) => ({
    id: `recipe:${recipe.id}`,
    recipeId: recipe.id,
    label: recipe.name,
    description: recipe.description,
    entrypointModelNames: entrypointNamesForRecipe(catalog.entrypoints, recipe.id),
    document: recipeDocument(recipe),
    source: 'recipe',
    hydrated: true,
  }))
}

function buildTopologyScopes(
  models: RoutingModelCardView[],
  recipes: RoutingRecipe[],
  summaries: RoutingEntrypoint[],
  hydratedEntrypoints: RoutingEntrypoint[],
): ManagedRoutingScope[] {
  const modelsByID = new Map(models.map((model) => [model.id, model]))
  const recipesByID = new Map(recipes.map((recipe) => [recipe.id, recipe]))
  const hydratedIDs = new Set(hydratedEntrypoints.map((entrypoint) => entrypoint.id))
  const referencedRecipeIDs = new Set<string>()
  const scopes: ManagedRoutingScope[] = []

  for (const entrypoint of hydratedEntrypoints) {
    const names = publicNames(entrypoint)
    const rules = entrypoint.rules ?? []
    for (const rule of rules) {
      const recipe = recipesByID.get(rule.recipeId)
      if (!recipe) continue
      referencedRecipeIDs.add(recipe.id)
      scopes.push({
        id: `entrypoint:${entrypoint.id}:${rule.id}`,
        recipeId: recipe.id,
        label: rules.length > 1 ? `${names[0]} · ${rule.name}` : names[0],
        description: recipe.description,
        entrypointModelNames: names,
        document: assignedDocument(recipe, rule, modelsByID),
        source: 'entrypoint',
        hydrated: true,
      })
    }
  }

  for (const entrypoint of summaries) {
    if (hydratedIDs.has(entrypoint.id)) continue
    scopes.push({
      id: `entrypoint:${entrypoint.id}`,
      recipeId: '',
      label: publicNames(entrypoint)[0],
      description: `${entrypoint.ruleCount} routing rule${entrypoint.ruleCount === 1 ? '' : 's'}`,
      entrypointModelNames: publicNames(entrypoint),
      document: { decisions: [] },
      source: 'entrypoint',
      hydrated: false,
    })
  }

  for (const recipe of recipes) {
    if (referencedRecipeIDs.has(recipe.id)) continue
    scopes.push({
      id: `recipe:${recipe.id}`,
      recipeId: recipe.id,
      label: recipe.name,
      description: recipe.description,
      entrypointModelNames: entrypointNamesForRecipe(summaries, recipe.id),
      document: recipeDocument(recipe),
      source: 'recipe',
      hydrated: true,
    })
  }

  return scopes
}

export function buildManagedRoutingSnapshot(
  models: RoutingModelCardView[],
  recipes: RoutingRecipe[],
  entrypoints: RoutingEntrypoint[],
): ManagedRoutingSnapshot {
  const hydratedEntrypoints = entrypoints.filter((entrypoint) => entrypoint.rules !== undefined)
  return {
    models: clone(models),
    recipes: clone(recipes),
    entrypoints: clone(entrypoints),
    routingScopes: buildTopologyScopes(models, recipes, entrypoints, hydratedEntrypoints),
  }
}

export function buildManagedRoutingSummary(
  models: RoutingModelCardView[],
  recipes: RoutingRecipe[],
  entrypoints: RoutingEntrypoint[],
): ManagedRoutingSummary {
  return {
    models: clone(models),
    recipes: clone(recipes),
    entrypoints: clone(entrypoints),
  }
}

const entrypointIDFromScope = (scopeId?: string | null) => {
  if (!scopeId?.startsWith('entrypoint:')) return ''
  return scopeId.slice('entrypoint:'.length).split(':', 1)[0] ?? ''
}

async function fetchManagedRoutingCatalog() {
  const [models, recipes, entrypoints] = await Promise.all([
    routingManagementApi.listModelCards(),
    routingManagementApi.listRecipes(),
    routingManagementApi.listEntrypoints(),
  ])
  return { models, recipes, entrypoints }
}

export async function fetchManagedRoutingSummary(): Promise<ManagedRoutingSummary> {
  const { models, recipes, entrypoints } = await fetchManagedRoutingCatalog()
  return buildManagedRoutingSummary(models, recipes, entrypoints)
}

/**
 * Loads one representative Entrypoint per Recipe for overview surfaces. The
 * list contract carries only Recipe references, so assignment hydration stays
 * bounded by the number of Recipes rather than growing with every Entrypoint.
 */
export async function fetchManagedRoutingOverviewSnapshot(): Promise<ManagedRoutingSnapshot> {
  const { models, recipes, entrypoints: summaries } = await fetchManagedRoutingCatalog()
  const uncoveredRecipes = new Set(recipes.map((recipe) => recipe.id))
  const representatives: RoutingEntrypoint[] = []
  for (const summary of summaries) {
    const covers = summary.recipeIds.filter((recipeID) => uncoveredRecipes.has(recipeID))
    if (covers.length === 0) continue
    representatives.push(summary)
    covers.forEach((recipeID) => uncoveredRecipes.delete(recipeID))
    if (uncoveredRecipes.size === 0) break
  }
  const hydratedEntrypoints = await Promise.all(
    representatives.map((entrypoint) =>
      entrypoint.rules
        ? Promise.resolve(entrypoint)
        : routingManagementApi.getEntrypointTopology(entrypoint.id),
    ),
  )
  const hydratedRecipeIDs = new Set(
    hydratedEntrypoints.flatMap(
      (entrypoint) => entrypoint.rules?.map((rule) => rule.recipeId) ?? [],
    ),
  )
  const missingAssignments = [...uncoveredRecipes].filter((recipeID) =>
    summaries.some((entrypoint) => entrypoint.recipeIds.includes(recipeID)),
  )
  if (
    missingAssignments.length > 0 ||
    representatives.some((entrypoint) =>
      entrypoint.recipeIds.some((recipeID) => !hydratedRecipeIDs.has(recipeID)),
    )
  ) {
    throw new Error('Router returned an incomplete Entrypoint assignment topology.')
  }
  return {
    models: clone(models),
    recipes: clone(recipes),
    entrypoints: clone(summaries),
    routingScopes: buildTopologyScopes(models, recipes, summaries, hydratedEntrypoints),
  }
}

/**
 * Loads one topology at most. A Recipe deep link needs no Entrypoint hydration;
 * an Entrypoint deep link hydrates only that Entrypoint; an unscoped visit uses
 * one bounded default. List summaries remain available without turning the
 * topology page into an N+1 client.
 */
export async function fetchManagedRoutingSnapshot(
  requestedScopeId?: string | null,
): Promise<ManagedRoutingSnapshot> {
  const { models, recipes, entrypoints: summaries } = await fetchManagedRoutingCatalog()
  const requestedEntrypointID = entrypointIDFromScope(requestedScopeId)
  const selectedSummary = requestedEntrypointID
    ? summaries.find((entrypoint) => entrypoint.id === requestedEntrypointID)
    : requestedScopeId?.startsWith('recipe:')
      ? undefined
      : summaries[0]
  const hydratedEntrypoints = selectedSummary
    ? [
        selectedSummary.rules
          ? selectedSummary
          : await routingManagementApi.getEntrypointTopology(selectedSummary.id),
      ]
    : []

  return {
    models: clone(models),
    recipes: clone(recipes),
    entrypoints: clone(summaries),
    routingScopes: buildTopologyScopes(models, recipes, summaries, hydratedEntrypoints),
  }
}
