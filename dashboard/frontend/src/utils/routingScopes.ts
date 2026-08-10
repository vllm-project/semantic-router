export const DEFAULT_ROUTING_SCOPE_ID = 'default'

export interface RoutingProfileLike {
  signals?: Record<string, unknown>
  projections?: Record<string, unknown>
  decisions?: unknown[]
  strategy?: unknown
  [key: string]: unknown
}

export interface RoutingRecipeLike {
  name: string
  description?: string
  routing: RoutingProfileLike
}

export interface RoutingEntrypointLike {
  model_names: string[]
  recipe: string
}

export interface RoutingScopedConfigLike {
  routing?: RoutingProfileLike
  signals?: Record<string, unknown>
  projections?: Record<string, unknown>
  decisions?: unknown[]
  recipes?: RoutingRecipeLike[]
  entrypoints?: RoutingEntrypointLike[]
  [key: string]: unknown
}

export interface RoutingScope {
  id: string
  label: string
  description?: string
  entrypointModelNames: string[]
  routing: RoutingProfileLike
  isDefault: boolean
  source: 'routing' | 'recipe'
}

export interface ScopedRoutingValue<T> {
  scope: RoutingScope
  value: T
}

const cloneValue = <T>(value: T): T =>
  value === undefined ? value : (JSON.parse(JSON.stringify(value)) as T)

const arrayCount = (value: unknown): number => (Array.isArray(value) ? value.length : 0)

export function countSignalsInProfile(profile: RoutingProfileLike | undefined): {
  total: number
  byType: Record<string, number>
} {
  const byType: Record<string, number> = {}
  let total = 0
  for (const [type, value] of Object.entries(profile?.signals ?? {})) {
    const count = arrayCount(value)
    if (count === 0) continue
    byType[type] = count
    total += count
  }
  return { total, byType }
}

export function countProjectionsInProfile(profile: RoutingProfileLike | undefined): number {
  return Object.values(profile?.projections ?? {}).reduce<number>(
    (total, value) => total + (Array.isArray(value) ? value.length : value ? 1 : 0),
    0,
  )
}

export function hasRoutingProfileContent(profile: RoutingProfileLike | undefined): boolean {
  return (
    countSignalsInProfile(profile).total > 0 ||
    countProjectionsInProfile(profile) > 0 ||
    arrayCount(profile?.decisions) > 0
  )
}

function defaultRoutingProfile(config: RoutingScopedConfigLike): RoutingProfileLike {
  return {
    ...(config.routing ?? {}),
    signals: config.routing?.signals ?? config.signals,
    projections: config.routing?.projections ?? config.projections,
    decisions: config.routing?.decisions ?? config.decisions ?? [],
  }
}

function entrypointNamesForRecipe(
  config: RoutingScopedConfigLike,
  recipeName: string,
): string[] {
  return [
    ...new Set(
      (config.entrypoints ?? [])
        .filter((entrypoint) => entrypoint.recipe === recipeName)
        .flatMap((entrypoint) => entrypoint.model_names)
        .map((name) => name.trim())
        .filter(Boolean),
    ),
  ]
}

export function listRoutingScopes(config: RoutingScopedConfigLike | null): RoutingScope[] {
  if (!config) return []

  const recipes = config.recipes ?? []
  const explicitDefault = recipes.find((recipe) => recipe.name === DEFAULT_ROUTING_SCOPE_ID)
  const namedRecipes = recipes.filter((recipe) => recipe.name !== DEFAULT_ROUTING_SCOPE_ID)
  const defaultRouting = explicitDefault?.routing ?? defaultRoutingProfile(config)
  const includeDefault =
    Boolean(explicitDefault) || hasRoutingProfileContent(defaultRouting) || namedRecipes.length === 0

  const scopes: RoutingScope[] = []
  if (includeDefault) {
    scopes.push({
      id: DEFAULT_ROUTING_SCOPE_ID,
      label: 'Default routing',
      description: explicitDefault?.description ?? 'Top-level routing profile.',
      entrypointModelNames: entrypointNamesForRecipe(config, DEFAULT_ROUTING_SCOPE_ID),
      routing: defaultRouting,
      isDefault: true,
      source: explicitDefault ? 'recipe' : 'routing',
    })
  }

  for (const recipe of namedRecipes) {
    const entrypointModelNames = entrypointNamesForRecipe(config, recipe.name)
    scopes.push({
      id: recipe.name,
      label: recipe.name,
      description: recipe.description,
      entrypointModelNames,
      routing: recipe.routing,
      isDefault: false,
      source: 'recipe',
    })
  }
  return scopes
}

export function resolveRoutingScope(
  config: RoutingScopedConfigLike | null,
  scopeId?: string | null,
): RoutingScope | null {
  const scopes = listRoutingScopes(config)
  return scopes.find((scope) => scope.id === scopeId) ?? scopes[0] ?? null
}

export function projectConfigForRoutingScope<T extends RoutingScopedConfigLike>(
  config: T,
  scopeId?: string | null,
): T {
  const scope = resolveRoutingScope(config, scopeId)
  if (!scope) return cloneValue(config)

  const projected = cloneValue(config)
  const routing = cloneValue(scope.routing)
  projected.routing = routing
  projected.signals = cloneValue(routing.signals)
  projected.projections = cloneValue(routing.projections)
  projected.decisions = cloneValue(routing.decisions ?? [])
  return projected
}

export function applyRoutingScopeProjection<T extends RoutingScopedConfigLike>(
  baseConfig: T,
  projectedConfig: T,
  scopeId?: string | null,
): T {
  const scope = resolveRoutingScope(baseConfig, scopeId)
  if (!scope) return cloneValue(projectedConfig)

  const next = cloneValue(baseConfig)
  const routing: RoutingProfileLike = {
    ...cloneValue(scope.routing),
    signals: cloneValue(projectedConfig.signals ?? projectedConfig.routing?.signals),
    projections: cloneValue(projectedConfig.projections ?? projectedConfig.routing?.projections),
    decisions: cloneValue(projectedConfig.decisions ?? projectedConfig.routing?.decisions ?? []),
  }

  if (scope.source === 'routing') {
    next.routing = routing
    next.signals = cloneValue(routing.signals)
    next.projections = cloneValue(routing.projections)
    next.decisions = cloneValue(routing.decisions ?? [])
    return next
  }

  const recipe = next.recipes?.find((candidate) => candidate.name === scope.id)
  if (!recipe) {
    throw new Error(`Routing recipe "${scope.id}" no longer exists.`)
  }
  recipe.routing = routing
  return next
}

export function collectScopedDecisions<T = unknown>(
  config: RoutingScopedConfigLike | null,
): Array<ScopedRoutingValue<T>> {
  return listRoutingScopes(config).flatMap((scope) =>
    (scope.routing.decisions ?? []).map((decision) => ({
      scope,
      value: decision as T,
    })),
  )
}

export function countSignalsAcrossScopes(config: RoutingScopedConfigLike | null): {
  total: number
  byType: Record<string, number>
} {
  const byType: Record<string, number> = {}
  let total = 0
  for (const scope of listRoutingScopes(config)) {
    const counts = countSignalsInProfile(scope.routing)
    total += counts.total
    for (const [type, count] of Object.entries(counts.byType)) {
      byType[type] = (byType[type] ?? 0) + count
    }
  }
  return { total, byType }
}
