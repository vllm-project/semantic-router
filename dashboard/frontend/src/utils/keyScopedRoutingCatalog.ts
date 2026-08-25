import {
  managementOperationRequest,
  hasOnlyKeys,
  isNonEmptyString,
  isRecord,
  isStringArray,
} from './managementApiContract'
import { buildManagedRoutingSnapshot, type ManagedRoutingSnapshot } from './managedRoutingSnapshot'
import type {
  RoutingAssignmentModel,
  RoutingAssignmentReasoning,
  RoutingAssignmentSet,
  RoutingClaimValue,
  RoutingDecision,
  RoutingEntrypoint,
  RoutingEntrypointRule,
  RoutingFallbackPolicy,
  RoutingMatcher,
  RoutingModelCardView,
  RoutingRecipe,
} from './routingManagementApi'

export interface KeyScopedRoutingCatalog {
  keyId: string
  policyRevision: number
  policyDigest: string
  routingRevision: number
  routingDigest: string
  models: KeyScopedRoutingModel[]
  recipes: KeyScopedRoutingRecipe[]
  entrypoints: KeyScopedRoutingEntrypoint[]
}

export interface KeyScopedRoutingModel {
  id: string
  revision: number
  name: string
  aliases: string[]
  paramSize?: string
  contextWindowSize?: number
  description?: string
  capabilities: string[]
  reasoning?: { type?: string; efforts?: string[] }
  loras: string[]
  qualityScore?: number
  modality?: string
  tags: string[]
  pricing: {
    inputCostPerMillionTokens: string | null
    outputCostPerMillionTokens: string | null
    cacheReadCostPerMillionTokens: string | null
    cacheWriteCostPerMillionTokens: string | null
  }
}

export interface KeyScopedRoutingRecipe {
  id: string
  revision: number
  name: string
  description?: string
  decisions: RoutingDecision[]
}

export interface KeyScopedRoutingEntrypoint {
  id: string
  revision: number
  name: string
  aliases: string[]
  rules: RoutingEntrypointRule[]
}

const finiteInteger = (value: unknown): value is number =>
  typeof value === 'number' && Number.isSafeInteger(value)

const positiveRevision = (value: unknown): value is number => finiteInteger(value) && value > 0

const optionalString = (value: unknown): value is string | undefined =>
  value === undefined || typeof value === 'string'

function assertReasoningFamily(value: unknown): { type?: string; efforts?: string[] } {
  if (
    !isRecord(value) ||
    !hasOnlyKeys(value, [], ['type', 'efforts']) ||
    (value.type !== undefined && typeof value.type !== 'string') ||
    (value.efforts !== undefined && !isStringArray(value.efforts))
  ) {
    throw new Error('Router returned an invalid key-scoped reasoning family.')
  }
  return value as { type?: string; efforts?: string[] }
}

function assertPricing(value: unknown): KeyScopedRoutingModel['pricing'] {
  const fields = [
    'inputCostPerMillionTokens',
    'outputCostPerMillionTokens',
    'cacheReadCostPerMillionTokens',
    'cacheWriteCostPerMillionTokens',
  ] as const
  if (
    !isRecord(value) ||
    !hasOnlyKeys(value, fields) ||
    !fields.every((field) => value[field] === null || typeof value[field] === 'string')
  ) {
    throw new Error('Router returned invalid key-scoped Model pricing.')
  }
  return value as unknown as KeyScopedRoutingModel['pricing']
}

function assertModel(value: unknown): KeyScopedRoutingModel {
  if (
    !isRecord(value) ||
    !hasOnlyKeys(
      value,
      ['id', 'revision', 'name', 'aliases', 'capabilities', 'loras', 'tags', 'pricing'],
      ['paramSize', 'contextWindowSize', 'description', 'reasoning', 'qualityScore', 'modality'],
    ) ||
    !isNonEmptyString(value.id) ||
    !positiveRevision(value.revision) ||
    !isNonEmptyString(value.name) ||
    !isStringArray(value.aliases) ||
    !optionalString(value.paramSize) ||
    (value.contextWindowSize !== undefined &&
      (!finiteInteger(value.contextWindowSize) || value.contextWindowSize < 0)) ||
    !optionalString(value.description) ||
    !isStringArray(value.capabilities) ||
    !isStringArray(value.loras) ||
    (value.qualityScore !== undefined && typeof value.qualityScore !== 'number') ||
    !optionalString(value.modality) ||
    !isStringArray(value.tags)
  ) {
    throw new Error('Router returned an invalid key-scoped Model.')
  }
  const reasoning =
    value.reasoning === undefined ? undefined : assertReasoningFamily(value.reasoning)
  return {
    ...(value as unknown as KeyScopedRoutingModel),
    ...(reasoning ? { reasoning } : {}),
    pricing: assertPricing(value.pricing),
  }
}

function assertDecision(value: unknown): RoutingDecision {
  if (
    !isRecord(value) ||
    !hasOnlyKeys(value, ['id', 'name', 'dispatchCardinality']) ||
    !isNonEmptyString(value.id) ||
    !isNonEmptyString(value.name) ||
    (value.dispatchCardinality !== 'single' && value.dispatchCardinality !== 'multi')
  ) {
    throw new Error('Router returned an invalid key-scoped Recipe decision.')
  }
  return value as unknown as RoutingDecision
}

function assertClaimValue(value: unknown): RoutingClaimValue {
  if (!isRecord(value) || !isNonEmptyString(value.kind)) {
    throw new Error('Router returned an invalid key-scoped claim matcher.')
  }
  const valid =
    (value.kind === 'string' &&
      hasOnlyKeys(value, ['kind', 'string']) &&
      typeof value.string === 'string') ||
    (value.kind === 'boolean' &&
      hasOnlyKeys(value, ['kind', 'boolean']) &&
      typeof value.boolean === 'boolean') ||
    (value.kind === 'integer' &&
      hasOnlyKeys(value, ['kind', 'integer']) &&
      finiteInteger(value.integer))
  if (!valid) throw new Error('Router returned an invalid key-scoped claim matcher.')
  return value as unknown as RoutingClaimValue
}

function assertMatcher(value: unknown): RoutingMatcher {
  if (!isRecord(value) || !hasOnlyKeys(value, [], ['claim', 'exactPath', 'pathPrefix'])) {
    throw new Error('Router returned an invalid key-scoped matcher.')
  }
  const result: RoutingMatcher = {}
  if (value.claim !== undefined) {
    if (
      !isRecord(value.claim) ||
      !hasOnlyKeys(value.claim, ['name', 'value']) ||
      !isNonEmptyString(value.claim.name)
    ) {
      throw new Error('Router returned an invalid key-scoped matcher.')
    }
    result.claim = { name: value.claim.name, value: assertClaimValue(value.claim.value) }
  }
  if (value.exactPath !== undefined) {
    if (typeof value.exactPath !== 'string')
      throw new Error('Router returned an invalid key-scoped matcher.')
    result.exactPath = value.exactPath
  }
  if (value.pathPrefix !== undefined) {
    if (typeof value.pathPrefix !== 'string')
      throw new Error('Router returned an invalid key-scoped matcher.')
    result.pathPrefix = value.pathPrefix
  }
  return result
}

function assertAssignmentReasoning(value: unknown): RoutingAssignmentReasoning {
  if (
    !isRecord(value) ||
    !hasOnlyKeys(value, ['enabled'], ['effort', 'description']) ||
    typeof value.enabled !== 'boolean' ||
    !optionalString(value.effort) ||
    !optionalString(value.description)
  ) {
    throw new Error('Router returned invalid key-scoped assignment reasoning.')
  }
  return value as unknown as RoutingAssignmentReasoning
}

function assertAssignmentModel(value: unknown): RoutingAssignmentModel {
  if (
    !isRecord(value) ||
    !hasOnlyKeys(
      value,
      ['modelId', 'modelRevision', 'priority', 'weight'],
      ['loraName', 'reasoning'],
    ) ||
    !isNonEmptyString(value.modelId) ||
    !positiveRevision(value.modelRevision) ||
    !finiteInteger(value.priority) ||
    !isNonEmptyString(value.weight) ||
    !optionalString(value.loraName)
  ) {
    throw new Error('Router returned an invalid key-scoped Model assignment.')
  }
  const reasoning =
    value.reasoning === undefined ? undefined : assertAssignmentReasoning(value.reasoning)
  return { ...(value as unknown as RoutingAssignmentModel), ...(reasoning ? { reasoning } : {}) }
}

function assertFallback(value: unknown): RoutingFallbackPolicy {
  if (
    !isRecord(value) ||
    !hasOnlyKeys(value, ['strategy', 'on']) ||
    value.strategy !== 'priority' ||
    !Array.isArray(value.on) ||
    !value.on.every(
      (trigger) => trigger === 'unavailable' || trigger === 'overloaded' || trigger === 'timeout',
    )
  ) {
    throw new Error('Router returned an invalid key-scoped fallback policy.')
  }
  return value as unknown as RoutingFallbackPolicy
}

function assertAssignmentSet(value: unknown): RoutingAssignmentSet {
  if (
    !isRecord(value) ||
    !hasOnlyKeys(value, ['models'], ['fallback']) ||
    !Array.isArray(value.models)
  ) {
    throw new Error('Router returned an invalid key-scoped decision assignment.')
  }
  return {
    models: value.models.map(assertAssignmentModel),
    ...(value.fallback === undefined ? {} : { fallback: assertFallback(value.fallback) }),
  }
}

function assertRule(value: unknown): RoutingEntrypointRule {
  if (
    !isRecord(value) ||
    !hasOnlyKeys(
      value,
      ['id', 'name', 'recipeId', 'recipeRevision', 'assignments'],
      ['matchers'],
    ) ||
    !isNonEmptyString(value.id) ||
    !isNonEmptyString(value.name) ||
    !isNonEmptyString(value.recipeId) ||
    !positiveRevision(value.recipeRevision) ||
    !isRecord(value.assignments) ||
    (value.matchers !== undefined && !Array.isArray(value.matchers))
  ) {
    throw new Error('Router returned an invalid key-scoped Entrypoint rule.')
  }
  return {
    id: value.id,
    name: value.name,
    ...(value.matchers === undefined ? {} : { matchers: value.matchers.map(assertMatcher) }),
    recipeId: value.recipeId,
    recipeRevision: value.recipeRevision,
    assignments: Object.fromEntries(
      Object.entries(value.assignments).map(([decisionId, assignment]) => [
        decisionId,
        assertAssignmentSet(assignment),
      ]),
    ),
  }
}

function assertRecipe(value: unknown): KeyScopedRoutingRecipe {
  if (
    !isRecord(value) ||
    !hasOnlyKeys(value, ['id', 'revision', 'name', 'decisions'], ['description']) ||
    !isNonEmptyString(value.id) ||
    !positiveRevision(value.revision) ||
    !isNonEmptyString(value.name) ||
    !optionalString(value.description) ||
    !Array.isArray(value.decisions)
  ) {
    throw new Error('Router returned an invalid key-scoped Recipe.')
  }
  return {
    ...(value as unknown as KeyScopedRoutingRecipe),
    decisions: value.decisions.map(assertDecision),
  }
}

function assertEntrypoint(value: unknown): KeyScopedRoutingEntrypoint {
  if (
    !isRecord(value) ||
    !hasOnlyKeys(value, ['id', 'revision', 'name', 'aliases', 'rules']) ||
    !isNonEmptyString(value.id) ||
    !positiveRevision(value.revision) ||
    !isNonEmptyString(value.name) ||
    !isStringArray(value.aliases) ||
    !Array.isArray(value.rules)
  ) {
    throw new Error('Router returned an invalid key-scoped Entrypoint.')
  }
  return { ...(value as unknown as KeyScopedRoutingEntrypoint), rules: value.rules.map(assertRule) }
}

export function assertKeyScopedRoutingCatalog(value: unknown): KeyScopedRoutingCatalog {
  if (
    !isRecord(value) ||
    !hasOnlyKeys(value, [
      'keyId',
      'policyRevision',
      'policyDigest',
      'routingRevision',
      'routingDigest',
      'models',
      'recipes',
      'entrypoints',
    ]) ||
    !isNonEmptyString(value.keyId) ||
    !positiveRevision(value.policyRevision) ||
    !isNonEmptyString(value.policyDigest) ||
    !/^[a-f0-9]{64}$/.test(value.policyDigest) ||
    !positiveRevision(value.routingRevision) ||
    !isNonEmptyString(value.routingDigest) ||
    !/^[a-f0-9]{64}$/.test(value.routingDigest) ||
    !Array.isArray(value.models) ||
    !Array.isArray(value.recipes) ||
    !Array.isArray(value.entrypoints)
  ) {
    throw new Error('Router returned an invalid key-scoped routing catalog.')
  }
  const catalog: KeyScopedRoutingCatalog = {
    keyId: value.keyId,
    policyRevision: value.policyRevision,
    policyDigest: value.policyDigest,
    routingRevision: value.routingRevision,
    routingDigest: value.routingDigest,
    models: value.models.map(assertModel),
    recipes: value.recipes.map(assertRecipe),
    entrypoints: value.entrypoints.map(assertEntrypoint),
  }
  const modelIds = new Set(catalog.models.map((model) => model.id))
  for (const entrypoint of catalog.entrypoints) {
    for (const rule of entrypoint.rules) {
      for (const assignment of Object.values(rule.assignments)) {
        if (assignment.models.some((model) => !modelIds.has(model.modelId))) {
          throw new Error('Router returned an inconsistent key-scoped routing catalog.')
        }
      }
    }
  }
  return catalog
}

export async function fetchKeyScopedRoutingCatalog(
  keyId: string,
  signal?: AbortSignal,
): Promise<KeyScopedRoutingCatalog> {
  return assertKeyScopedRoutingCatalog(
    await managementOperationRequest('getApiKeysByKeyIdRoutingCatalog', {
      pathParameters: { keyId },
      signal,
    }),
  )
}

function countAssignedModels(rules: RoutingEntrypointRule[]): number {
  return new Set(
    rules.flatMap((rule) =>
      Object.values(rule.assignments).flatMap((assignment) =>
        assignment.models.map((model) => model.modelId),
      ),
    ),
  ).size
}

export function keyScopedCatalogSnapshot(catalog: KeyScopedRoutingCatalog): ManagedRoutingSnapshot {
  const models: RoutingModelCardView[] = catalog.models.map((model) => ({
    id: model.id,
    name: model.name,
    card: {
      aliases: model.aliases,
      ...(model.paramSize === undefined ? {} : { paramSize: model.paramSize }),
      ...(model.contextWindowSize === undefined
        ? {}
        : { contextWindowSize: model.contextWindowSize }),
      ...(model.description === undefined ? {} : { description: model.description }),
      capabilities: model.capabilities,
      ...(model.reasoning === undefined ? {} : { reasoning: model.reasoning }),
      loras: model.loras,
      ...(model.qualityScore === undefined ? {} : { qualityScore: model.qualityScore }),
      ...(model.modality === undefined ? {} : { modality: model.modality }),
      tags: model.tags,
    },
  }))
  const recipes: RoutingRecipe[] = catalog.recipes.map((recipe) => ({
    id: recipe.id,
    name: recipe.name,
    ...(recipe.description === undefined ? {} : { description: recipe.description }),
    status: 'active',
    revision: recipe.revision,
    recipeRevision: recipe.revision,
    origin: 'distribution',
    immutable: true,
    decisions: recipe.decisions,
    document: {
      decisions: recipe.decisions.map((decision) => ({
        id: decision.id,
        name: decision.name,
        dispatch_cardinality: decision.dispatchCardinality,
      })),
    },
    createdAt: '',
    updatedAt: '',
  }))
  const entrypoints: RoutingEntrypoint[] = catalog.entrypoints.map((entrypoint) => ({
    id: entrypoint.id,
    name: entrypoint.name,
    status: 'active',
    revision: entrypoint.revision,
    entrypointRevision: entrypoint.revision,
    aliases: entrypoint.aliases,
    ruleCount: entrypoint.rules.length,
    assignedModelCount: countAssignedModels(entrypoint.rules),
    rules: entrypoint.rules,
    createdAt: '',
    updatedAt: '',
  }))
  return buildManagedRoutingSnapshot(models, recipes, entrypoints)
}
