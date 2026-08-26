import {
  assertManagementApiResponseSchema,
  type RoutingCatalog,
  type RoutingCatalogEntrypoint,
  type RoutingCatalogModel,
  type RoutingCatalogRecipe,
} from '../generated/managementApiContract'

import { managementOperationRequest } from './managementApiContract'
import { buildManagedRoutingSnapshot, type ManagedRoutingSnapshot } from './managedRoutingSnapshot'
import type {
  RoutingEntrypoint,
  RoutingEntrypointRule,
  RoutingModelCardView,
  RoutingRecipe,
} from './routingManagementApi'

export type KeyScopedRoutingCatalog = RoutingCatalog
export type KeyScopedRoutingModel = RoutingCatalogModel
export type KeyScopedRoutingRecipe = RoutingCatalogRecipe
export type KeyScopedRoutingEntrypoint = RoutingCatalogEntrypoint

/**
 * Apply the one semantic invariant that crosses generated wire resources.
 * OpenAPI owns every field-level shape check; this mapper only verifies that
 * the visible Entrypoint graph cannot reference a hidden Model.
 */
export function assertKeyScopedRoutingCatalog(value: unknown): KeyScopedRoutingCatalog {
  const catalog = assertManagementApiResponseSchema('RoutingCatalog', value)
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

function readonlySignal(type: string, name: string): Record<string, unknown> {
  const base = { name }
  switch (type) {
    case 'keywords':
      return { ...base, operator: 'OR', keywords: [], case_sensitive: false }
    case 'embeddings':
      return { ...base, threshold: 0, candidates: [], aggregation_method: 'max' }
    case 'domains':
      return { ...base, description: '', mmlu_categories: [] }
    case 'context':
      return { ...base, min_tokens: '', max_tokens: '' }
    case 'complexity':
      return { ...base, threshold: 0, easy: { candidates: [] }, hard: { candidates: [] } }
    case 'role_bindings':
      return { ...base, role: '', subjects: [] }
    case 'jailbreak':
      return { ...base, threshold: 0, include_history: false }
    case 'pii':
      return { ...base, threshold: 0, pii_types_allowed: [], include_history: false }
    case 'kb':
      return { ...base, kb: '', target: { kind: 'label', value: '' }, match: 'best' }
    case 'metadata':
      return { ...base, key: '', predicate: {} }
    case 'classifiers':
      return { ...base, type: 'local', labels: [] }
    default:
      return base
  }
}

function readonlySignals(recipe: RoutingCatalogRecipe): Record<string, unknown[]> {
  const result: Record<string, unknown[]> = {}
  for (const signal of recipe.signals) {
    const group = result[signal.type] ?? []
    group.push(readonlySignal(signal.type, signal.name))
    result[signal.type] = group
  }
  return result
}

function readonlyProjections(recipe: RoutingCatalogRecipe) {
  const partitions: Record<string, unknown>[] = []
  const scores: Record<string, unknown>[] = []
  const mappings: Record<string, unknown>[] = []
  for (const projection of recipe.projections) {
    if (projection.type === 'partition') {
      partitions.push({
        name: projection.name,
        semantics: '',
        members: projection.members,
        default: '',
      })
    }
    if (projection.type === 'score') {
      scores.push({
        name: projection.name,
        method: '',
        inputs: projection.inputs.map((input) => ({ ...input, weight: 0 })),
      })
    }
    if (projection.type === 'mapping') {
      mappings.push({
        name: projection.name,
        source: projection.source ?? '',
        method: '',
        outputs: projection.outputs.map((name) => ({ name })),
      })
    }
  }
  return { partitions, scores, mappings }
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
      signals: readonlySignals(recipe),
      projections: readonlyProjections(recipe),
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
    recipeIds: [...new Set(entrypoint.rules.map((rule) => rule.recipeId))].sort(),
    ruleCount: entrypoint.rules.length,
    assignedModelCount: countAssignedModels(entrypoint.rules),
    rules: entrypoint.rules,
    createdAt: '',
    updatedAt: '',
  }))
  return buildManagedRoutingSnapshot(models, recipes, entrypoints)
}
