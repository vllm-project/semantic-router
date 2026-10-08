import type { CanonicalGlobalConfig, CanonicalModelDeployment } from './configPageSupport'
import type { RouterConfig } from './dashboardPageTypes'
import type { ModelRuntimeInventory } from './decisionModelManagement'
import {
  DECISION_RUNTIME_CAPABILITIES,
  DECISION_RUNTIME_CATALOG,
  type DecisionRuntimeCatalogEntry,
} from './decisionRuntimeCatalog'

export interface DecisionRuntimeConsumer {
  id: string
  recipe: string | null
  name: string
  kind: 'question' | 'selector'
  questionType: string
  deployment: string
}

export interface DecisionRuntimeDeploymentRequest {
  entry: DecisionRuntimeCatalogEntry
  name: string
  device: string
  consumerId: string
  existingName?: string
}

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {}
}

export function decisionRuntimeDeclarations(
  config: RouterConfig | null,
): Record<string, CanonicalModelDeployment> {
  return (config?.global as CanonicalGlobalConfig | undefined)?.model_catalog?.deployments ?? {}
}

export function decisionRuntimeConsumers(config: RouterConfig | null): DecisionRuntimeConsumer[] {
  const scopes = [
    { recipe: null, routing: config?.routing },
    ...(config?.recipes ?? []).map((recipe) => ({ recipe: recipe.name, routing: recipe.routing })),
  ]
  return scopes.flatMap(({ recipe, routing }) => [
    ...(routing?.signals?.decision ?? []).flatMap((signal): DecisionRuntimeConsumer[] =>
      signal.name
        ? [
            {
              id: JSON.stringify([recipe, 'question', signal.name]),
              recipe,
              name: signal.name,
              kind: 'question',
              questionType: String(record(signal.question).type ?? ''),
              deployment: String(signal.deployment ?? ''),
            },
          ]
        : [],
    ),
    ...(routing?.decisions ?? []).flatMap((decision): DecisionRuntimeConsumer[] => {
      const algorithm = record(decision.algorithm)
      if (!decision.name || algorithm.type !== 'decision' || !algorithm.decision) return []
      return [
        {
          id: JSON.stringify([recipe, 'selector', decision.name]),
          recipe,
          name: decision.name,
          kind: 'selector',
          questionType: 'choice',
          deployment: String(record(algorithm.decision).deployment ?? ''),
        },
      ]
    }),
  ])
}

export function configuredDecisionRuntimes(
  config: RouterConfig | null,
  inventory: ModelRuntimeInventory | null,
): Array<[string, CanonicalModelDeployment]> {
  const artifacts = new Set(DECISION_RUNTIME_CATALOG.map((entry) => entry.id))
  const bindings = new Set(decisionRuntimeConsumers(config).map((consumer) => consumer.deployment))
  const questionRuntimes = new Set(
    inventory?.deployments
      .filter((runtime) => runtime.surfaces?.includes('question'))
      .map((runtime) => runtime.name),
  )
  return Object.entries(decisionRuntimeDeclarations(config)).filter(
    ([name, deployment]) =>
      deployment.provider === 'model_runtime' &&
      (artifacts.has(deployment.artifact ?? '') || bindings.has(name) || questionRuntimes.has(name)),
  )
}

export function compatibleRuntimeConsumers(config: RouterConfig | null): DecisionRuntimeConsumer[] {
  return decisionRuntimeConsumers(config).filter((consumer) =>
    DECISION_RUNTIME_CAPABILITIES.some((kind) => kind === consumer.questionType),
  )
}

export function consumerLabel(consumer: DecisionRuntimeConsumer): string {
  return `${consumer.recipe ?? 'Default routing'} / ${consumer.name}`
}

// Update a fresh canonical snapshot in one transaction. Unselected consumers,
// recipes, provider credentials, admission policies and the Vela default survive.
export function withDecisionRuntimeDeployment(
  config: RouterConfig,
  request: DecisionRuntimeDeploymentRequest,
): RouterConfig {
  const name = request.name.trim()
  if (!name || name.startsWith('@'))
    throw new Error('Choose a nonempty deployment name without the reserved @ prefix.')
  if (!/^(auto|cpu|cuda(?::\d+)?|rocm(?::\d+)?)$/.test(request.device))
    throw new Error('Choose an automatic, CPU, CUDA or ROCm device.')
  const existing = decisionRuntimeDeclarations(config)[name]
  if (
    existing &&
    (request.existingName !== name ||
      existing.provider !== 'model_runtime' ||
      existing.artifact !== request.entry.id ||
      existing.endpoint)
  ) {
    throw new Error(
      'This deployment name is already in use. Choose another name or manage its existing configuration.',
    )
  }
  if (request.existingName && !existing)
    throw new Error('This deployment was removed. Refresh the catalog before trying again.')
  const consumer = request.consumerId
    ? compatibleRuntimeConsumers(config).find((value) => value.id === request.consumerId)
    : undefined
  if (request.consumerId && !consumer)
    throw new Error(
      'The selected consumer is no longer compatible. Refresh and select a choice, score or noul question, or a decision selector.',
    )
  const next = structuredClone(config)
  const global = (next.global ?? {}) as CanonicalGlobalConfig
  next.global = {
    ...global,
    model_catalog: {
      ...global.model_catalog,
      deployments: {
        ...global.model_catalog?.deployments,
        [name]: {
          ...existing,
          provider: 'model_runtime',
          artifact: request.entry.id,
          ...(existing ? {} : { revision: request.entry.revision }),
          device: request.device,
        },
      },
    },
  }
  if (consumer) {
    const routing =
      consumer.recipe === null
        ? next.routing
        : next.recipes?.find((recipe) => recipe.name === consumer.recipe)?.routing
    if (consumer.kind === 'question') {
      const signal = routing?.signals?.decision?.find((value) => value.name === consumer.name)
      if (!signal) throw new Error('The selected question is no longer available.')
      signal.deployment = name
    } else {
      const decision = routing?.decisions?.find((value) => value.name === consumer.name)
      if (!decision) throw new Error('The selected decision selector is no longer available.')
      const algorithm = record(decision.algorithm)
      decision.algorithm = {
        ...algorithm,
        decision: { ...record(algorithm.decision), deployment: name },
      }
    }
  }
  return next
}
