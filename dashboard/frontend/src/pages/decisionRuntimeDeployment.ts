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
  /** The deployment it names; empty when it asks the default decision model. */
  deployment: string
}

export interface DecisionRuntimeDeploymentRequest {
  name: string
  /** Declares the catalog release under name; omitted to bind an existing deployment as it is. */
  entry?: DecisionRuntimeCatalogEntry
  device?: string
  /** Managed workers for a new declaration; one keeps a single device. */
  replicas?: number
  existingName?: string
  /** What the deployment answers when no entry says; the shared System One kinds otherwise. */
  questionTypes?: readonly string[]
  consumerIds?: readonly string[]
  makeDefault?: boolean
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
    ...(routing?.signals?.systemone ?? []).flatMap((signal): DecisionRuntimeConsumer[] =>
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
      if (!decision.name || algorithm.type !== 'systemone' || !algorithm.systemone) return []
      return [
        {
          id: JSON.stringify([recipe, 'selector', decision.name]),
          recipe,
          name: decision.name,
          kind: 'selector',
          questionType: 'choice',
          deployment: String(record(algorithm.systemone).deployment ?? ''),
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
      (artifacts.has(deployment.artifact ?? '') ||
        bindings.has(name) ||
        questionRuntimes.has(name)),
  )
}

export function compatibleRuntimeConsumers(
  config: RouterConfig | null,
  questionTypes: readonly string[] = DECISION_RUNTIME_CAPABILITIES,
): DecisionRuntimeConsumer[] {
  return decisionRuntimeConsumers(config).filter((consumer) =>
    questionTypes.includes(consumer.questionType),
  )
}

// Update a fresh canonical snapshot in one transaction. Unselected consumers,
// recipes, provider credentials, admission policies and the default survive
// unless the request makes this deployment the default.
export function withDecisionRuntimeDeployment(
  config: RouterConfig,
  request: DecisionRuntimeDeploymentRequest,
): RouterConfig {
  const name = request.name.trim()
  if (!name || name.startsWith('@'))
    throw new Error('Choose a nonempty deployment name without the reserved @ prefix.')
  const existing = decisionRuntimeDeclarations(config)[name]
  if (request.existingName && !existing)
    throw new Error('This deployment was removed. Refresh the catalog before trying again.')
  if (!request.entry && request.existingName !== name)
    throw new Error('Choose a deployment from the catalog to bind.')
  if (
    request.entry &&
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
  const device = request.device ?? 'auto'
  if (request.entry && !/^(auto|cpu|cuda(?::\d+)?|rocm(?::\d+)?)$/.test(device))
    throw new Error('Choose an automatic, CPU, CUDA or ROCm device.')
  const replicas = Math.floor(request.replicas ?? 1)
  if (replicas < 1 || replicas > 64) throw new Error('Choose between 1 and 64 replicas.')
  const compatible = compatibleRuntimeConsumers(
    config,
    request.entry?.questionTypes ?? request.questionTypes,
  )
  const consumers = (request.consumerIds ?? []).map((id) => {
    const consumer = compatible.find((value) => value.id === id)
    if (!consumer)
      throw new Error(
        'A selected consumer is no longer compatible. Refresh and select questions this model answers, or a System One selector.',
      )
    return consumer
  })
  const next = structuredClone(config)
  const global = (next.global ?? {}) as CanonicalGlobalConfig
  const placement: Partial<CanonicalModelDeployment> =
    replicas > 1 && !existing
      ? { replicas: Array.from({ length: replicas }, () => ({ device })) }
      : { device }
  next.global = {
    ...global,
    model_catalog: {
      ...global.model_catalog,
      deployments: request.entry
        ? {
            ...global.model_catalog?.deployments,
            [name]: {
              ...existing,
              provider: 'model_runtime',
              artifact: request.entry.id,
              ...(existing ? {} : { revision: request.entry.revision }),
              ...(existing?.replicas?.length ? {} : placement),
            },
          }
        : global.model_catalog?.deployments,
      ...(request.makeDefault
        ? {
            system: { ...global.model_catalog?.system, decision_model: { deployment: name } },
          }
        : {}),
    },
  }
  for (const consumer of consumers) {
    const routing =
      consumer.recipe === null
        ? next.routing
        : next.recipes?.find((recipe) => recipe.name === consumer.recipe)?.routing
    if (consumer.kind === 'question') {
      const signal = routing?.signals?.systemone?.find((value) => value.name === consumer.name)
      if (!signal) throw new Error('The selected question is no longer available.')
      signal.deployment = name
    } else {
      const decision = routing?.decisions?.find((value) => value.name === consumer.name)
      if (!decision) throw new Error('The selected System One selector is no longer available.')
      const algorithm = record(decision.algorithm)
      decision.algorithm = {
        ...algorithm,
        systemone: { ...record(algorithm.systemone), deployment: name },
      }
    }
  }
  return next
}
