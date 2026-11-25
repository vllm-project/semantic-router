import type { StatusTone } from '../ui'
import {
  getRouterModelResources,
  getRouterModelState,
  type RouterModelsInfo,
} from '../utils/routerRuntime'
import { listRoutingScopes } from '../utils/routingScopes'
import { decisionModelRuntimeState, type ModelRuntimeInventory } from './decisionModelManagement'
import {
  configuredDecisionModel,
  configuredDecisionDeployment,
  type DecisionModelName,
} from './decisionModelSupport'
import type { RouterConfig } from './dashboardPageTypes'

interface NamedItem {
  name: string
  kind: string
}

export interface IntelligenceRoutingScope {
  id: string
  label: string
  entrypoints: string[]
  questions: NamedItem[]
  signals: Array<{ type: string; names: string[] }>
  projections: NamedItem[]
}

function record(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined
}

function namedItems(value: unknown): Array<Record<string, unknown> & { name: string }> {
  if (!Array.isArray(value)) return []
  return value.flatMap((item) => {
    const entry = record(item)
    return typeof entry?.name === 'string' && entry.name.trim()
      ? [{ ...entry, name: entry.name }]
      : []
  })
}

export function buildIntelligenceRoutingScopes(config: RouterConfig): IntelligenceRoutingScope[] {
  return listRoutingScopes(config).map((scope) => ({
    id: scope.id,
    label: scope.label,
    entrypoints: scope.entrypointModelNames,
    questions: namedItems(scope.routing.signals?.systemone).map((signal) => ({
      name: signal.name,
      kind:
        typeof record(signal.question)?.type === 'string'
          ? String(record(signal.question)?.type)
          : 'question',
    })),
    signals: Object.entries(scope.routing.signals ?? {})
      .filter(([type]) => type !== 'systemone')
      .map(([type, values]) => ({ type, names: namedItems(values).map(({ name }) => name) }))
      .filter(({ names }) => names.length > 0),
    projections: Object.entries(scope.routing.projections ?? {}).flatMap(([kind, values]) =>
      namedItems(values).map(({ name }) => ({ name, kind })),
    ),
  }))
}

export interface DecisionRuntimeSummary {
  model: DecisionModelName
  state: 'ready' | 'attention' | 'unreported'
  resources: number
  bindings: number
}

export function getDecisionRuntimeSummary(
  config: RouterConfig,
  inventory?: RouterModelsInfo | null,
): DecisionRuntimeSummary {
  const model = configuredDecisionModel(config)
  // Only the runtime's declared deployment links prepared consumers to the
  // configured decision model. A healthy Router or a matching artifact name
  // cannot establish readiness of this model.
  const bindings = (inventory?.models ?? []).filter(
    (entry) =>
      entry.metadata?.provider === 'model_runtime' &&
      entry.metadata.deployment === configuredDecisionDeployment(config),
  )
  const resources = getRouterModelResources(bindings)
  const state =
    resources.length === 0
      ? 'unreported'
      : resources.every(
            ({ model: entry }) => entry.loaded && getRouterModelState(entry) === 'ready',
          )
        ? 'ready'
        : 'attention'
  return { model, state, resources: resources.length, bindings: bindings.length }
}

export interface DecisionModelHealth {
  model: DecisionModelName
  deployment: string
  tone: StatusTone
  label: string
  /** Where it runs and how much of it is up, or why its state is what it is. */
  detail: string
}

const RUNTIME_TONE: Record<string, StatusTone> = {
  Ready: 'success',
  'Degraded capacity': 'warning',
  'Needs attention': 'warning',
}

/**
 * The configured decision model's deployment state as Models › Decision models
 * reads it from the model runtime; the Router's bindings to the deployment stand
 * in when the runtime does not list it.
 */
export function decisionModelHealth(
  config: RouterConfig,
  inventory?: ModelRuntimeInventory | null,
  routerModels?: RouterModelsInfo | null,
): DecisionModelHealth {
  const model = configuredDecisionModel(config)
  const deployment = configuredDecisionDeployment(config)
  const observed = inventory?.deployments?.find((entry) => entry.name === deployment)
  if (observed && inventory) {
    const label = decisionModelRuntimeState(deployment, inventory)
    const replicas = observed.desired_replicas
      ? `${observed.ready_replicas ?? 0} of ${observed.desired_replicas} replicas ready`
      : ''
    return {
      model,
      deployment,
      tone: RUNTIME_TONE[label] ?? 'neutral',
      label,
      detail: observed.ready
        ? [observed.device, replicas].filter(Boolean).join(' · ') ||
          'The model runtime reports it ready'
        : observed.reason ||
          `The model runtime reports it ${(observed.state || 'not ready').replace(/_/g, ' ')}`,
    }
  }
  const bindings = getDecisionRuntimeSummary(config, routerModels)
  if (bindings.state === 'ready')
    return {
      model,
      deployment,
      tone: 'success',
      label: 'Ready',
      detail: 'The Router reports it ready',
    }
  if (bindings.state === 'attention')
    return {
      model,
      deployment,
      tone: 'warning',
      label: 'Needs attention',
      detail: 'The Router reports it is not ready',
    }
  return {
    model,
    deployment,
    tone: 'neutral',
    label: 'Not reported',
    detail: `The model runtime has not reported the ${deployment} deployment yet`,
  }
}
