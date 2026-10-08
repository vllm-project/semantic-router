import {
  getRouterModelResources,
  getRouterModelState,
  type RouterModelsInfo,
} from '../utils/routerRuntime'
import { listRoutingScopes } from '../utils/routingScopes'
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
    questions: namedItems(scope.routing.signals?.decision).map((signal) => ({
      name: signal.name,
      kind:
        typeof record(signal.question)?.type === 'string'
          ? String(record(signal.question)?.type)
          : 'question',
    })),
    signals: Object.entries(scope.routing.signals ?? {})
      .filter(([type]) => type !== 'decision')
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
