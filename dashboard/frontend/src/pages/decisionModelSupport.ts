// The Router's decision model: global.model_catalog.system.decision_model.
// It answers the built-in signals and every routing.signals.decision question
// that names no deployment (src/semantic-router/pkg/config/decision_model.go).

export const DECISION_MODELS = [
  'Vela-2.0-0.3B',
  'Vela-2.0-0.8B',
  'Vela-2.0-4B',
  'Vela-2.0-9B',
  'Vela-1.0',
] as const

export type DecisionModelName = (typeof DECISION_MODELS)[number]

export const DEFAULT_DECISION_MODEL: DecisionModelName = 'Vela-2.0-0.3B'

export interface DecisionModelOption {
  name: DecisionModelName
  label: string
  hardware: string
  summary: string
}

export const DECISION_MODEL_OPTIONS: readonly DecisionModelOption[] = [
  {
    name: 'Vela-2.0-0.3B',
    label: 'Vela 2.0 0.3B',
    hardware: 'CPU or GPU',
    summary: 'The default: tens of milliseconds per request on a CPU.',
  },
  {
    name: 'Vela-2.0-0.8B',
    label: 'Vela 2.0 0.8B',
    hardware: 'GPU recommended',
    summary: 'Runs on a CPU at seconds per request; tens of milliseconds on a GPU.',
  },
  {
    name: 'Vela-2.0-4B',
    label: 'Vela 2.0 4B',
    hardware: 'GPU only, about 17 GB',
    summary: 'Needs a GPU in the Router: serve with --platform amd or nvidia.',
  },
  {
    name: 'Vela-2.0-9B',
    label: 'Vela 2.0 9B',
    hardware: 'GPU only, about 32 GB',
    summary: 'Needs a GPU in the Router: serve with --platform amd or nvidia.',
  },
  {
    name: 'Vela-1.0',
    label: 'Vela 1.0 specialists',
    hardware: 'CPU or GPU',
    summary: 'One encoder per signal. A decision question then needs its own deployment.',
  },
]

export const DECISION_MODEL_HINT = DECISION_MODEL_OPTIONS.map(
  (option) => `${option.name}: ${option.hardware}`,
).join('; ')

function record(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null
}

// configuredDecisionModel returns the decision model a config names, matched
// case-insensitively as the Router does, or the default.
export function configuredDecisionModel(config: unknown): DecisionModelName {
  const system = record(record(record(record(config)?.global)?.model_catalog)?.system)
  const value = typeof system?.decision_model === 'string' ? system.decision_model.trim() : ''
  return (
    DECISION_MODELS.find((name) => name.toLowerCase() === value.toLowerCase()) ??
    DEFAULT_DECISION_MODEL
  )
}

// withDecisionModel returns a copy of a config that names the decision model.
export function withDecisionModel<T extends Record<string, unknown>>(
  config: T,
  name: DecisionModelName,
): T {
  const global = { ...(record(config.global) ?? {}) }
  const catalog = { ...(record(global.model_catalog) ?? {}) }
  catalog.system = { ...(record(catalog.system) ?? {}), decision_model: name }
  global.model_catalog = catalog
  return { ...config, global }
}
