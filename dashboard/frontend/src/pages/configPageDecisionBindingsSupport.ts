import type { EditFormData, FieldConfig } from '../components/EditModal'
import type { CanonicalGlobalConfig, ConfigData } from './configPageSupport'
import { configuredDecisionModel } from './decisionModelSupport'
import { SIGNAL_BINDING_KEYS, bindingsBySignal, isSignalBindingKey } from './modelBindings'

const SIGNAL_LABELS: Record<(typeof SIGNAL_BINDING_KEYS)[number], string> = {
  domain: 'Domain',
  fact_check: 'Fact check',
  user_feedback: 'User feedback',
  jailbreak: 'Jailbreak',
  safety: 'Safety',
  pii: 'PII',
  hallucination: 'Hallucination',
  refusal: 'Refusal',
  turn_relation: 'Turn relation',
  complexity: 'Complexity',
  modality: 'Modality',
}

const SUMMARY_SIGNALS = ['jailbreak', 'domain', 'pii']

const BINDING_FIELD_PREFIX = 'binding:'

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {}
}

function nested(value: unknown, path: readonly string[]): Record<string, unknown> {
  return path.reduce<Record<string, unknown>>(
    (current, part) => record(current[part]),
    record(value),
  )
}

function text(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

export interface DecisionBindingsState {
  decisionModel: string
  explicitDecisionModel: boolean
  decisionModelLabel: string
  deployments: string[]
  /** The deployment each authored binding names, by signal type. */
  bindings: Record<string, string>
  keys: readonly string[]
}

/**
 * The card edits what the configuration authors (config.yaml's global block);
 * the Router's effective view supplies the decision model and the pre-declared
 * deployments.
 */
export function decisionBindingsState(
  section: unknown,
  authored: ConfigData | null,
  effective: CanonicalGlobalConfig | null,
): DecisionBindingsState {
  const authoredCatalog = nested(authored?.global, ['model_catalog'])
  const effectiveCatalog = record(effective?.model_catalog)
  const system = record(section)
  const decisionModel =
    text(record(system.decision_model).deployment) ||
    text(nested(authoredCatalog, ['system', 'decision_model']).deployment) ||
    'primary'
  const bindings = Object.fromEntries(
    Object.entries(bindingsBySignal(authoredCatalog.bindings)).filter(([key]) =>
      isSignalBindingKey(key),
    ),
  )
  const declared = {
    ...record(authoredCatalog.deployments),
    ...record(effectiveCatalog.deployments),
  }
  const deployments = [...new Set([...Object.keys(declared), decisionModel])].sort()
  return {
    decisionModel,
    explicitDecisionModel: Boolean(record(system.decision_model).deployment),
    decisionModelLabel: configuredDecisionModel({ global: { model_catalog: { system } } }),
    deployments,
    bindings,
    keys: SIGNAL_BINDING_KEYS,
  }
}

function signalLabel(key: string): string {
  return SIGNAL_LABELS[key as keyof typeof SIGNAL_LABELS] ?? key
}

function currentValue(state: DecisionBindingsState, key: string): string {
  return state.bindings[key] ?? ''
}

/** What answers a signal: its binding, or the decision model. */
export function signalModelSummary(state: DecisionBindingsState, key: string): string {
  return currentValue(state, key) || `Follows ${state.decisionModelLabel}`
}

export function decisionBindingsSummary(state: DecisionBindingsState) {
  return [
    {
      label: 'Decision Model',
      value: state.explicitDecisionModel
        ? state.decisionModelLabel
        : `${state.decisionModelLabel} (default)`,
    },
    ...SUMMARY_SIGNALS.map((key) => ({
      label: signalLabel(key),
      value: signalModelSummary(state, key),
    })),
  ]
}

export function decisionBindingsBadges(state: DecisionBindingsState) {
  const explicit = Object.keys(state.bindings).length
  return [
    {
      label: `${explicit} explicit ${explicit === 1 ? 'binding' : 'bindings'}`,
      tone: explicit > 0 ? ('active' as const) : ('inactive' as const),
    },
  ]
}

export function decisionBindingsEditData(state: DecisionBindingsState): EditFormData {
  return {
    decision_model: state.decisionModel,
    ...Object.fromEntries(
      state.keys.map((key) => [`${BINDING_FIELD_PREFIX}${key}`, currentValue(state, key)]),
    ),
  }
}

export function decisionBindingsFields(state: DecisionBindingsState): FieldConfig[] {
  return [
    {
      name: 'decision_model',
      label: 'Decision Model',
      section: 'Decision model',
      type: 'select',
      required: true,
      options: state.deployments,
      description:
        'The deployment that answers built-in signals and decision questions. Question signals set to (None) follow it.',
    },
    ...state.keys.map(
      (key): FieldConfig => ({
        name: `${BINDING_FIELD_PREFIX}${key}`,
        label: signalLabel(key),
        section: 'Signal bindings',
        type: 'select',
        options: [
          ...new Set(
            ['', state.bindings[key], ...state.deployments].filter(
              (option): option is string => option !== undefined,
            ),
          ),
        ],
      }),
    ),
  ]
}

/**
 * A merge patch of the global block: it writes the decision model, and for
 * each changed signal its binding under the signal-type key.
 */
export function decisionBindingsPatch(
  state: DecisionBindingsState,
  data: EditFormData,
): Partial<ConfigData> {
  const system: Record<string, unknown> = {
    decision_model: { deployment: text(data.decision_model) || state.decisionModel },
  }
  const bindings: Record<string, unknown> = {}
  for (const key of state.keys) {
    const value = text(data[`${BINDING_FIELD_PREFIX}${key}`])
    if (value === currentValue(state, key)) continue
    bindings[key] = value || null
  }
  return {
    model_catalog: {
      system,
      ...(Object.keys(bindings).length ? { bindings } : {}),
    },
  } as Partial<ConfigData>
}
