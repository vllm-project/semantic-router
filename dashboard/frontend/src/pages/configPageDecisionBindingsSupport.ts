import { createElement } from 'react'
import type { EditFormData, FieldConfig } from '../components/EditModal'
import type { CanonicalGlobalConfig, ConfigData } from './configPageSupport'
import { configuredDecisionModel } from './decisionModelSupport'
import {
  SIGNAL_BINDING_KEYS,
  bindingSpellings,
  bindingsBySignal,
  isSignalBindingKey,
} from './modelBindings'

const SIGNAL_LABELS: Record<(typeof SIGNAL_BINDING_KEYS)[number], string> = {
  domain: 'Domain',
  jailbreak: 'Jailbreak',
  pii: 'PII',
  fact_check: 'Fact check',
  user_feedback: 'User feedback',
  modality: 'Modality',
  hallucination: 'Hallucination',
  preference: 'Preference',
  reask: 'Reask',
}

const SUMMARY_SIGNALS = ['jailbreak', 'domain', 'pii']

// Before bindings, a signal's model came from its system line or from its
// module's model fields. The Router still reads them, so a new binding or a
// return to the decision model must clear them.
interface LegacyModelSource {
  line?: string
  module: string[]
  fields: string[]
}
const MODEL_FIELDS = ['model_id', 'model_ref']
const REMOTE_MODEL_FIELDS = ['model_id', 'model_ref', 'backend']
const LEGACY_MODEL_SOURCES: Record<string, LegacyModelSource> = {
  domain: {
    line: 'domain_classifier',
    module: ['classifier', 'domain'],
    fields: REMOTE_MODEL_FIELDS,
  },
  jailbreak: { line: 'prompt_guard', module: ['prompt_guard'], fields: REMOTE_MODEL_FIELDS },
  pii: { line: 'pii_classifier', module: ['classifier', 'pii'], fields: REMOTE_MODEL_FIELDS },
  fact_check: {
    line: 'fact_check_classifier',
    module: ['hallucination_mitigation', 'fact_check'],
    fields: MODEL_FIELDS,
  },
  hallucination: {
    line: 'hallucination_detector',
    module: ['hallucination_mitigation', 'detector'],
    fields: MODEL_FIELDS,
  },
  user_feedback: { line: 'feedback_detector', module: ['feedback_detector'], fields: MODEL_FIELDS },
  modality: { module: ['modality_detector', 'classifier'], fields: ['model_path'] },
}

// A deprecated backend is a remote classifier named by its external model.
function legacyModelName(value: unknown): string {
  if (typeof value === 'string') return value.trim()
  if (value && typeof value === 'object') return text(record(value).model) || 'remote classifier'
  return ''
}

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
  /** The deprecated model field each unbound signal still reads, by signal type. */
  legacy: Record<string, string>
  /** Authored binding keys no signal type has: rule keys and keys of rule-level types. */
  deprecated: Record<string, string>
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
  const authoredBindings = Object.entries(bindingsBySignal(authoredCatalog.bindings))
  const bindings = Object.fromEntries(authoredBindings.filter(([key]) => isSignalBindingKey(key)))
  const deprecated = Object.fromEntries(
    authoredBindings
      .filter(([key]) => !isSignalBindingKey(key))
      .sort(([a], [b]) => a.localeCompare(b)),
  )
  const authoredSystem = record(authoredCatalog.system)
  const legacy: Record<string, string> = {}
  for (const [key, source] of Object.entries(LEGACY_MODEL_SOURCES)) {
    if (bindings[key]) continue
    const linePath = source.line ? text(authoredSystem[source.line]) : ''
    const moduleConfig = nested(authoredCatalog, ['modules', ...source.module])
    const field = source.fields.find((name) => legacyModelName(moduleConfig[name]))
    if (field) {
      legacy[key] =
        `${legacyModelName(moduleConfig[field])} (deprecated modules.${source.module.join('.')}.${field})`
    } else if (linePath) {
      legacy[key] = `${linePath} (deprecated system.${source.line})`
    }
  }
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
    legacy,
    deprecated,
    keys: SIGNAL_BINDING_KEYS,
  }
}

function signalLabel(key: string): string {
  return SIGNAL_LABELS[key as keyof typeof SIGNAL_LABELS] ?? key
}

function currentValue(state: DecisionBindingsState, key: string): string {
  return state.bindings[key] ?? state.legacy[key] ?? ''
}

/** What answers a signal: its binding, a deprecated model path, or the decision model. */
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
  const legacy = Object.keys(state.legacy).length
  const deprecated = Object.keys(state.deprecated).length
  return [
    {
      label: `${explicit} explicit ${explicit === 1 ? 'binding' : 'bindings'}`,
      tone: explicit > 0 ? ('active' as const) : ('inactive' as const),
    },
    ...(legacy > 0
      ? [
          {
            label: `${legacy} deprecated model ${legacy === 1 ? 'path' : 'paths'}`,
            tone: 'info' as const,
          },
        ]
      : []),
    ...(deprecated > 0
      ? [
          {
            label: `${deprecated} deprecated ${deprecated === 1 ? 'binding' : 'bindings'}`,
            tone: 'info' as const,
          },
        ]
      : []),
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
            ['', state.legacy[key], state.bindings[key], ...state.deployments].filter(
              (option): option is string => option !== undefined,
            ),
          ),
        ],
        description: state.legacy[key]
          ? 'This deprecated model path applies until you choose a deployment or (None).'
          : undefined,
      }),
    ),
    ...(Object.keys(state.deprecated).length
      ? [
          {
            name: 'deprecated_bindings',
            label: 'Deprecated bindings',
            section: 'Signal bindings',
            type: 'custom' as const,
            description:
              "Rules name their own deployment instead. These are kept as written until you set each rule's deployment and remove them in Raw YAML.",
            customRender: () =>
              createElement(
                'ul',
                null,
                Object.entries(state.deprecated).map(([key, deployment]) =>
                  createElement(
                    'li',
                    { key },
                    createElement('code', null, `${key}: ${deployment}`),
                  ),
                ),
              ),
          },
        ]
      : []),
  ]
}

/**
 * A merge patch of the global block: it writes the decision model, and for
 * each changed signal its binding under the signal-type key, deleting the old
 * consumer-name key and the deprecated model fields the binding replaces.
 */
export function decisionBindingsPatch(
  state: DecisionBindingsState,
  data: EditFormData,
): Partial<ConfigData> {
  const system: Record<string, unknown> = {
    decision_model: { deployment: text(data.decision_model) || state.decisionModel },
  }
  const bindings: Record<string, unknown> = {}
  const modules: Record<string, unknown> = {}
  for (const key of state.keys) {
    const value = text(data[`${BINDING_FIELD_PREFIX}${key}`])
    if (value === currentValue(state, key)) continue
    for (const spelling of bindingSpellings(key)) bindings[spelling] = null
    if (value) bindings[key] = value
    const source = LEGACY_MODEL_SOURCES[key]
    if (!source) continue
    if (source.line) system[source.line] = null
    const target = source.module.reduce<Record<string, unknown>>((parent, part) => {
      parent[part] = record(parent[part])
      return parent[part] as Record<string, unknown>
    }, modules)
    for (const field of source.fields) target[field] = null
  }
  return {
    model_catalog: {
      system,
      ...(Object.keys(bindings).length ? { bindings } : {}),
      ...(Object.keys(modules).length ? { modules } : {}),
    },
  } as Partial<ConfigData>
}
