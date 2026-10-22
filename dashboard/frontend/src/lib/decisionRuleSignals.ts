/**
 * Signal descriptors for one routing scope (the root routing profile or a
 * named recipe). The signal types and their config collections come from the
 * generated router contract through SIGNAL_CATALOG; there is no hand-written
 * type list here.
 */

import { DECISION_SIGNAL_TYPES } from '@/generated/routerConfigContract'

import type { ConfigProjections, ConfigSignals } from '../pages/configPageSupport'
import { SIGNAL_CATALOG } from '../pages/configPageSignalCatalog'
import type { SignalDescriptor } from './decisionRuleAst'
import { ruleSignalScope, type RuleSignalScope } from './decisionRuleValidation'

// Matches config.RemoteClassifierContractLabelScores.
const LABEL_SCORES_CONTRACT = 'label_scores.v1'

export interface DecisionRuleSignalSource {
  signals?: ConfigSignals
  projections?: ConfigProjections
  model_bindings?: Record<string, unknown>
  /**
   * global.model_catalog.bindings. Like Go's EffectiveModelBindings, a
   * recipe-local binding wins over the global one of the same name.
   */
  global_model_bindings?: Record<string, unknown>
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function namedItems(collection: unknown): Record<string, unknown>[] {
  if (!Array.isArray(collection)) return []
  return collection.filter(
    (item): item is Record<string, unknown> => isRecord(item) && typeof item.name === 'string' && item.name !== '',
  )
}

function choiceKeys(choices: unknown): string[] {
  if (!Array.isArray(choices)) return []
  return choices
    .map((choice) => (isRecord(choice) ? choice.key : undefined))
    .filter((key): key is string => typeof key === 'string')
}

/** Mirrors config.DecisionQuestion.OptionKeys and Labelled. */
function decisionQuestionDescriptor(name: string, question: unknown): SignalDescriptor {
  const descriptor: SignalDescriptor = { signalType: 'decision', name }
  if (!isRecord(question)) return descriptor
  switch (question.type) {
    case 'score':
      descriptor.labels = Array.isArray(question.levels) ? question.levels.map((_, index) => String(index)) : []
      descriptor.labelForbidden = true
      break
    case 'noul':
      descriptor.labels = ['false', 'true']
      descriptor.labelForbidden = true
      break
    case 'set':
    case 'span':
      descriptor.labels = choiceKeys(question.labels)
      descriptor.labelRequired = true
      break
    default:
      descriptor.labels = choiceKeys(question.choices)
      descriptor.labelRequired = true
  }
  return descriptor
}

function classifierDescriptor(item: Record<string, unknown>, source: DecisionRuleSignalSource): SignalDescriptor {
  const name = item.name as string
  const labels = Array.isArray(item.labels) ? item.labels.filter((label): label is string => typeof label === 'string') : []
  const descriptor: SignalDescriptor = {
    signalType: 'classifier',
    name,
    labels,
    labelRequired: true,
    localClassifier: item.type === 'local',
  }
  // A prepared label_scores.v1 operating point supplies the default label
  // threshold, so Go does not require an explicit predicate. A global
  // classifier.* binding applies because this scope declares the classifier.
  const key = `classifier.${name}`
  const binding = source.model_bindings?.[key] ?? source.global_model_bindings?.[key]
  if (isRecord(binding) && binding.operating_point != null && binding.contract === LABEL_SCORES_CONTRACT) {
    descriptor.predicateRequired = false
  }
  return descriptor
}

/** Mirrors config.collectRoleBindingNames: a binding is referenced by name or role. */
function roleBindingDescriptors(collection: unknown): SignalDescriptor[] {
  if (!Array.isArray(collection)) return []
  const names = new Set<string>()
  for (const item of collection) {
    if (!isRecord(item)) continue
    for (const key of ['name', 'role']) {
      const value = item[key]
      if (typeof value === 'string' && value !== '') names.add(value)
    }
  }
  return Array.from(names, (name) => ({ signalType: 'authz', name }))
}

export function decisionRuleSignalDescriptors(source: DecisionRuleSignalSource): SignalDescriptor[] {
  const signals = (source.signals ?? {}) as Record<string, unknown>
  const decisionTypes = new Set<string>(DECISION_SIGNAL_TYPES)
  const descriptors: SignalDescriptor[] = []
  for (const entry of SIGNAL_CATALOG) {
    if (!decisionTypes.has(entry.type)) continue
    if (entry.type === 'authz') {
      descriptors.push(...roleBindingDescriptors(signals[entry.collection]))
      continue
    }
    for (const item of namedItems(signals[entry.collection])) {
      const name = item.name as string
      if (entry.type === 'classifier') {
        descriptors.push(classifierDescriptor(item, source))
      } else if (entry.type === 'decision') {
        descriptors.push(decisionQuestionDescriptor(name, item.question))
      } else if (entry.referenceSuffixes.length > 0) {
        for (const suffix of entry.referenceSuffixes) {
          descriptors.push({ signalType: entry.type, name: `${name}:${suffix}` })
        }
      } else {
        descriptors.push({ signalType: entry.type, name })
      }
    }
  }
  if (decisionTypes.has('projection')) {
    for (const mapping of source.projections?.mappings ?? []) {
      for (const output of mapping.outputs ?? []) {
        if (output?.name) descriptors.push({ signalType: 'projection', name: output.name })
      }
    }
  }
  return descriptors
}

/**
 * Builds the signal scope a decision rule is validated against. Pass the
 * selected scope's projected config (scopedConfig), never the root config: a
 * named recipe only sees its own declarations.
 */
export function decisionRuleSignalScope(
  scoped: DecisionRuleSignalSource,
  opts?: { strictNames?: boolean },
): RuleSignalScope {
  return ruleSignalScope(decisionRuleSignalDescriptors(scoped), opts?.strictNames ?? false)
}
