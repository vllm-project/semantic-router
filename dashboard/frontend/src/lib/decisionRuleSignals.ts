/**
 * Signal choices for one routing scope (the root routing profile or a named
 * recipe), shown in the rule editor's picker. The signal types and their
 * config collections come from the generated router contract through
 * SIGNAL_CATALOG; there is no hand-written type list here. Whether a
 * reference is valid is checked by the Router when the config is saved.
 */

import { DECISION_SIGNAL_TYPES, ROUTER_CONFIG_EXTENSION } from '@/generated/routerConfigContract'

import type { ConfigProjections, ConfigSignals } from '../pages/configPageSupport'
import { SIGNAL_CATALOG } from '../pages/configPageSignalCatalog'
import type { SignalDescriptor } from './decisionRuleAst'

export interface DecisionRuleSignalSource {
  signals?: ConfigSignals
  projections?: ConfigProjections
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

// Signal types whose conditions name a label (the contract's
// reference_qualifier), and the one type that sets its own on_error.
const LABELLED_SIGNAL_TYPES = new Set<string>(
  ROUTER_CONFIG_EXTENSION.signals.flatMap((surface) =>
    'reference_qualifier' in surface && surface.reference_qualifier === 'label' ? [surface.type] : [],
  ),
)

export function isLabelledSignalType(signalType: string): boolean {
  return LABELLED_SIGNAL_TYPES.has(signalType.trim().toLowerCase())
}

export function supportsOnError(signalType: string): boolean {
  return signalType.trim().toLowerCase() === 'systemone'
}

const COMPLEXITY_LEVELS = ['easy', 'medium', 'hard']

function stringLabels(labels: unknown): string[] {
  if (!Array.isArray(labels)) return []
  return labels.filter((label): label is string => typeof label === 'string' && label !== '')
}

// A System One question offers its option keys as label suggestions; score
// and yes/no questions are matched without a label.
function systemOneLabels(item: Record<string, unknown>): string[] {
  const question = item.question
  if (isRecord(question) && question.type !== 'score' && question.type !== 'noul') {
    const options = question.type === 'set' || question.type === 'span' ? question.labels : question.choices
    const keys = choiceKeys(options)
    if (keys.length > 0) return keys
    const levels = stringLabels(question.levels)
    if (levels.length > 0) return levels
  }
  return stringLabels(item.labels)
}

// A role binding is referenced by its name or its role.
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
      const descriptor: SignalDescriptor = { signalType: entry.type, name }
      if (isLabelledSignalType(entry.type)) {
        const labels = entry.type === 'systemone' ? systemOneLabels(item) : stringLabels(item.labels)
        // A complexity rule without labels still answers easy, medium or hard.
        if (entry.type === 'complexity' && labels.length === 0) labels.push(...COMPLEXITY_LEVELS)
        if (labels.length > 0) descriptor.labels = labels
      }
      descriptors.push(descriptor)
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
