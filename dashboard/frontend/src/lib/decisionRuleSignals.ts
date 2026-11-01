/**
 * Signal choices for one routing scope (the root routing profile or a named
 * recipe), shown in the rule editor's picker. The signal types and their
 * config collections come from the generated router contract through
 * SIGNAL_CATALOG; there is no hand-written type list here. Whether a
 * reference is valid is checked by the Router when the config is saved.
 */

import { DECISION_SIGNAL_TYPES } from '@/generated/routerConfigContract'

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

// Score and yes/no questions are matched without a label, so only choice,
// set and span questions offer option keys as label suggestions.
function decisionQuestionDescriptor(name: string, question: unknown): SignalDescriptor {
  const descriptor: SignalDescriptor = { signalType: 'decision', name }
  if (!isRecord(question) || question.type === 'score' || question.type === 'noul') return descriptor
  const options = question.type === 'set' || question.type === 'span' ? question.labels : question.choices
  descriptor.labels = choiceKeys(options)
  return descriptor
}

function classifierDescriptor(item: Record<string, unknown>): SignalDescriptor {
  const labels = Array.isArray(item.labels) ? item.labels.filter((label): label is string => typeof label === 'string') : []
  return { signalType: 'classifier', name: item.name as string, labels }
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
      if (entry.type === 'classifier') {
        descriptors.push(classifierDescriptor(item))
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
