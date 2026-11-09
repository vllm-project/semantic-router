/**
 * A binding names the deployment that answers one shared-answer signal type,
 * written as the deployment's name (`pii: local-vela`). The object form also
 * selects a head and an operating point; its other fields are deprecated: the
 * signal decides the contract and the deployment's provider the adapter.
 */
export type ModelBindingValue = string | ModelBindingObject

export interface ModelBindingObject {
  deployment: string
  head?: string
  operating_point?: Record<string, unknown>
  /** @deprecated The signal decides the contract. */
  contract?: string
  /** @deprecated The deployment's provider decides the adapter. */
  adapter?: string
  /** @deprecated */
  mapping_path?: string
  /** @deprecated */
  pair_scorer?: Record<string, unknown>
}

export type ModelBindings = Record<string, ModelBindingValue>

/**
 * The signal types a binding is keyed by. Rules that ask a question of their
 * own (decision, classifier, safety and its hazard part, complexity) name
 * their deployment on the rule instead.
 */
export const SIGNAL_BINDING_KEYS = [
  'domain',
  'jailbreak',
  'pii',
  'fact_check',
  'user_feedback',
  'modality',
  'hallucination',
  'preference',
  'reask',
] as const

// Earlier releases keyed these signals by consumer name. The Router still
// reads those names, but refuses both spellings for one signal.
const SIGNAL_OF_CONSUMER: ReadonlyMap<string, string> = new Map([
  ['domain_classifier', 'domain'],
  ['prompt_guard', 'jailbreak'],
  ['pii_classifier', 'pii'],
  ['fact_check_classifier', 'fact_check'],
  ['feedback_detector', 'user_feedback'],
  ['modality_detector', 'modality'],
  ['hallucination_detector', 'hallucination'],
])

const UNSAFE_KEYS = new Set(['__proto__', 'constructor', 'prototype'])

export function isUnsafeBindingKey(key: string): boolean {
  return UNSAFE_KEYS.has(key)
}

/** The signal-type key a binding for a task consumer is written under. */
export function bindingKey(consumer: string): string {
  return SIGNAL_OF_CONSUMER.get(consumer) ?? consumer
}

/** Whether key, in either spelling, is a signal type a binding may name. */
export function isSignalBindingKey(key: string): boolean {
  return (SIGNAL_BINDING_KEYS as readonly string[]).includes(bindingKey(key))
}

/** Every key that binds the same signal as key: its signal type and the consumer name it replaced. */
export function bindingSpellings(key: string): string[] {
  const signal = bindingKey(key)
  return [
    signal,
    ...[...SIGNAL_OF_CONSUMER].filter(([, type]) => type === signal).map(([consumer]) => consumer),
  ]
}

/** The rules of one signal collection a rule-level task consumer names. */
export interface RuleTarget {
  collection: 'classifiers' | 'safety' | 'complexity' | 'decision'
  /** Every rule of the collection when absent. */
  rule?: string
  /** The rule's hazard part rather than the rule. */
  hazard?: boolean
}

/**
 * The rules a task consumer of the Router's catalog stands for when the rule
 * names its own deployment: classifier.<rule>, safety.<rule>,
 * safety.<rule>.hazard, complexity, complexity.<rule> and decision.<rule>.
 */
export function ruleTarget(consumer: string): RuleTarget | undefined {
  if (consumer === 'complexity') return { collection: 'complexity' }
  if (consumer.startsWith('safety.') && consumer.endsWith('.hazard')) {
    const rule = consumer.slice('safety.'.length, -'.hazard'.length)
    return rule ? { collection: 'safety', rule, hazard: true } : undefined
  }
  const collections = {
    classifier: 'classifiers',
    safety: 'safety',
    complexity: 'complexity',
    decision: 'decision',
  } as const
  const separator = consumer.indexOf('.')
  const type = consumer.slice(0, separator) as keyof typeof collections
  const rule = consumer.slice(separator + 1)
  return separator > 0 && rule && Object.prototype.hasOwnProperty.call(collections, type)
    ? { collection: collections[type], rule }
    : undefined
}

/** The deployment a binding written in either form names. */
export function bindingDeployment(value: unknown): string | undefined {
  const deployment =
    value && typeof value === 'object' && !Array.isArray(value)
      ? (value as { deployment?: unknown }).deployment
      : value
  return typeof deployment === 'string' && deployment.trim() ? deployment.trim() : undefined
}

/** The deployment each binding names, keyed by signal type whichever spelling was written. */
export function bindingsBySignal(value: unknown): Record<string, string> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return {}
  const bindings: Record<string, string> = {}
  for (const [key, binding] of Object.entries(value)) {
    const deployment = bindingDeployment(binding)
    if (deployment && !isUnsafeBindingKey(key)) bindings[bindingKey(key)] = deployment
  }
  return bindings
}
