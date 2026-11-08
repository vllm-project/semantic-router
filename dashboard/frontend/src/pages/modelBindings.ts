/**
 * A binding names the deployment that answers one shared-answer signal type,
 * written as the deployment's name (`pii: local-vela`) or as
 * `{deployment: local-vela}`; the signal decides the question and the
 * deployment's provider how it is asked.
 */
export type ModelBindingValue = string | ModelBindingObject

export interface ModelBindingObject {
  deployment: string
}

export type ModelBindings = Record<string, ModelBindingValue>

/**
 * The signal types a binding is keyed by: the Router's shared-answer signal
 * types. Rules that ask a question of their own (systemone, safety,
 * complexity) may also name their deployment on the rule.
 */
export const SIGNAL_BINDING_KEYS = [
  'domain',
  'fact_check',
  'user_feedback',
  'jailbreak',
  'safety',
  'pii',
  'hallucination',
  'refusal',
  'turn_relation',
  'complexity',
  'modality',
] as const

const UNSAFE_KEYS = new Set(['__proto__', 'constructor', 'prototype'])

export function isUnsafeBindingKey(key: string): boolean {
  return UNSAFE_KEYS.has(key)
}

/** Whether key is a signal type a binding may name. */
export function isSignalBindingKey(key: string): boolean {
  return (SIGNAL_BINDING_KEYS as readonly string[]).includes(key)
}

/** The deployment a binding written in either form names. */
export function bindingDeployment(value: unknown): string | undefined {
  const deployment =
    value && typeof value === 'object' && !Array.isArray(value)
      ? (value as { deployment?: unknown }).deployment
      : value
  return typeof deployment === 'string' && deployment.trim() ? deployment.trim() : undefined
}

/** The deployment each binding names, keyed by signal type. */
export function bindingsBySignal(value: unknown): Record<string, string> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return {}
  const bindings: Record<string, string> = {}
  for (const [key, binding] of Object.entries(value)) {
    const deployment = bindingDeployment(binding)
    if (deployment && !isUnsafeBindingKey(key)) bindings[key] = deployment
  }
  return bindings
}
