import { ROUTER_CONFIG_EXTENSION } from '../generated/routerConfigContract'
import { isEmptyNode, isRecord } from './configPageDecisionRuleSupport'
import type { DecisionConfig, NormalizedModel } from './configPageSupport'

export type DecisionKind = 'routing' | 'guardrail' | 'fallback'

export const DECISION_KIND_LABEL: Record<DecisionKind, string> = {
  routing: 'Routing',
  guardrail: 'Guardrail',
  fallback: 'Fallback',
}

export const DECISION_KIND_HINT: Record<DecisionKind, string> = {
  routing: 'Matches its conditions and sends the request to its models.',
  guardrail: 'Answers with a fixed response or blocks the request, without calling a model.',
  fallback: 'Has no conditions, so it matches every request that reaches it.',
}

/** How a decision answers: a fixed response, a catch-all, or ordinary routing. */
export function decisionKind(decision: DecisionConfig): DecisionKind {
  if ((decision.plugins ?? []).some((plugin) => plugin.type === 'fast_response')) return 'guardrail'
  if (!isRecord(decision.rules) || isEmptyNode(decision.rules)) return 'fallback'
  return 'routing'
}

export interface AlgorithmMeta {
  type: string
  label: string
  description: string
  kind: 'selector' | 'executor'
  tier: string
}

export const ALGORITHMS: AlgorithmMeta[] = ROUTER_CONFIG_EXTENSION.algorithms.map((surface) => ({
  type: surface.type,
  label: surface.display_name,
  description: surface.description,
  kind: surface.kind as AlgorithmMeta['kind'],
  tier: surface.tier,
}))

export function algorithmMeta(type: unknown): AlgorithmMeta | undefined {
  return ALGORITHMS.find((algorithm) => algorithm.type === type)
}

export const PLUGINS = ROUTER_CONFIG_EXTENSION.plugins.map((surface) => ({
  type: surface.type,
  label: surface.display_name,
  description: surface.description,
}))

export function pluginLabel(type: string): string {
  return PLUGINS.find((plugin) => plugin.type === type)?.label ?? type
}

/** The logo key a model card names (package:qwen), else its first backend's provider. */
export function modelLogo(model: NormalizedModel | undefined): string {
  const logo = model?.card_override?.presentation?.logo
  if (logo?.startsWith('package:')) return logo.slice('package:'.length)
  return model?.backend_refs?.[0]?.provider || 'vllm'
}

export function modelDisplayName(model: NormalizedModel | undefined, name: string): string {
  return model?.card_override?.display_name || name
}
