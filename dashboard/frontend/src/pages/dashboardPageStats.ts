import type { DecisionRule, RouterConfig } from './dashboardPageTypes'
import {
  collectScopedDecisions,
  countSignalsAcrossScopes,
  type RoutingScopedConfigLike,
} from '../utils/routingScopes'

const scopedConfig = (config: RouterConfig): RoutingScopedConfigLike =>
  config as RouterConfig & RoutingScopedConfigLike

export function countSignals(cfg: RouterConfig): { total: number; byType: Record<string, number> } {
  return countSignalsAcrossScopes(scopedConfig(cfg))
}

export function countDecisions(cfg: RouterConfig): number {
  return getAllDecisions(cfg).length
}

export function countModels(cfg: RouterConfig): number {
  const models = cfg.providers?.models
  return Array.isArray(models) ? models.length : 0
}

export function countPlugins(cfg: RouterConfig): number {
  return getAllDecisions(cfg).reduce(
    (count, decision) => count + (Array.isArray(decision.plugins) ? decision.plugins.length : 0),
    0,
  )
}

export function getAllDecisions(cfg: RouterConfig): DecisionRule[] {
  return collectScopedDecisions<DecisionRule>(scopedConfig(cfg)).map(({ scope, value }) => ({
    ...value,
    routingScope: scope.id,
    routingEntrypoints: scope.entrypointModelNames,
  }))
}

/** Classify decision by priority range */
export function getDecisionCategory(priority?: number): 'guardrail' | 'routing' | 'fallback' {
  if (priority == null) return 'routing'
  if (priority >= 999) return 'guardrail'
  if (priority <= 100) return 'fallback'
  return 'routing'
}

/** Palette for signal type labels (flow diagram + breakdown). */
export const SIGNAL_COLORS: Record<string, string> = {
  keywords: '#4EC9B0',
  domains: '#DCDCAA',
  fact_check: '#CE9178',
  user_feedbacks: '#C586C0',
  language: '#B5CEA8',
  context: '#D7BA7D',
  complexity: '#569CD6',
  modality: '#D4D4D4',
  role_bindings: '#F48771',
  jailbreak: '#F48771',
  safety: '#F44747',
  hallucination: '#D16969',
  refusal: '#CCA700',
  turn_relation: '#9CDCFE',
  pii: '#FF6B6B',
  conversation: '#6A9955',
  metadata: '#808080',
  input_modality: '#4FC1FF',
  systemone: '#B180D7',
}

export function categorizeDecisions(config: RouterConfig | null): {
  guardrails: DecisionRule[]
  routing: DecisionRule[]
  fallbacks: DecisionRule[]
} {
  const decisions = config ? getAllDecisions(config) : []
  if (!decisions) return { guardrails: [], routing: [], fallbacks: [] }
  const guardrails: DecisionRule[] = []
  const routing: DecisionRule[] = []
  const fallbacks: DecisionRule[] = []
  for (const d of decisions) {
    const cat = getDecisionCategory(d.priority)
    if (cat === 'guardrail') guardrails.push(d)
    else if (cat === 'fallback') fallbacks.push(d)
    else routing.push(d)
  }
  return { guardrails, routing, fallbacks }
}
