import type { DecisionRule, RouterConfig } from './dashboardPageTypes'

export function countSignals(cfg: RouterConfig): { total: number; byType: Record<string, number> } {
  const byType: Record<string, number> = {}
  let total = 0
  const signals = cfg.routing?.signals ?? cfg.signals
  if (signals) {
    for (const [type, arr] of Object.entries(signals)) {
      if (Array.isArray(arr)) {
        byType[type] = arr.length
        total += arr.length
      }
    }
  }
  return { total, byType }
}

export function countDecisions(cfg: RouterConfig): number {
  const decisions = cfg.routing?.decisions ?? cfg.decisions
  return Array.isArray(decisions) ? decisions.length : 0
}

export function countModels(cfg: RouterConfig): number {
  const models = cfg.providers?.models
  if (Array.isArray(models)) {
    return models.length
  }

  const legacyRootEndpoints = cfg.vllm_endpoints
  if (Array.isArray(legacyRootEndpoints)) return legacyRootEndpoints.length

  const legacyProviderEndpoints = cfg.providers?.vllm_endpoints
  return Array.isArray(legacyProviderEndpoints) ? legacyProviderEndpoints.length : 0
}

export function countPlugins(cfg: RouterConfig): number {
  const decisions = cfg.routing?.decisions ?? cfg.decisions
  if (Array.isArray(decisions)) {
    return decisions.reduce((count, decision) => count + (Array.isArray(decision.plugins) ? decision.plugins.length : 0), 0)
  }
  if (!cfg.plugins || typeof cfg.plugins !== 'object') return 0
  return Object.keys(cfg.plugins).length
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
  embeddings: '#9CDCFE',
  domains: '#DCDCAA',
  fact_check: '#CE9178',
  user_feedbacks: '#C586C0',
  reasks: '#FFB454',
  preferences: '#4FC1FF',
  language: '#B5CEA8',
  context: '#D7BA7D',
  complexity: '#569CD6',
  modality: '#D4D4D4',
  authz: '#F48771',
  jailbreak: '#F48771',
  pii: '#FF6B6B',
}

export function categorizeDecisions(config: RouterConfig | null): {
  guardrails: DecisionRule[]
  routing: DecisionRule[]
  fallbacks: DecisionRule[]
} {
  const decisions = config?.routing?.decisions ?? config?.decisions
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
