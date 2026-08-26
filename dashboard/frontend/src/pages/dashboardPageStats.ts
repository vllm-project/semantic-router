import type { DecisionRule } from './dashboardPageTypes'
import {
  listManagedRecipeScopes,
  type ManagedRoutingScope,
  type ManagedRoutingSnapshot,
  type ManagedRoutingSummary,
} from '../utils/managedRoutingSnapshot'
import { countSignalsInProfile } from '../utils/routingScopes'

const recipeScopes = (config: ManagedRoutingSummary): ManagedRoutingScope[] =>
  listManagedRecipeScopes(config)

const decisionScopes = (config: ManagedRoutingSummary): ManagedRoutingScope[] => {
  const topologyScopes = (config as Partial<ManagedRoutingSnapshot>).routingScopes
  if (!Array.isArray(topologyScopes)) return recipeScopes(config)

  const representativeByRecipe = new Map<string, ManagedRoutingScope>()
  for (const scope of topologyScopes) {
    if (!scope.hydrated || scope.source !== 'entrypoint' || !scope.recipeId) continue
    if (!representativeByRecipe.has(scope.recipeId)) {
      representativeByRecipe.set(scope.recipeId, scope)
    }
  }
  return representativeByRecipe.size > 0
    ? [...representativeByRecipe.values()]
    : recipeScopes(config)
}

export function countSignals(cfg: ManagedRoutingSummary): {
  total: number
  byType: Record<string, number>
} {
  const byType: Record<string, number> = {}
  let total = 0
  for (const scope of recipeScopes(cfg)) {
    const counts = countSignalsInProfile(scope.document)
    total += counts.total
    for (const [type, count] of Object.entries(counts.byType)) {
      byType[type] = (byType[type] ?? 0) + count
    }
  }
  return { total, byType }
}

export function countDecisions(cfg: ManagedRoutingSummary): number {
  return getAllDecisions(cfg).length
}

export function countModels(cfg: ManagedRoutingSummary): number {
  return cfg.models.length
}

export function countPlugins(cfg: ManagedRoutingSummary): number {
  const decisions = getAllDecisions(cfg)
  if (decisions.length > 0) {
    return decisions.reduce(
      (count, decision) => count + (Array.isArray(decision.plugins) ? decision.plugins.length : 0),
      0,
    )
  }
  return 0
}

export function getAllDecisions(cfg: ManagedRoutingSummary): DecisionRule[] {
  return decisionScopes(cfg).flatMap((scope) =>
    (scope.document.decisions ?? []).map((decision) => ({
      ...(decision as DecisionRule),
      routingScope: scope.id,
      routingEntrypoints: scope.entrypointModelNames,
    })),
  )
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

export function categorizeDecisions(config: ManagedRoutingSummary | null): {
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
