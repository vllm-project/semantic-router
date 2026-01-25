// topology/utils/signalMatcher.ts - Test Query Signal Matcher

import {
  ParsedTopology,
  TestQueryResult,
  MatchedSignal,
  SignalType,
  RuleCombination,
  KeywordSignalConfig,
} from '../types'

/**
 * Simulate signal matching for a test query (frontend-only)
 * Note: Only keyword and language signals can be accurately simulated
 */
export async function simulateSignalMatching(
  query: string,
  topology: ParsedTopology
): Promise<TestQueryResult> {
  const matchedSignals: MatchedSignal[] = []

  // 1. Keyword matching (accurate simulation)
  topology.signals
    .filter(s => s.type === 'keyword')
    .forEach(signal => {
      const config = signal.config as KeywordSignalConfig
      const keywords = config.keywords || []
      const queryToMatch = config.case_sensitive ? query : query.toLowerCase()

      let matched = false
      if (config.operator === 'AND') {
        matched = keywords.every(kw =>
          queryToMatch.includes(config.case_sensitive ? kw : kw.toLowerCase())
        )
      } else {
        matched = keywords.some(kw =>
          queryToMatch.includes(config.case_sensitive ? kw : kw.toLowerCase())
        )
      }

      matchedSignals.push({
        type: 'keyword',
        name: signal.name,
        matched,
        score: matched ? 1.0 : 0,
        needsBackend: false,
        reason: matched ? `Keywords matched: ${keywords.join(', ')}` : undefined,
      })
    })

  // 2. Language matching (heuristic simulation)
  topology.signals
    .filter(s => s.type === 'language')
    .forEach(signal => {
      // Simple detection: Chinese characters > 30% = Chinese
      const chineseChars = (query.match(/[\u4e00-\u9fa5]/g) || []).length
      const chineseRatio = chineseChars / Math.max(query.length, 1)
      const isLikelyChinese = chineseRatio > 0.3
      const isLikelyEnglish = /^[a-zA-Z\s.,!?'"()-]+$/.test(query.trim())

      const signalNameLower = signal.name.toLowerCase()
      let matched = false
      if (signalNameLower.includes('chinese') || signalNameLower.includes('zh')) {
        matched = isLikelyChinese
      } else if (signalNameLower.includes('english') || signalNameLower.includes('en')) {
        matched = isLikelyEnglish
      }

      matchedSignals.push({
        type: 'language',
        name: signal.name,
        matched,
        score: matched ? 1.0 : 0,
        needsBackend: false,
        reason: `Language detection: ${isLikelyChinese ? 'Chinese' : isLikelyEnglish ? 'English' : 'Unknown'}`,
      })
    })

  // 3. Other signal types - mark as "needs backend verification"
  const backendOnlyTypes: SignalType[] = ['embedding', 'domain', 'fact_check', 'user_feedback', 'preference', 'latency']
  backendOnlyTypes.forEach(type => {
    topology.signals
      .filter(s => s.type === type)
      .forEach(signal => {
        matchedSignals.push({
          type,
          name: signal.name,
          matched: false, // Cannot accurately simulate
          score: undefined,
          needsBackend: true,
          reason: `Requires backend verification (${type})`,
        })
      })
  })

  // 4. Evaluate decisions based on matched signals
  const sortedDecisions = [...topology.decisions].sort((a, b) => b.priority - a.priority)
  let matchedDecision: string | null = null
  let matchedModels: string[] = []
  const highlightedPath: string[] = ['client']

  // Add global plugins to path
  topology.globalPlugins.forEach(plugin => {
    if (plugin.enabled) {
      highlightedPath.push(`global-plugin-${plugin.type}`)
    }
  })

  for (const decision of sortedDecisions) {
    const ruleMatched = evaluateRules(decision.rules, matchedSignals)
    if (ruleMatched) {
      matchedDecision = decision.name
      matchedModels = decision.modelRefs.map(r => r.model)

      // Build highlighted path
      decision.rules.conditions.forEach(cond => {
        highlightedPath.push(`signal-group-${cond.type}`)
      })
      highlightedPath.push(`decision-${decision.name}`)
      if (decision.algorithm) {
        highlightedPath.push(`algorithm-${decision.name}`)
      }
      if (decision.plugins && decision.plugins.length > 0) {
        highlightedPath.push(`plugin-chain-${decision.name}`)
      }
      decision.modelRefs.forEach(ref => {
        highlightedPath.push(`model-${ref.model.replace(/[^a-zA-Z0-9]/g, '-')}`)
      })

      break
    }
  }

  return {
    query,
    mode: 'simulate',
    matchedSignals,
    matchedDecision,
    matchedModels,
    highlightedPath,
    isAccurate: false, // Frontend simulation is not 100% accurate
  }
}

/**
 * Evaluate rules against matched signals
 */
function evaluateRules(rules: RuleCombination, matchedSignals: MatchedSignal[]): boolean {
  if (rules.conditions.length === 0) {
    return true // No conditions = always match (default decision)
  }

  const conditionResults = rules.conditions.map(cond => {
    const signal = matchedSignals.find(s => s.type === cond.type && s.name === cond.name)
    // If signal needs backend, we can't evaluate accurately - assume false for safety
    if (signal?.needsBackend) {
      return false
    }
    return signal?.matched ?? false
  })

  if (rules.operator === 'AND') {
    return conditionResults.every(r => r)
  } else {
    return conditionResults.some(r => r)
  }
}

/**
 * Get signal icon by type
 */
export function getSignalIcon(type: SignalType): string {
  const icons: Record<SignalType, string> = {
    keyword: '🔑',
    embedding: '📐',
    domain: '🎯',
    fact_check: '✓',
    user_feedback: '💬',
    preference: '⚙️',
    language: '🌐',
    latency: '⚡',
    context: '📏',
  }
  return icons[type] || '❓'
}

/**
 * Get signal color by type
 */
export function getSignalColor(type: SignalType): string {
  const colors: Record<SignalType, string> = {
    keyword: '#4CAF50',
    embedding: '#2196F3',
    domain: '#9C27B0',
    fact_check: '#FF9800',
    user_feedback: '#E91E63',
    preference: '#00BCD4',
    language: '#795548',
    latency: '#FFC107',
    context: '#607D8B', // Blue Grey
  }
  return colors[type] || '#607D8B'
}
