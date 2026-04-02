// topology/utils/api.ts - API calls for topology

import { TestQueryResult, MatchedSignal, SignalType, EvaluatedRule, ConfigData } from '../types'

/**
 * Backend API response format for test-query
 */
interface TestQueryResponse {
  query: string
  mode: 'simulate' | 'dry-run'
  matchedSignals: Array<{
    type: string
    name: string
    confidence: number
    value?: number
    reason?: string
  }>
  matchedDecision: string | null
  matchedModels: string[]
  highlightedPath: string[]
  isAccurate: boolean
  evaluatedRules?: Array<{
    decisionName: string
    ruleOperator: string
    conditions?: string[] | null
    matchedCount: number
    totalCount: number
    isMatch: boolean
    priority: number
    matchedModels?: string[]
  }>
  routingLatency?: number
  warning?: string
  isFallbackDecision?: boolean  // True if matched decision is a system fallback
  fallbackReason?: string       // Reason for fallback
}

/**
 * Call backend Dry-Run API for accurate routing verification
 */
export async function testQueryDryRun(query: string): Promise<TestQueryResult> {
  const response = await fetch('/api/topology/test-query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query,
      mode: 'dry-run',
    }),
  })

  if (!response.ok) {
    throw new Error(`Dry-run API failed: ${response.statusText}`)
  }

  const data: TestQueryResponse = await response.json()

  // Convert to frontend TestQueryResult format
  return {
    query: data.query,
    mode: data.mode,
    isAccurate: data.isAccurate,
    matchedSignals: convertSignals(data.matchedSignals),
    matchedDecision: data.matchedDecision,
    matchedModels: data.matchedModels,
    highlightedPath: data.highlightedPath,
    evaluatedRules: convertEvaluatedRules(data.evaluatedRules),
    routingLatency: data.routingLatency,
    warning: data.warning,
    isFallbackDecision: data.isFallbackDecision,
    fallbackReason: data.fallbackReason,
  }
}

/**
 * Call backend Simulate API for simulated routing (also uses backend now)
 */
export async function testQuerySimulate(query: string): Promise<TestQueryResult> {
  const response = await fetch('/api/topology/test-query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query,
      mode: 'simulate',
    }),
  })

  if (!response.ok) {
    throw new Error(`Simulate API failed: ${response.statusText}`)
  }

  const data: TestQueryResponse = await response.json()

  return {
    query: data.query,
    mode: data.mode,
    isAccurate: data.isAccurate,
    matchedSignals: convertSignals(data.matchedSignals),
    matchedDecision: data.matchedDecision,
    matchedModels: data.matchedModels,
    highlightedPath: data.highlightedPath,
    evaluatedRules: convertEvaluatedRules(data.evaluatedRules),
    routingLatency: data.routingLatency,
    warning: data.warning,
    isFallbackDecision: data.isFallbackDecision,
    fallbackReason: data.fallbackReason,
  }
}

/**
 * Convert backend signal format to frontend format
 */
function convertSignals(signals: TestQueryResponse['matchedSignals']): MatchedSignal[] {
  return signals.map(s => ({
    type: s.type as SignalType,
    name: s.name,
    matched: true, // Backend only returns matched signals
    value: s.value,
    confidence: s.confidence,
    score: s.confidence,
    reason: s.reason,
    needsBackend: false,
  }))
}

/**
 * Convert backend evaluated rules to frontend format
 */
function convertEvaluatedRules(rules?: TestQueryResponse['evaluatedRules']): EvaluatedRule[] | undefined {
  if (!rules) return undefined

  return rules.map(r => {
    const conditions = r.conditions ?? []
    return {
      decisionName: r.decisionName,
      condition: `${r.ruleOperator}(${conditions.join(', ')})`,
      result: r.isMatch,
      priority: r.priority,
      matchedConditions: r.matchedCount,
      totalConditions: r.totalCount,
      matchedModels: r.matchedModels,
    }
  })
}

/**
 * Fetch topology configuration
 */
export async function fetchTopologyConfig() {
  const [configResponse, globalResponse] = await Promise.all([
    fetch('/api/router/config/all'),
    fetch('/api/router/config/global'),
  ])

  if (!configResponse.ok) {
    throw new Error(`Failed to fetch config: ${configResponse.statusText}`)
  }

  const config = await configResponse.json() as ConfigData

  if (!globalResponse.ok) {
    return config
  }

  const effectiveGlobal = await globalResponse.json() as ConfigData['global']
  return {
    ...config,
    global: effectiveGlobal,
  }
}
