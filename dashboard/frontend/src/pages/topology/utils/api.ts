// topology/utils/api.ts - API calls for topology

import { TestQueryResult, MatchedSignal, SignalType, EvaluatedRule } from '../types'

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
    reason?: string
  }>
  matchedDecision: string | null
  matchedModels: string[]
  highlightedPath: string[]
  isAccurate: boolean
  evaluatedRules?: Array<{
    decisionName: string
    ruleOperator: string
    conditions: string[]
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

  return rules.map(r => ({
    decisionName: r.decisionName,
    condition: `${r.ruleOperator}(${r.conditions.join(', ')})`,
    result: r.isMatch,
    priority: r.priority,
    matchedConditions: r.matchedCount,
    totalConditions: r.totalCount,
    matchedModels: r.matchedModels,
  }))
}

/**
 * Fetch topology configuration
 */
export async function fetchTopologyConfig() {
  const response = await fetch('/api/router/config/all')
  if (!response.ok) {
    throw new Error(`Failed to fetch config: ${response.statusText}`)
  }
  return response.json()
}
