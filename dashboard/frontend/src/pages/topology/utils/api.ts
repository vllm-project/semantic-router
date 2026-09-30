// topology/utils/api.ts - API calls for topology

import { TestQueryResult, MatchedSignal, SignalType, EvaluatedRule, ConfigData, TestQueryMode } from '../types'

/**
 * Backend API response format for test-query
 */
interface TestQueryResponse {
  query: string
  mode: 'simulate' | 'dry-run'
  matchedSignals?: Array<{
    type: string
    name: string
    confidence: number | null
    confidenceAvailable?: boolean
    value?: number
    reason?: string
  }> | null
  decisionConfidence?: number | null
  decisionConfidenceAvailable?: boolean
  signalErrorMatches?: Record<string, boolean>
  matchedDecision: string | null
  matchedModels?: string[] | null
  highlightedPath?: string[] | null
  isAccurate: boolean
  evaluatedRules?: Array<{
    decisionName: string
    expression?: string
    state?: string
    ruleOperator: string
    conditions?: string[] | null
    matchedCount: number
    totalCount: number
    isMatch: boolean
    priority: number
    matchedModels?: string[]
  }>
  evalTrace?: TestQueryResult['evalTrace']
  signalErrors?: Record<string, string>
  appliedUnknownPolicies?: Record<string, string>
  decisionError?: string
  selectedModel?: string
  recommendedModels?: string[]
  selectionStatus?: string
  selectionMethod?: string
  selectionReason?: string
  routingLatency?: number
  warning?: string
  isFallbackDecision?: boolean  // True if matched decision is a system fallback
  fallbackReason?: string       // Reason for fallback
}

/** Call the live router; failed previews remain failed previews. */
export async function testQueryDryRun(
  query: string,
  model?: string,
): Promise<TestQueryResult> {
  return requestTestQuery(query, 'dry-run', model)
}

/** Explicit simulation requests never substitute for live verification. */
export async function testQuerySimulate(
  query: string,
  model?: string,
): Promise<TestQueryResult> {
  return requestTestQuery(query, 'simulate', model)
}

async function requestTestQuery(
  query: string,
  mode: TestQueryMode,
  model?: string,
): Promise<TestQueryResult> {
  const response = await fetch('/api/topology/test-query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, mode, model }),
  })
  let data: TestQueryResponse
  try {
    data = await response.json()
  } catch {
    throw new Error(`Router preview failed (HTTP ${response.status})`)
  }
  if (!data || typeof data.query !== 'string') {
    throw new Error(`Router preview failed (HTTP ${response.status})`)
  }

  return {
    query: data.query,
    mode: data.mode,
    isAccurate: response.ok && data.isAccurate === true,
    matchedSignals: convertSignals(data.matchedSignals),
    matchedDecision: data.matchedDecision ?? null,
    decisionConfidence: data.decisionConfidence,
    decisionConfidenceAvailable: data.decisionConfidenceAvailable,
    signalErrorMatches: data.signalErrorMatches,
    matchedModels: data.matchedModels ?? [],
    highlightedPath: data.highlightedPath ?? [],
    evaluatedRules: convertEvaluatedRules(data.evaluatedRules),
    evalTrace: data.evalTrace,
    signalErrors: data.signalErrors,
    appliedUnknownPolicies: data.appliedUnknownPolicies,
    decisionError: data.decisionError,
    selectedModel: data.selectedModel,
    recommendedModels: data.recommendedModels,
    selectionStatus: data.selectionStatus,
    selectionMethod: data.selectionMethod,
    selectionReason: data.selectionReason,
    routingLatency: data.routingLatency,
    warning: data.warning || (!response.ok ? `Router preview failed (HTTP ${response.status})` : undefined),
    isFallbackDecision: data.isFallbackDecision,
    fallbackReason: data.fallbackReason,
  }
}

/**
 * Convert backend signal format to frontend format
 */
function convertSignals(signals: TestQueryResponse['matchedSignals']): MatchedSignal[] {
  return (signals ?? []).map(s => ({
    type: s.type as SignalType,
    name: s.name,
    matched: true, // Backend only returns matched signals
    value: s.value,
    confidence: s.confidenceAvailable === false ? null : s.confidence,
    confidenceAvailable: s.confidenceAvailable,
    score: s.confidenceAvailable === false ? null : s.confidence,
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
      condition: r.expression ?? `${r.ruleOperator}(${conditions.join(', ')})`,
      state: r.state,
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
