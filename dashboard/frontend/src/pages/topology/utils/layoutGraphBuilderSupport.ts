import { Edge, MarkerType, Node } from 'reactflow'
import { DecisionConfig, ModelRefConfig, SignalType, TestQueryResult } from '../types'
import { EDGE_COLORS, LAYOUT_CONFIG, SIGNAL_LATENCY } from '../constants'
import { collectRuleConditions, summarizeRuleNode } from './ruleTree'

export interface ModelConnection {
  modelRef: ModelRefConfig
  decisionName: string
  sourceId: string
  hasReasoning: boolean
  reasoningEffort?: string
}

export type DecisionDensityMode = 'compact' | 'balanced' | 'cinematic'

export interface LayoutInteractions {
  expandHiddenDecisions?: boolean
  onExpandHiddenDecisions?: () => void
  focusMode?: boolean
  focusedDecisionName?: string | null
  onFocusDecision?: (decisionName: string) => void
}

export interface GraphBuildResult {
  nodes: import('reactflow').Node[]
  edges: Edge[]
  nodeDimensions: Map<string, { width: number; height: number }>
  hiddenDecisionCount: number
  visibleDecisionCount: number
}

export const DENSITY_VISIBLE_DECISION_LIMIT: Record<DecisionDensityMode, number> = {
  compact: 16,
  balanced: 12,
  cinematic: 8,
}

export function createFlowEdge(baseEdge: Partial<Edge>): Edge {
  return {
    ...baseEdge,
  } as Edge
}

export function getDecisionNodeHeight(decision: DecisionConfig, collapsed: boolean): number {
  const { decisionBaseHeight, decisionConditionHeight } = LAYOUT_CONFIG

  if (collapsed) return 90

  const visibleRuleLineCount = (decision.rules?.conditions || [])
    .slice(0, 4)
    .reduce((total, condition) => {
      const wrappedLines = Math.max(1, Math.ceil(summarizeRuleNode(condition).length / 30))
      return total + Math.min(wrappedLines, 3)
    }, 0)
  const hasAlgorithm = decision.algorithm && decision.algorithm.type !== 'static'
  const hasPlugins = decision.plugins && decision.plugins.length > 0
  const hasReasoning = decision.modelRefs?.some((m) => m.use_reasoning)

  let height = decisionBaseHeight
  height += visibleRuleLineCount * decisionConditionHeight
  if (hasAlgorithm) height += 18
  if (hasPlugins) height += 18
  if (hasReasoning) height += 18
  const modelCount = Math.min(decision.modelRefs?.length || 0, 2)
  height += modelCount * 20

  return Math.max(height, 140)
}

export function getDecisionReachability(
  decision: DecisionConfig,
  configuredSignals: ReadonlySet<string>,
): { isFallback: boolean; isUnreachable: boolean; unreachableReason?: string } {
  const leafConditions = collectRuleConditions(decision.rules)
  if (leafConditions.length === 0) {
    return { isFallback: true, isUnreachable: false }
  }
  const hasConfiguredCondition = leafConditions.some((condition) =>
    configuredSignals.has(`${condition.type}:${condition.name}`),
  )
  return {
    isFallback: false,
    isUnreachable: !hasConfiguredCondition,
    ...(hasConfiguredCondition ? {} : { unreachableReason: 'Referenced signals not configured' }),
  }
}

export function getSignalGroupHeight(signals: { name: string }[], collapsed: boolean): number {
  const { signalGroupBaseHeight, signalItemHeight } = LAYOUT_CONFIG
  if (collapsed) return 70
  const itemCount = Math.min(signals.length, 5)
  return signalGroupBaseHeight + itemCount * signalItemHeight
}

export function appendDynamicSignalGroups({
  testResult,
  activeSignalTypes,
  nodes,
  edges,
  nodeDimensions,
  sourceId,
}: {
  testResult?: TestQueryResult | null
  activeSignalTypes: SignalType[]
  nodes: Node[]
  edges: Edge[]
  nodeDimensions: Map<string, { width: number; height: number }>
  sourceId: string
}): void {
  if (!testResult?.matchedSignals?.length) return
  const existingGroupTypes = new Set(activeSignalTypes)
  const dynamicSignalsByType = new Map<SignalType, { name: string; confidence?: number }[]>()
  testResult.matchedSignals.forEach((signal) => {
    if (existingGroupTypes.has(signal.type)) return
    const signals = dynamicSignalsByType.get(signal.type) ?? []
    signals.push({ name: signal.name, confidence: signal.score })
    dynamicSignalsByType.set(signal.type, signals)
  })
  dynamicSignalsByType.forEach((signals, signalType) => {
    const signalGroupId = `signal-group-${signalType}`
    const syntheticSignals = signals.map((signal) => ({
      type: signalType,
      name: signal.name,
      description: `Detected by ML model (confidence: ${signal.confidence ? `${(signal.confidence * 100).toFixed(0)}%` : 'N/A'})`,
      latency: SIGNAL_LATENCY[signalType] || '~100ms',
      config: {},
      isDynamic: true,
    }))
    nodeDimensions.set(signalGroupId, {
      width: 160,
      height: getSignalGroupHeight(syntheticSignals, false),
    })
    nodes.push({
      id: signalGroupId,
      type: 'signalGroupNode',
      position: { x: 0, y: 0 },
      data: {
        signalType,
        signals: syntheticSignals,
        collapsed: false,
        isHighlighted: true,
        isDynamic: true,
      },
    })
    edges.push(
      createFlowEdge({
        id: `e-${sourceId}-${signalGroupId}`,
        source: sourceId,
        target: signalGroupId,
        animated: true,
        style: { stroke: EDGE_COLORS.normal, strokeWidth: 2, strokeDasharray: '5, 5' },
        markerEnd: { type: MarkerType.ArrowClosed, color: EDGE_COLORS.normal },
      }),
    )
    activeSignalTypes.push(signalType)
  })
}

export function getPluginChainHeight(plugins: { type: string }[], collapsed: boolean): number {
  const { pluginChainBaseHeight, pluginItemHeight } = LAYOUT_CONFIG
  if (collapsed) return 55
  const itemCount = Math.min(plugins.length, 4)
  return pluginChainBaseHeight + itemCount * pluginItemHeight
}

export function getPhysicalModelKey(modelRef: ModelRefConfig): string {
  const parts = [modelRef.model]
  if (modelRef.lora_name) parts.push(`lora-${modelRef.lora_name}`)
  return parts.join('|')
}

export function getModelConfigKey(modelRef: ModelRefConfig): string {
  const parts = [modelRef.model]
  if (modelRef.use_reasoning) parts.push('reasoning')
  if (modelRef.reasoning_effort) parts.push(`effort-${modelRef.reasoning_effort}`)
  if (modelRef.lora_name) parts.push(`lora-${modelRef.lora_name}`)
  return parts.join('|')
}
