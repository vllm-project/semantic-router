import { Edge } from 'reactflow'
import { DecisionConfig, ModelRefConfig } from '../types'
import { LAYOUT_CONFIG } from '../constants'
import { summarizeRuleNode } from './ruleTree'

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
  const hasReasoning = decision.modelRefs?.some(m => m.use_reasoning)

  let height = decisionBaseHeight
  height += visibleRuleLineCount * decisionConditionHeight
  if (hasAlgorithm) height += 18
  if (hasPlugins) height += 18
  if (hasReasoning) height += 18
  const modelCount = Math.min(decision.modelRefs?.length || 0, 2)
  height += modelCount * 20

  return Math.max(height, 140)
}

export function getSignalGroupHeight(signals: { name: string }[], collapsed: boolean): number {
  const { signalGroupBaseHeight, signalItemHeight } = LAYOUT_CONFIG
  if (collapsed) return 70
  const itemCount = Math.min(signals.length, 5)
  return signalGroupBaseHeight + itemCount * signalItemHeight
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
