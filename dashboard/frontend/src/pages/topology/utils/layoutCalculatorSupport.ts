import type { Edge, Node } from 'reactflow'

import { TOPOLOGY_LAYER_LAYOUT } from '../constants'
import type { DecisionDensityMode } from './layoutGraphBuilderSupport'

export interface LayoutMeta {
  hiddenDecisionCount: number
  visibleDecisionCount: number
  totalDecisionCount: number
}

export interface LayoutResult {
  nodes: Node[]
  edges: Edge[]
  meta: LayoutMeta
}

export type LayerName = keyof typeof TOPOLOGY_LAYER_LAYOUT.x

export interface LayerFrame {
  left: number
  center: number
  right: number
  width: number
}

export interface LayoutOptions {
  densityMode?: DecisionDensityMode
  expandHiddenDecisions?: boolean
  onExpandHiddenDecisions?: () => void
  focusMode?: boolean
  focusedDecisionName?: string | null
  onFocusDecision?: (decisionName: string) => void
}

export const DENSITY_SPACING_SCALE: Record<DecisionDensityMode, number> = {
  compact: 0.82,
  balanced: 1,
  cinematic: 1.24,
}

export const DENSITY_LANE_GAP_SCALE: Record<DecisionDensityMode, number> = {
  compact: 0.9,
  balanced: 1,
  cinematic: 1.12,
}

export const DENSITY_HORIZONTAL_GAP_SCALE: Record<DecisionDensityMode, number> = {
  compact: 0.92,
  balanced: 1.08,
  cinematic: 1.2,
}

export const DENSITY_FRAME_PADDING_SCALE: Record<DecisionDensityMode, number> = {
  compact: 0.9,
  balanced: 1.04,
  cinematic: 1.12,
}

export const ORDERED_LAYERS: LayerName[] = [
  'client',
  'signals',
  'projections',
  'decisions',
  'algorithms',
  'pluginChains',
  'models',
]

export const HORIZONTAL_GAP_BY_LAYER: Record<LayerName, number> = {
  client: TOPOLOGY_LAYER_LAYOUT.horizontalGap.clientToSignals,
  signals: TOPOLOGY_LAYER_LAYOUT.horizontalGap.signalsToProjections,
  projections: TOPOLOGY_LAYER_LAYOUT.horizontalGap.projectionsToDecisions,
  decisions: TOPOLOGY_LAYER_LAYOUT.horizontalGap.decisionsToAlgorithms,
  algorithms: TOPOLOGY_LAYER_LAYOUT.horizontalGap.algorithmsToPluginChains,
  pluginChains: TOPOLOGY_LAYER_LAYOUT.horizontalGap.pluginChainsToModels,
  models: 0,
}

export function getAdaptiveLayerSpacing(layerName: LayerName, nodeCount: number): number {
  const rule = TOPOLOGY_LAYER_LAYOUT.verticalSpacing[layerName]
  if (nodeCount <= rule.compactThreshold) return rule.base
  const overflow = nodeCount - rule.compactThreshold
  return Math.max(rule.min, rule.base - overflow * rule.compactStep)
}

export function isTopologyNodeHighlighted(id: string, highlightedPath: string[]): boolean {
  if (highlightedPath.includes(id)) return true
  if (id.startsWith('model-')) {
    const normalizedId = id.toLowerCase().replace(/[^a-z0-9-]/g, '-')
    return highlightedPath.some((path) => {
      if (!path.startsWith('model-')) return false
      return normalizedId === path.toLowerCase().replace(/[^a-z0-9-]/g, '-')
    })
  }
  if (id.startsWith('plugin-chain-')) {
    const decisionName = id.substring(13)
    return highlightedPath.some((path) => {
      if (path.startsWith('plugins-')) return decisionName === path.substring(8)
      if (path.startsWith('plugin-chain-')) return decisionName === path.substring(13)
      return false
    })
  }
  return false
}
