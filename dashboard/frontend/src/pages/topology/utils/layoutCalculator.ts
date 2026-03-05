// topology/utils/layoutCalculator.ts - Layout calculation for Full View using Dagre

import { Node, Edge, MarkerType } from 'reactflow'
import Dagre from '@dagrejs/dagre'
import {
  ParsedTopology,
  SignalType,
  CollapseState,
  DecisionConfig,
  ModelRefConfig,
  TestQueryResult,
} from '../types'
import {
  LAYOUT_CONFIG,
  TOPOLOGY_LAYER_LAYOUT,
  SIGNAL_TYPES,
  SIGNAL_LATENCY,
  EDGE_COLORS,
} from '../constants'
import { groupSignalsByType } from './topologyParser'

interface LayoutResult {
  nodes: Node[]
  edges: Edge[]
  meta: LayoutMeta
}

interface ModelConnection {
  modelRef: ModelRefConfig
  decisionName: string
  sourceId: string
  hasReasoning: boolean
  reasoningEffort?: string
}

type LayerName = keyof typeof TOPOLOGY_LAYER_LAYOUT.x
export type DecisionDensityMode = 'compact' | 'balanced' | 'cinematic'

interface LayoutMeta {
  hiddenDecisionCount: number
  visibleDecisionCount: number
  totalDecisionCount: number
}

interface LayoutOptions {
  densityMode?: DecisionDensityMode
  expandHiddenDecisions?: boolean
  onExpandHiddenDecisions?: () => void
  focusMode?: boolean
  focusedDecisionName?: string | null
  onFocusDecision?: (decisionName: string) => void
}

const DENSITY_SPACING_SCALE: Record<DecisionDensityMode, number> = {
  compact: 0.82,
  balanced: 1,
  cinematic: 1.24,
}

const DENSITY_LANE_GAP_SCALE: Record<DecisionDensityMode, number> = {
  compact: 0.9,
  balanced: 1,
  cinematic: 1.12,
}

const DENSITY_VISIBLE_DECISION_LIMIT: Record<DecisionDensityMode, number> = {
  compact: 16,
  balanced: 12,
  cinematic: 8,
}

// Helper function to create edge using each node's default handles.
// Node handles are configured as left-in / right-out for LR flow.
function createFlowEdge(baseEdge: Partial<Edge>): Edge {
  return {
    ...baseEdge,
  } as Edge
}

function getAdaptiveLayerSpacing(layerName: LayerName, nodeCount: number): number {
  const rule = TOPOLOGY_LAYER_LAYOUT.verticalSpacing[layerName]
  if (nodeCount <= rule.compactThreshold) return rule.base
  const overflow = nodeCount - rule.compactThreshold
  return Math.max(rule.min, rule.base - overflow * rule.compactStep)
}

// Calculate decision node height based on content
function getDecisionNodeHeight(decision: DecisionConfig, collapsed: boolean): number {
  const { decisionBaseHeight, decisionConditionHeight } = LAYOUT_CONFIG

  if (collapsed) return 90

  const conditionCount = Math.min(decision.rules?.conditions?.length || 0, 4)
  const hasAlgorithm = decision.algorithm && decision.algorithm.type !== 'static'
  const hasPlugins = decision.plugins && decision.plugins.length > 0
  const hasReasoning = decision.modelRefs?.some(m => m.use_reasoning)

  let height = decisionBaseHeight
  height += conditionCount * decisionConditionHeight
  if (hasAlgorithm) height += 18
  if (hasPlugins) height += 18
  if (hasReasoning) height += 18
  const modelCount = Math.min(decision.modelRefs?.length || 0, 2)
  height += modelCount * 20

  return Math.max(height, 140)
}

// Calculate signal group node height
function getSignalGroupHeight(signals: { name: string }[], collapsed: boolean): number {
  const { signalGroupBaseHeight, signalItemHeight } = LAYOUT_CONFIG
  if (collapsed) return 70
  const itemCount = Math.min(signals.length, 5)
  return signalGroupBaseHeight + itemCount * signalItemHeight
}

// Calculate plugin chain node height
function getPluginChainHeight(plugins: { type: string }[], collapsed: boolean): number {
  const { pluginChainBaseHeight, pluginItemHeight } = LAYOUT_CONFIG
  if (collapsed) return 55
  const itemCount = Math.min(plugins.length, 4)
  return pluginChainBaseHeight + itemCount * pluginItemHeight
}

// Generate unique key for a model - now based on physical model only (not reasoning config)
// This allows the same physical model to be shared across different reasoning modes
function getPhysicalModelKey(modelRef: ModelRefConfig): string {
  // Physical model key: base model + LoRA (if any)
  // Reasoning configuration is NOT part of the key - same model can have different modes
  const parts = [modelRef.model]
  if (modelRef.lora_name) parts.push(`lora-${modelRef.lora_name}`)
  return parts.join('|')
}

// Generate unique key for a specific model configuration (for highlighting purposes)
// This includes reasoning info to match backend highlightedPath format
function getModelConfigKey(modelRef: ModelRefConfig): string {
  const parts = [modelRef.model]
  if (modelRef.use_reasoning) parts.push('reasoning')
  if (modelRef.reasoning_effort) parts.push(`effort-${modelRef.reasoning_effort}`)
  if (modelRef.lora_name) parts.push(`lora-${modelRef.lora_name}`)
  return parts.join('|')
}

/**
 * Calculate full topology layout using Dagre for automatic node positioning
 * This ensures no overlapping nodes while maintaining logical flow
 */
export function calculateFullLayout(
  topology: ParsedTopology,
  collapseState: CollapseState,
  highlightedPath: string[] = [],
  testResult?: TestQueryResult | null,
  layoutOptions?: LayoutOptions
): LayoutResult {
  const nodes: Node[] = []
  const edges: Edge[] = []
  const densityMode = layoutOptions?.densityMode ?? 'balanced'
  const spacingScale = DENSITY_SPACING_SCALE[densityMode]
  const laneGapScale = DENSITY_LANE_GAP_SCALE[densityMode]

  // Helper to check if node is highlighted
  const isHighlighted = (id: string): boolean => {
    // Exact match first
    if (highlightedPath.includes(id)) return true
    
    // For model nodes: compare normalized versions (handle special char differences)
    // Backend: model-qwen2-5-7b-reasoning  Frontend: model-qwen2-5-7b-reasoning
    if (id.startsWith('model-')) {
      const normalizedId = id.toLowerCase().replace(/[^a-z0-9-]/g, '-')
      return highlightedPath.some(path => {
        if (!path.startsWith('model-')) return false
        const normalizedPath = path.toLowerCase().replace(/[^a-z0-9-]/g, '-')
        // Exact match after normalization
        return normalizedId === normalizedPath
      })
    }
    
    // For plugin chain nodes
    if (id.startsWith('plugin-chain-')) {
      const decisionName = id.substring(13)
      return highlightedPath.some(path => {
        if (path.startsWith('plugins-')) {
          return decisionName === path.substring(8)
        }
        if (path.startsWith('plugin-chain-')) {
          return decisionName === path.substring(13)
        }
        return false
      })
    }
    
    return false
  }

  // Group signals by type
  const signalGroups = groupSignalsByType(topology.signals)
  const activeSignalTypes = SIGNAL_TYPES.filter(type => signalGroups[type].length > 0)

  // ============== Build node dimensions map ==============
  const nodeDimensions: Map<string, { width: number; height: number }> = new Map()

  // ============== 1. Client Node ==============
  const clientId = 'client'
  nodeDimensions.set(clientId, { width: 120, height: 80 })
  nodes.push({
    id: clientId,
    type: 'clientNode',
    position: { x: 0, y: 0 }, // Will be set by Dagre
    data: {
      label: 'User Query',
      isHighlighted: isHighlighted(clientId),
    },
  })

  // ============== 2. Global Plugins (Temporarily disabled) ==============
  // Global plugins are now shown per-decision in Plugin Chain, so we skip them here
  const lastSourceId = clientId

  // ============== 3. Signal Groups ==============
  activeSignalTypes.forEach(signalType => {
    const signals = signalGroups[signalType]
    if (signals.length === 0) return

    const signalGroupId = `signal-group-${signalType}`
    const isCollapsed = collapseState.signalGroups[signalType]
    const nodeHeight = getSignalGroupHeight(signals, isCollapsed)

    nodeDimensions.set(signalGroupId, { width: 160, height: nodeHeight })

    nodes.push({
      id: signalGroupId,
      type: 'signalGroupNode',
      position: { x: 0, y: 0 },
      data: {
        signalType,
        signals,
        collapsed: isCollapsed,
        isHighlighted: isHighlighted(signalGroupId),
      },
    })

    edges.push(createFlowEdge({
      id: `e-${lastSourceId}-${signalGroupId}`,
      source: lastSourceId,
      target: signalGroupId,
      style: {
        stroke: EDGE_COLORS.normal,
        strokeWidth: 1.5,
      },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: EDGE_COLORS.normal,
      },
    }))
  })

  // ============== 3.5. Dynamic Signal Groups from Test Result ==============
  // Add signal groups for matched signals that don't exist in config
  // (e.g., user_feedback detected by ML model but not configured in user_feedback_rules)
  if (testResult?.matchedSignals?.length) {
    const existingGroupTypes = new Set(activeSignalTypes)
    const dynamicSignalsByType = new Map<SignalType, { name: string; confidence?: number }[]>()
    
    // Group test result signals by type
    testResult.matchedSignals.forEach(signal => {
      if (!existingGroupTypes.has(signal.type)) {
        if (!dynamicSignalsByType.has(signal.type)) {
          dynamicSignalsByType.set(signal.type, [])
        }
        dynamicSignalsByType.get(signal.type)!.push({
          name: signal.name,
          confidence: signal.score,
        })
      }
    })
    
    // Create dynamic signal group nodes
    dynamicSignalsByType.forEach((signals, signalType) => {
      const signalGroupId = `signal-group-${signalType}`
      
      // Create synthetic signal configs for display
      const syntheticSignals = signals.map(s => ({
        type: signalType,
        name: s.name,
        description: `Detected by ML model (confidence: ${s.confidence ? (s.confidence * 100).toFixed(0) + '%' : 'N/A'})`,
        latency: SIGNAL_LATENCY[signalType] || '~100ms',
        config: {},
        isDynamic: true, // Mark as dynamically detected
      }))
      
      const nodeHeight = getSignalGroupHeight(syntheticSignals, false)
      nodeDimensions.set(signalGroupId, { width: 160, height: nodeHeight })
      
      nodes.push({
        id: signalGroupId,
        type: 'signalGroupNode',
        position: { x: 0, y: 0 },
        data: {
          signalType,
          signals: syntheticSignals,
          collapsed: false,
          isHighlighted: true, // Always highlight dynamic signals from test result
          isDynamic: true, // Mark the group as dynamic
        },
      })
      
      // Connect from client to dynamic signal group
      edges.push(createFlowEdge({
        id: `e-${lastSourceId}-${signalGroupId}`,
        source: lastSourceId,
        target: signalGroupId,
        animated: true,
        style: {
          stroke: EDGE_COLORS.normal,
          strokeWidth: 2,
          strokeDasharray: '5, 5', // Dashed to indicate dynamic
        },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: EDGE_COLORS.normal,
        },
      }))
      
      // Add to active signal types for decision routing
      activeSignalTypes.push(signalType)
    })
  }

  // ============== 4. Decisions ==============
  // Track the final source node for each decision
  const decisionFinalSources: Record<string, string> = {}
  const forcedVisibleDecisionNames = new Set<string>()
  highlightedPath
    .filter(id => id.startsWith('decision-'))
    .forEach(id => forcedVisibleDecisionNames.add(id.substring(9)))
  if (testResult?.matchedDecision) forcedVisibleDecisionNames.add(testResult.matchedDecision)
  if (layoutOptions?.focusedDecisionName) forcedVisibleDecisionNames.add(layoutOptions.focusedDecisionName)

  const sortedDecisions = [...topology.decisions].sort((a, b) => b.priority - a.priority)
  const defaultVisibleLimit = DENSITY_VISIBLE_DECISION_LIMIT[densityMode]
  const visibleDecisions = layoutOptions?.expandHiddenDecisions
    ? sortedDecisions
    : sortedDecisions.filter((decision, index) => index < defaultVisibleLimit || forcedVisibleDecisionNames.has(decision.name))
  const hiddenDecisionCount = Math.max(0, topology.decisions.length - visibleDecisions.length)

  // Determine the default upstream source for decisions without signal connections
  // Prefer first signal group if exists, otherwise use last global plugin
  const signalGroupIds = activeSignalTypes.map(t => `signal-group-${t}`)
  const defaultUpstream = signalGroupIds.length > 0 ? signalGroupIds[0] : lastSourceId

  // Create a set of existing signal group IDs for quick lookup
  const existingSignalGroups = new Set(signalGroupIds)

  visibleDecisions.forEach(decision => {
    const decisionId = `decision-${decision.name}`
    const isRulesCollapsed = collapseState.decisions[decision.name]
    const nodeHeight = getDecisionNodeHeight(decision, isRulesCollapsed)

    nodeDimensions.set(decisionId, { width: 200, height: nodeHeight })

    // Check if decision has valid conditions that can be matched
    // A decision is "unreachable" if:
    // 1. It has no conditions (empty rules.conditions), OR
    // 2. All its conditions reference signal types that don't exist
    const hasConditions = decision.rules.conditions.length > 0
    const hasValidConditions = hasConditions && decision.rules.conditions.some(
      cond => existingSignalGroups.has(`signal-group-${cond.type}`)
    )
    const isUnreachable = !hasValidConditions

    nodes.push({
      id: decisionId,
      type: 'decisionNode',
      position: { x: 0, y: 0 },
      data: {
        decision,
        rulesCollapsed: isRulesCollapsed,
        isHighlighted: isHighlighted(decisionId),
        isFocusTarget: layoutOptions?.focusMode && layoutOptions?.focusedDecisionName === decision.name,
        focusModeEnabled: layoutOptions?.focusMode ?? false,
        onFocusDecision: layoutOptions?.onFocusDecision,
        isUnreachable,  // Pass unreachable flag to node
        unreachableReason: !hasConditions 
          ? 'No conditions defined' 
          : 'Referenced signals not configured',
      },
    })

    // Edges from signal groups to decision
    const connectedSignalTypes = new Set<SignalType>()
    decision.rules.conditions.forEach(cond => {
      connectedSignalTypes.add(cond.type)
    })

    let hasConnection = false
    connectedSignalTypes.forEach(signalType => {
      const signalGroupId = `signal-group-${signalType}`
      if (nodes.find(n => n.id === signalGroupId)) {
        hasConnection = true
        edges.push(createFlowEdge({
          id: `e-${signalGroupId}-${decisionId}`,
          source: signalGroupId,
          target: decisionId,
          style: {
            stroke: EDGE_COLORS.normal,
            strokeWidth: 1.5,
          },
          markerEnd: {
            type: MarkerType.ArrowClosed,
            color: EDGE_COLORS.normal,
          },
          label: decision.priority ? `P${decision.priority}` : '',
          labelStyle: { fontSize: 9, fill: '#888' },
          labelBgStyle: { fill: '#1a1a2e', fillOpacity: 0.8 },
        }))
      }
    })

    // If no valid signal connections found, connect from default upstream
    if (!hasConnection) {
      edges.push(createFlowEdge({
        id: `e-${defaultUpstream}-${decisionId}`,
        source: defaultUpstream,
        target: decisionId,
        style: { stroke: EDGE_COLORS.normal, strokeWidth: 1.5 },
        markerEnd: { type: MarkerType.ArrowClosed, color: EDGE_COLORS.normal },
      }))
    }

    let currentSourceId = decisionId

    // ============== 5. Algorithm Node (only for non-ReMoM algorithms) ==============
    // ReMoM is displayed in the plugin chain, not as a separate node
    const hasAlgorithm = decision.algorithm && decision.algorithm.type !== 'static' && decision.modelRefs.length > 1
    const isRemomAlgorithm = hasAlgorithm && decision.algorithm?.type === 'remom'

    if (hasAlgorithm && !isRemomAlgorithm) {
      const algorithmId = `algorithm-${decision.name}`
      nodeDimensions.set(algorithmId, { width: 140, height: 60 })

      nodes.push({
        id: algorithmId,
        type: 'algorithmNode',
        position: { x: 0, y: 0 },
        data: {
          algorithm: decision.algorithm,
          decisionName: decision.name,
          isHighlighted: isHighlighted(algorithmId),
        },
      })

      edges.push(createFlowEdge({
        id: `e-${currentSourceId}-${algorithmId}`,
        source: currentSourceId,
        target: algorithmId,
        style: { stroke: EDGE_COLORS.normal, strokeWidth: 2 },
        markerEnd: { type: MarkerType.ArrowClosed, color: EDGE_COLORS.normal },
      }))

      currentSourceId = algorithmId
    }

    // ============== 6. Plugin Chain Node ==============
    // Include ReMoM algorithm in the plugin chain if present
    const hasPluginsOrRemom = (decision.plugins && decision.plugins.length > 0) || isRemomAlgorithm

    if (hasPluginsOrRemom) {
      const pluginChainId = `plugin-chain-${decision.name}`
      const isPluginCollapsed = collapseState.pluginChains[decision.name]
      const plugins = decision.plugins || []

      // Calculate height including algorithm if present
      const baseHeight = isRemomAlgorithm ? 30 : 0  // Extra height for algorithm
      const pluginHeight = getPluginChainHeight(plugins, isPluginCollapsed) + baseHeight

      nodeDimensions.set(pluginChainId, { width: 160, height: pluginHeight })

      const globalCachePlugin = topology.globalPlugins.find(p => p.type === 'semantic_cache')

      nodes.push({
        id: pluginChainId,
        type: 'pluginChainNode',
        position: { x: 0, y: 0 },
        data: {
          decisionName: decision.name,
          plugins,
          algorithm: isRemomAlgorithm ? decision.algorithm : undefined,  // Pass algorithm to plugin chain
          collapsed: isPluginCollapsed,
          isHighlighted: isHighlighted(pluginChainId),
          globalCacheEnabled: globalCachePlugin?.enabled,
          globalCacheThreshold: globalCachePlugin?.config?.similarity_threshold as number | undefined,
        },
      })

      edges.push(createFlowEdge({
        id: `e-${currentSourceId}-${pluginChainId}`,
        source: currentSourceId,
        target: pluginChainId,
        style: { stroke: EDGE_COLORS.normal, strokeWidth: 2 },
        markerEnd: { type: MarkerType.ArrowClosed, color: EDGE_COLORS.normal },
      }))

      currentSourceId = pluginChainId
    }

    decisionFinalSources[decision.name] = currentSourceId
  })

  if (hiddenDecisionCount > 0 && !layoutOptions?.expandHiddenDecisions) {
    const moreDecisionsId = 'more-decisions'
    nodeDimensions.set(moreDecisionsId, { width: 200, height: 86 })
    nodes.push({
      id: moreDecisionsId,
      type: 'moreDecisionsNode',
      position: { x: 0, y: 0 },
      data: {
        hiddenCount: hiddenDecisionCount,
        onExpand: layoutOptions?.onExpandHiddenDecisions,
      },
    })
  }

  // ============== 5. Default Route Node ==============
  // Add a default route node when a default model is configured
  // This provides visual feedback when no decision matches and fallback is used
  const defaultRouteId = 'default-route'
  if (topology.defaultModel) {
    nodeDimensions.set(defaultRouteId, { width: 160, height: 80 })

    nodes.push({
      id: defaultRouteId,
      type: 'defaultRouteNode',
      position: { x: 0, y: 0 },
      data: {
        label: 'Default Route',
        defaultModel: topology.defaultModel,
        isHighlighted: isHighlighted(defaultRouteId),
      },
    })

    // Connect default route from client (bypasses signal matching)
    edges.push(createFlowEdge({
      id: `e-${clientId}-${defaultRouteId}`,
      source: clientId,
      target: defaultRouteId,
      style: {
        stroke: EDGE_COLORS.normal,
        strokeWidth: 1.5,
        strokeDasharray: '8, 4',  // Dashed to indicate fallback path
      },
      markerEnd: { type: MarkerType.ArrowClosed, color: EDGE_COLORS.normal },
      label: 'fallback',
      labelStyle: { fontSize: 9, fill: '#888' },
      labelBgStyle: { fill: '#1a1a2e', fillOpacity: 0.8 },
    }))
  }

  // ============== 6. Fallback Decision Node (Dynamic) ==============
  // Add a fallback decision node when test result shows a system fallback decision
  // e.g., "low_confidence_general" or "high_confidence_specialized"
  const fallbackDecisionId = 'fallback-decision'
  let fallbackDecisionSourceId: string | null = null
  
  if (testResult?.isFallbackDecision && testResult.matchedDecision) {
    nodeDimensions.set(fallbackDecisionId, { width: 180, height: 100 })
    
    nodes.push({
      id: fallbackDecisionId,
      type: 'fallbackDecisionNode',
      position: { x: 0, y: 0 },
      data: {
        decisionName: testResult.matchedDecision,
        fallbackReason: testResult.fallbackReason,
        defaultModel: topology.defaultModel,
        isHighlighted: isHighlighted(fallbackDecisionId) || highlightedPath.includes(`decision-${testResult.matchedDecision}`),
      },
    })
    
    // Connect from matched signal groups to fallback decision
    const matchedSignalTypes = new Set<SignalType>()
    testResult.matchedSignals?.forEach(signal => {
      matchedSignalTypes.add(signal.type)
    })
    
    let hasSignalConnection = false
    matchedSignalTypes.forEach(signalType => {
      const signalGroupId = `signal-group-${signalType}`
      if (nodes.find(n => n.id === signalGroupId)) {
        hasSignalConnection = true
        edges.push(createFlowEdge({
          id: `e-${signalGroupId}-${fallbackDecisionId}`,
          source: signalGroupId,
          target: fallbackDecisionId,
          animated: true,
          style: {
            stroke: EDGE_COLORS.highlighted,
            strokeWidth: 2,
            strokeDasharray: '5, 5',
          },
          markerEnd: {
            type: MarkerType.ArrowClosed,
            color: EDGE_COLORS.highlighted,
          },
          label: 'fallback',
          labelStyle: { fontSize: 9, fill: '#fff' },
          labelBgStyle: { fill: '#FF9800', fillOpacity: 0.8 },
        }))
      }
    })
    
    // If no signal connections, connect from client directly
    if (!hasSignalConnection) {
      edges.push(createFlowEdge({
        id: `e-${clientId}-${fallbackDecisionId}`,
        source: clientId,
        target: fallbackDecisionId,
        animated: true,
        style: {
          stroke: EDGE_COLORS.highlighted,
          strokeWidth: 2,
          strokeDasharray: '5, 5',
        },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: EDGE_COLORS.highlighted,
        },
      }))
    }
    
    fallbackDecisionSourceId = fallbackDecisionId
  }

  // ============== 7. Model Nodes ==============
  // For multi-model decisions (with algorithms), create separate nodes for each modelRef
  // For single-model decisions, aggregate by physical model
  const modelConnections: Map<string, ModelConnection[]> = new Map()

  visibleDecisions.forEach(decision => {
    const finalSourceId = decisionFinalSources[decision.name]
    const hasAlgorithm = decision.algorithm && decision.algorithm.type !== 'static'
    const isMultiModel = decision.modelRefs.length > 1

    decision.modelRefs.forEach((modelRef, index) => {
      // For multi-model decisions with algorithms, use unique key per decision+model
      // For single-model decisions, use physical model key for aggregation
      let modelKey: string
      if (hasAlgorithm && isMultiModel) {
        // Unique key: decision name + model + index
        modelKey = `${decision.name}|${modelRef.model}|${index}`
      } else {
        // Physical model key for aggregation
        modelKey = getPhysicalModelKey(modelRef)
      }

      if (!modelConnections.has(modelKey)) {
        modelConnections.set(modelKey, [])
      }
      modelConnections.get(modelKey)!.push({
        modelRef,
        decisionName: decision.name,
        sourceId: finalSourceId,
        hasReasoning: modelRef.use_reasoning || false,
        reasoningEffort: modelRef.reasoning_effort,
      })
    })
  })

  // Create model nodes with aggregated mode information
  modelConnections.forEach((connections, physicalKey) => {
    const modelId = `model-${physicalKey.replace(/[^a-zA-Z0-9]/g, '-')}`
    const primaryConnection = connections[0]
    const fromDecisions = connections.map(c => c.decisionName)
    
    // Aggregate modes for this physical model
    const modes = connections.map(conn => ({
      decisionName: conn.decisionName,
      hasReasoning: conn.hasReasoning,
      reasoningEffort: conn.reasoningEffort,
    }))
    
    // Calculate node height based on number of modes
    const uniqueModes = new Set(modes.map(m => m.hasReasoning ? 'reasoning' : 'standard'))
    const nodeHeight = 80 + (uniqueModes.size > 1 ? 30 : 0)
    
    nodeDimensions.set(modelId, { width: 180, height: nodeHeight })

    // Check if this model node should be highlighted
    // Match against any of the possible config keys for this physical model
    const configKeys = connections.map(c => getModelConfigKey(c.modelRef))
    const modelHighlighted = configKeys.some(configKey => {
      const configModelId = `model-${configKey.replace(/[^a-zA-Z0-9]/g, '-')}`
      return isHighlighted(configModelId)
    }) || isHighlighted(modelId)

    nodes.push({
      id: modelId,
      type: 'modelNode',
      position: { x: 0, y: 0 },
      data: {
        modelRef: primaryConnection.modelRef,
        decisionName: fromDecisions.join(', '),
        fromDecisions,
        isHighlighted: modelHighlighted,
        // New: aggregated mode information
        modes,
        hasMultipleModes: uniqueModes.size > 1,
      },
    })

    // Create edges from each source to this model
    // Edge style reflects the reasoning mode of that specific connection
    connections.forEach(conn => {
      // Generate config-specific ID for edge (to support highlighting specific paths)
      const configKey = getModelConfigKey(conn.modelRef)
      const edgeId = `e-${conn.sourceId}-${modelId}-${conn.hasReasoning ? 'reasoning' : 'std'}`
      
      // Check if this specific edge should be highlighted
      const configModelId = `model-${configKey.replace(/[^a-zA-Z0-9]/g, '-')}`
      const edgeHighlighted = isHighlighted(conn.sourceId) && isHighlighted(configModelId)
      
      edges.push(createFlowEdge({
        id: edgeId,
        source: conn.sourceId,
        target: modelId,
        animated: conn.hasReasoning || edgeHighlighted,
        style: {
          stroke: edgeHighlighted
            ? EDGE_COLORS.highlighted
            : (conn.hasReasoning ? EDGE_COLORS.reasoning : EDGE_COLORS.normal),
          strokeWidth: edgeHighlighted ? 3 : (conn.hasReasoning ? 2.5 : 1.5),
          strokeDasharray: conn.hasReasoning ? '0' : '5, 5',
        },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: edgeHighlighted
            ? EDGE_COLORS.highlighted
            : (conn.hasReasoning ? EDGE_COLORS.reasoning : EDGE_COLORS.normal),
          width: 18,
          height: 18,
        },
        // Show reasoning mode on edge label
        label: conn.hasReasoning
          ? `🧠${conn.reasoningEffort ? ` ${conn.reasoningEffort}` : ''}`
          : '',
        labelStyle: { fontSize: 9, fill: '#fff' },
        labelBgStyle: { fill: conn.hasReasoning ? '#9333ea' : 'transparent', fillOpacity: 0.8 },
        labelBgPadding: [4, 2] as [number, number],
        labelBgBorderRadius: 4,
      }))
    })
  })

  // ============== 8. Default Route to Default Model Edge ==============
  // Connect default route node to the default model (if both exist)
  if (topology.defaultModel) {
    const defaultModelKey = topology.defaultModel
    const normalizedDefaultKey = defaultModelKey.replace(/[^a-zA-Z0-9]/g, '-')
    const defaultModelId = `model-${normalizedDefaultKey}`

    // Check if default model node already exists (it might be used by a decision too)
    // Need to check both exact match and model nodes that contain this model name
    const existingModelNode = nodes.find(n => {
      if (n.type !== 'modelNode') return false
      // Check exact ID match
      if (n.id === defaultModelId) return true
      // Check if the node's model name matches (for multi-model decisions with unique IDs)
      const nodeModelName = n.data.modelRef?.model
      return nodeModelName === topology.defaultModel
    })

    if (existingModelNode) {
      // Default model already exists, just connect to it
      const edgeHighlighted = isHighlighted(defaultRouteId) && isHighlighted(existingModelNode.id)

      edges.push(createFlowEdge({
        id: `e-${defaultRouteId}-${existingModelNode.id}`,
        source: defaultRouteId,
        target: existingModelNode.id,
        animated: edgeHighlighted,
        style: {
          stroke: edgeHighlighted ? EDGE_COLORS.highlighted : EDGE_COLORS.normal,
          strokeWidth: edgeHighlighted ? 3 : 1.5,
          strokeDasharray: '8, 4',
        },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: edgeHighlighted ? EDGE_COLORS.highlighted : EDGE_COLORS.normal,
        },
      }))
    } else {
      // Create a new model node for the default model only if it doesn't exist
      nodeDimensions.set(defaultModelId, { width: 180, height: 80 })

      const modelHighlighted = isHighlighted(defaultModelId)

      nodes.push({
        id: defaultModelId,
        type: 'modelNode',
        position: { x: 0, y: 0 },
        data: {
          modelRef: { model: topology.defaultModel },
          decisionName: 'default',
          fromDecisions: ['default'],
          isHighlighted: modelHighlighted,
          modes: [{ decisionName: 'default', hasReasoning: false }],
          hasMultipleModes: false,
        },
      })

      const edgeHighlighted = isHighlighted(defaultRouteId) && modelHighlighted

      edges.push(createFlowEdge({
        id: `e-${defaultRouteId}-${defaultModelId}`,
        source: defaultRouteId,
        target: defaultModelId,
        animated: edgeHighlighted,
        style: {
          stroke: edgeHighlighted ? EDGE_COLORS.highlighted : EDGE_COLORS.normal,
          strokeWidth: edgeHighlighted ? 3 : 1.5,
          strokeDasharray: '8, 4',
        },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: edgeHighlighted ? EDGE_COLORS.highlighted : EDGE_COLORS.normal,
        },
      }))
    }
  }

  // ============== 8.5. Fallback Decision to Model Edge ==============
  // Connect fallback decision node to the matched model (if both exist)
  if (fallbackDecisionSourceId && testResult?.matchedModels?.length) {
    const matchedModelName = testResult.matchedModels[0]
    const normalizedModelKey = matchedModelName.replace(/[^a-zA-Z0-9]/g, '-')
    const matchedModelId = `model-${normalizedModelKey}`
    
    // Check if model node exists
    const existingModelNode = nodes.find(n => n.id === matchedModelId)
    
    if (existingModelNode) {
      // Connect fallback decision to existing model
      edges.push(createFlowEdge({
        id: `e-${fallbackDecisionSourceId}-${matchedModelId}`,
        source: fallbackDecisionSourceId,
        target: matchedModelId,
        animated: true,
        style: {
          stroke: EDGE_COLORS.highlighted,
          strokeWidth: 2.5,
        },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: EDGE_COLORS.highlighted,
        },
      }))
    } else if (topology.defaultModel) {
      // Fallback to default model connection
      const defaultModelKey = topology.defaultModel.replace(/[^a-zA-Z0-9]/g, '-')
      const defaultModelId = `model-${defaultModelKey}`
      const defaultModelNode = nodes.find(n => n.id === defaultModelId)

      if (defaultModelNode) {
        edges.push(createFlowEdge({
          id: `e-${fallbackDecisionSourceId}-${defaultModelId}`,
          source: fallbackDecisionSourceId,
          target: defaultModelId,
          animated: true,
          style: {
            stroke: EDGE_COLORS.highlighted,
            strokeWidth: 2.5,
          },
          markerEnd: {
            type: MarkerType.ArrowClosed,
            color: EDGE_COLORS.highlighted,
          },
        }))
      }
    }
  }

  // ============== 9. Apply Dagre Layout ==============
  const g = new Dagre.graphlib.Graph().setDefaultEdgeLabel(() => ({}))

  g.setGraph({
    rankdir: 'LR',              // Left to Right
    nodesep: 56,                // Vertical spacing in same rank
    ranksep: 190,               // Horizontal spacing between ranks/columns
    marginx: 80,
    marginy: 80,
    ranker: 'network-simplex',
    align: 'UL',
  })

  // Add nodes with dimensions to Dagre
  nodes.forEach(node => {
    const dim = nodeDimensions.get(node.id) || { width: 150, height: 80 }
    g.setNode(node.id, { width: dim.width, height: dim.height })
  })

  // Add edges to Dagre
  edges.forEach(edge => {
    g.setEdge(edge.source, edge.target)
  })

  // Run layout algorithm
  Dagre.layout(g)

  // Initialize from Dagre positions so each layer keeps a stable ordering
  nodes.forEach(node => {
    const dagreNode = g.node(node.id)
    if (!dagreNode) return
    const dim = nodeDimensions.get(node.id) || { width: 150, height: 80 }
    node.position = {
      x: dagreNode.x - dim.width / 2,
      y: dagreNode.y - dim.height / 2,
    }
  })

  // ============== Three-Layer Architecture (Left -> Right) ==============
  // Layer 1: Input (client + signals)
  // Layer 2: Decision (decision engine)
  // Layer 3: Projection (algorithm/plugin/model execution)
  const LAYER_X_POSITIONS = TOPOLOGY_LAYER_LAYOUT.x

  // Group nodes by layer
  const nodesByLayer: Record<LayerName, Node[]> = {
    client: [],
    signals: [],
    decisions: [],
    algorithms: [],
    pluginChains: [],
    models: [],
  }

  nodes.forEach(node => {
    if (node.id === 'client') {
      nodesByLayer.client.push(node)
    } else if (node.id.startsWith('signal-group-')) {
      nodesByLayer.signals.push(node)
    } else if (node.id.startsWith('decision-') || node.id === 'default-route' || node.id === 'fallback-decision' || node.id === 'more-decisions') {
      nodesByLayer.decisions.push(node)
    } else if (node.id.startsWith('algorithm-')) {
      nodesByLayer.algorithms.push(node)
    } else if (node.id.startsWith('plugin-chain-')) {
      nodesByLayer.pluginChains.push(node)
    } else if (node.id.startsWith('model-')) {
      nodesByLayer.models.push(node)
    }
  })

  const nodeById = new Map<string, Node>()
  nodes.forEach(node => {
    nodeById.set(node.id, node)
  })

  const incomingSourcesByTarget = new Map<string, string[]>()
  edges.forEach(edge => {
    if (!incomingSourcesByTarget.has(edge.target)) {
      incomingSourcesByTarget.set(edge.target, [])
    }
    incomingSourcesByTarget.get(edge.target)!.push(edge.source)
  })

  const getNodeCenterY = (node: Node): number => {
    const dim = nodeDimensions.get(node.id) || { width: 150, height: 80 }
    return (node.position?.y ?? 0) + dim.height / 2
  }

  // Use upstream barycenter ordering to reduce edge crossings in dense layers.
  const getIncomingBarycenter = (nodeId: string, currentLayerX: number): number | null => {
    const sourceIds = incomingSourcesByTarget.get(nodeId)
    if (!sourceIds || sourceIds.length === 0) return null

    const sourceCenters = sourceIds
      .map(sourceId => nodeById.get(sourceId))
      .filter((sourceNode): sourceNode is Node => Boolean(sourceNode))
      .filter(sourceNode => (sourceNode.position?.x ?? 0) < currentLayerX)
      .map(sourceNode => getNodeCenterY(sourceNode))
      .filter(centerY => Number.isFinite(centerY))

    if (sourceCenters.length === 0) return null

    const sum = sourceCenters.reduce((acc, centerY) => acc + centerY, 0)
    return sum / sourceCenters.length
  }

  const sortByBarycenter = (layerX: number) => (a: Node, b: Node) => {
    const aBarycenter = getIncomingBarycenter(a.id, layerX)
    const bBarycenter = getIncomingBarycenter(b.id, layerX)

    if (aBarycenter !== null && bBarycenter !== null && aBarycenter !== bBarycenter) {
      return aBarycenter - bBarycenter
    }
    if (aBarycenter !== null && bBarycenter === null) return -1
    if (aBarycenter === null && bBarycenter !== null) return 1
    return (a.position?.y ?? 0) - (b.position?.y ?? 0)
  }

  const getLaneOffsets = (laneCount: number, laneGap: number): number[] => {
    if (laneCount <= 1) return [0]
    return Array.from({ length: laneCount }, (_, index) => (index - (laneCount - 1) / 2) * laneGap)
  }

  const placeStack = (orderedNodes: Node[], x: number, spacing: number): void => {
    if (orderedNodes.length === 0) return
    const totalHeight = orderedNodes.reduce((sum, node) => {
      const dim = nodeDimensions.get(node.id) || { width: 150, height: 80 }
      return sum + dim.height
    }, 0)
    const totalSpacing = Math.max(orderedNodes.length - 1, 0) * spacing
    let currentY = -(totalHeight + totalSpacing) / 2

    orderedNodes.forEach(node => {
      const dim = nodeDimensions.get(node.id) || { width: 150, height: 80 }
      node.position = { x, y: currentY }
      currentY += dim.height + spacing
    })
  }

  const decisionLaneByName = new Map<string, number>()
  const decisionCenterYByName = new Map<string, number>()
  let decisionLaneCount = 1
  let decisionLaneOffsets = [0]

  const placeDecisionLayer = (): void => {
    const layerNodes = nodesByLayer.decisions
    if (layerNodes.length === 0) return

    const layerX = LAYER_X_POSITIONS.decisions
    const spacing = Math.max(8, getAdaptiveLayerSpacing('decisions', layerNodes.length) * spacingScale)
    const orderedNodes = [...layerNodes].sort(sortByBarycenter(layerX))

    const regularDecisionNodes = orderedNodes.filter(node => node.id.startsWith('decision-'))
    const auxiliaryNodes = orderedNodes.filter(node => !node.id.startsWith('decision-'))

    const laneRule = TOPOLOGY_LAYER_LAYOUT.lanes.decisions
    const maxPerLane = Math.min(6, laneRule.maxPerLane)
    const requiredLanes = Math.ceil(Math.max(regularDecisionNodes.length, 1) / maxPerLane)
    decisionLaneCount = Math.max(1, requiredLanes)
    decisionLaneOffsets = getLaneOffsets(decisionLaneCount, laneRule.laneGap * laneGapScale)

    const lanes: Node[][] = Array.from({ length: decisionLaneCount }, () => [])
    const laneChunkSize = Math.max(1, maxPerLane)

    regularDecisionNodes.forEach((node, index) => {
      const laneIndex = Math.min(decisionLaneCount - 1, Math.floor(index / laneChunkSize))
      lanes[laneIndex].push(node)
    })

    if (auxiliaryNodes.length > 0) {
      const centerLane = Math.floor(decisionLaneCount / 2)
      lanes[centerLane].push(...auxiliaryNodes)
    }

    lanes.forEach((laneNodes, laneIndex) => {
      const laneX = layerX + decisionLaneOffsets[laneIndex]
      placeStack(laneNodes, laneX, spacing)
      laneNodes.forEach(node => {
        if (!node.id.startsWith('decision-')) return
        const dim = nodeDimensions.get(node.id) || { width: 150, height: 80 }
        const decisionName = node.id.substring(9)
        decisionLaneByName.set(decisionName, laneIndex)
        decisionCenterYByName.set(decisionName, (node.position?.y ?? 0) + dim.height / 2)
      })
    })
  }

  const placeDecisionLinkedLayer = (
    layerName: 'algorithms' | 'pluginChains',
    idPrefix: string,
    laneGap: number
  ): void => {
    const layerNodes = nodesByLayer[layerName]
    if (layerNodes.length === 0) return

    const baseX = LAYER_X_POSITIONS[layerName]
    const spacing = Math.max(8, getAdaptiveLayerSpacing(layerName, layerNodes.length) * spacingScale)
    const alignedLaneOffsets = getLaneOffsets(decisionLaneCount, laneGap * laneGapScale)
    const fallbackNodes: Node[] = []

    layerNodes.forEach(node => {
      if (!node.id.startsWith(idPrefix)) {
        fallbackNodes.push(node)
        return
      }

      const decisionName = node.id.substring(idPrefix.length)
      const laneIndex = decisionLaneByName.get(decisionName)
      const decisionCenterY = decisionCenterYByName.get(decisionName)

      if (laneIndex === undefined || decisionCenterY === undefined) {
        fallbackNodes.push(node)
        return
      }

      const dim = nodeDimensions.get(node.id) || { width: 150, height: 80 }
      node.position = {
        x: baseX + alignedLaneOffsets[laneIndex],
        y: decisionCenterY - dim.height / 2,
      }
    })

    if (fallbackNodes.length > 0) {
      const orderedFallback = [...fallbackNodes].sort(sortByBarycenter(baseX))
      placeStack(orderedFallback, baseX, spacing)
    }
  }

  const placeWrappedLayer = (layerName: 'models'): void => {
    const layerNodes = nodesByLayer[layerName]
    if (layerNodes.length === 0) return

    const layerX = LAYER_X_POSITIONS[layerName]
    const spacing = Math.max(8, getAdaptiveLayerSpacing(layerName, layerNodes.length) * spacingScale)
    const orderedNodes = [...layerNodes].sort(sortByBarycenter(layerX))
    const laneRule = TOPOLOGY_LAYER_LAYOUT.lanes.models
    const laneCount = orderedNodes.length >= laneRule.enableAt
      ? Math.max(1, Math.min(laneRule.maxLanes, Math.ceil(orderedNodes.length / laneRule.maxPerLane)))
      : 1
    const laneOffsets = getLaneOffsets(laneCount, laneRule.laneGap * laneGapScale)
    const laneChunkSize = Math.max(1, Math.ceil(orderedNodes.length / laneCount))

    const lanes: Node[][] = Array.from({ length: laneCount }, () => [])
    orderedNodes.forEach((node, index) => {
      const laneIndex = Math.min(laneCount - 1, Math.floor(index / laneChunkSize))
      lanes[laneIndex].push(node)
    })

    lanes.forEach((laneNodes, laneIndex) => {
      placeStack(laneNodes, layerX + laneOffsets[laneIndex], spacing)
    })
  }

  const placedLayers = new Set<LayerName>()

  placeDecisionLayer()
  placedLayers.add('decisions')

  placeDecisionLinkedLayer('algorithms', 'algorithm-', TOPOLOGY_LAYER_LAYOUT.lanes.algorithms.laneGap)
  placedLayers.add('algorithms')

  placeDecisionLinkedLayer('pluginChains', 'plugin-chain-', TOPOLOGY_LAYER_LAYOUT.lanes.pluginChains.laneGap)
  placedLayers.add('pluginChains')

  placeWrappedLayer('models')
  placedLayers.add('models')

  // ============== Fix default model alignment ==============
  // After placeWrappedLayer, the default model may be mis-positioned because
  // its upstream (default-route) sits at the bottom of the decisions stack.
  // Re-align the default model's Y center to match the default-route node.
  if (topology.defaultModel) {
    const defaultRouteNode = nodeById.get('default-route')
    if (defaultRouteNode) {
      const routeDim = nodeDimensions.get('default-route') || { width: 160, height: 80 }
      const routeCenterY = (defaultRouteNode.position?.y ?? 0) + routeDim.height / 2

      // Find the default model node (could be shared with a decision)
      const normalizedDefaultKey = topology.defaultModel.replace(/[^a-zA-Z0-9]/g, '-')
      const defaultModelId = `model-${normalizedDefaultKey}`
      const defaultModelNode = nodeById.get(defaultModelId)
        || nodes.find(n => n.type === 'modelNode' && n.data.modelRef?.model === topology.defaultModel)

      if (defaultModelNode) {
        const modelDim = nodeDimensions.get(defaultModelNode.id) || { width: 180, height: 80 }
        // Only reposition if the default model is NOT shared with other decisions
        // (i.e. only connected from default-route)
        const isShared = defaultModelNode.data.fromDecisions
          && defaultModelNode.data.fromDecisions.length > 1
          && defaultModelNode.data.fromDecisions.some((d: string) => d !== 'default')
        if (!isShared) {
          defaultModelNode.position = {
            x: defaultModelNode.position?.x ?? LAYER_X_POSITIONS.models,
            y: routeCenterY - modelDim.height / 2,
          }
        }
      }
    }
  }

  // Apply standard placement for the remaining layers.
  ;(Object.entries(nodesByLayer) as [LayerName, Node[]][]).forEach(([layerName, layerNodes]) => {
    if (layerNodes.length === 0 || placedLayers.has(layerName)) return

    const layerX = LAYER_X_POSITIONS[layerName]
    const orderedNodes = [...layerNodes].sort(sortByBarycenter(layerX))

    if (orderedNodes.length === 1 && layerName === 'client') {
      orderedNodes[0].position = { x: layerX, y: 0 }
      return
    }

    const spacing = Math.max(8, getAdaptiveLayerSpacing(layerName, orderedNodes.length) * spacingScale)
    placeStack(orderedNodes, layerX, spacing)
  })

  // ============== 9. Apply Highlighting ==============
  if (highlightedPath.length > 0) {
    // Build a set of highlighted node IDs for quick lookup
    const highlightedNodeIds = new Set<string>()
    nodes.forEach(node => {
      if (isHighlighted(node.id)) {
        highlightedNodeIds.add(node.id)
      }
    })

    // Build forward edge map
    const edgeMap = new Map<string, string[]>()
    edges.forEach(edge => {
      if (!edgeMap.has(edge.source)) {
        edgeMap.set(edge.source, [])
      }
      edgeMap.get(edge.source)!.push(edge.target)
    })

    // Find the specific path from client to the highlighted model
    // Only include nodes that are in the highlightedPath from backend
    
    const nodesOnPath = new Set<string>()
    
    // Add all nodes that backend marked as highlighted
    highlightedNodeIds.forEach(id => nodesOnPath.add(id))
    
    // Find the highlighted decision (the one that was matched)
    const highlightedDecision = Array.from(highlightedNodeIds).find(id => id.startsWith('decision-'))
    
    if (highlightedDecision) {
      const decisionName = highlightedDecision.substring(9) // Remove 'decision-' prefix
      
      // Always include client
      nodesOnPath.add('client')
      
      // Only include signal groups that were actually matched (already in highlightedNodeIds)
      // Do NOT auto-include all signal groups connected to the decision
      
      // Include algorithm and plugin-chain for this specific decision
      const algorithmId = `algorithm-${decisionName}`
      const pluginChainId = `plugin-chain-${decisionName}`
      
      if (nodes.find(n => n.id === algorithmId)) {
        nodesOnPath.add(algorithmId)
      }
      if (nodes.find(n => n.id === pluginChainId)) {
        nodesOnPath.add(pluginChainId)
      }
    }

    // Highlight edges where both source and target are on the path
    edges.forEach(edge => {
      const sourceOnPath = nodesOnPath.has(edge.source)
      const targetOnPath = nodesOnPath.has(edge.target)
      
      if (sourceOnPath && targetOnPath) {
        edge.style = {
          ...edge.style,
          stroke: EDGE_COLORS.highlighted,
          strokeWidth: 4,
          strokeDasharray: '0',
          filter: 'drop-shadow(0 0 6px rgba(255, 215, 0, 0.8))',
        }
        edge.markerEnd = {
          type: MarkerType.ArrowClosed,
          color: EDGE_COLORS.highlighted,
          width: 24,
          height: 24,
        }
        edge.animated = true
        edge.className = 'highlighted-edge'
      }
    })
    
    // Update node highlight status for nodes on path
    nodes.forEach(node => {
      if (nodesOnPath.has(node.id)) {
        node.data.isHighlighted = true
      }
    })
  }

  if (layoutOptions?.focusMode && layoutOptions?.focusedDecisionName) {
    const focusedDecisionId = `decision-${layoutOptions.focusedDecisionName}`
    const focusedNodeIds = new Set<string>()

    if (nodes.some(node => node.id === focusedDecisionId)) {
      focusedNodeIds.add(focusedDecisionId)
      focusedNodeIds.add('client')

      const outgoingBySource = new Map<string, string[]>()
      const incomingByTarget = new Map<string, string[]>()
      edges.forEach(edge => {
        if (!outgoingBySource.has(edge.source)) outgoingBySource.set(edge.source, [])
        outgoingBySource.get(edge.source)!.push(edge.target)
        if (!incomingByTarget.has(edge.target)) incomingByTarget.set(edge.target, [])
        incomingByTarget.get(edge.target)!.push(edge.source)
      })

      const queue: string[] = [focusedDecisionId]
      while (queue.length > 0) {
        const current = queue.shift()!
        const downstream = outgoingBySource.get(current) || []
        downstream.forEach(next => {
          if (focusedNodeIds.has(next)) return
          focusedNodeIds.add(next)
          queue.push(next)
        })
      }

      const directInputs = incomingByTarget.get(focusedDecisionId) || []
      directInputs.forEach(sourceId => {
        focusedNodeIds.add(sourceId)
        const upstream = incomingByTarget.get(sourceId) || []
        upstream.forEach(upId => focusedNodeIds.add(upId))
      })
    }

    if (focusedNodeIds.size > 0) {
      nodes.forEach(node => {
        const isFocused = focusedNodeIds.has(node.id)
        if (!isFocused) {
          node.style = {
            ...(node.style || {}),
            opacity: 0.16,
            filter: 'grayscale(0.4)',
          }
        } else if (node.id === focusedDecisionId) {
          node.style = {
            ...(node.style || {}),
            opacity: 1,
            filter: 'drop-shadow(0 0 14px rgba(118, 185, 0, 0.6))',
          }
        }
      })

      edges.forEach(edge => {
        const inFocusPath = focusedNodeIds.has(edge.source) && focusedNodeIds.has(edge.target)
        edge.style = {
          ...(edge.style || {}),
          opacity: inFocusPath ? 1 : 0.08,
          strokeWidth: inFocusPath ? Math.max(Number(edge.style?.strokeWidth || 1.5), 2.6) : 1,
        }
        edge.animated = inFocusPath ? true : false
      })
    }
  }

  return {
    nodes,
    edges,
    meta: {
      hiddenDecisionCount,
      visibleDecisionCount: visibleDecisions.length,
      totalDecisionCount: topology.decisions.length,
    },
  }
}
