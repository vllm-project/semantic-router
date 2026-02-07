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
  SIGNAL_TYPES,
  SIGNAL_LATENCY,
  EDGE_COLORS,
} from '../constants'
import { groupSignalsByType } from './topologyParser'

interface LayoutResult {
  nodes: Node[]
  edges: Edge[]
}

interface ModelConnection {
  modelRef: ModelRefConfig
  decisionName: string
  sourceId: string
  hasReasoning: boolean
  reasoningEffort?: string
}

// Helper function to create edge with vertical connection points (top-to-bottom layout)
function createVerticalEdge(baseEdge: Partial<Edge>): Edge {
  return {
    ...baseEdge,
    sourceHandle: 'bottom',  // Connect from bottom of source node
    targetHandle: 'top',     // Connect to top of target node
  } as Edge
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
function getSignalGroupHeight(signals: any[], collapsed: boolean): number {
  const { signalGroupBaseHeight, signalItemHeight } = LAYOUT_CONFIG
  if (collapsed) return 70
  const itemCount = Math.min(signals.length, 5)
  return signalGroupBaseHeight + itemCount * signalItemHeight
}

// Calculate plugin chain node height
function getPluginChainHeight(plugins: any[], collapsed: boolean): number {
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
  testResult?: TestQueryResult | null
): LayoutResult {
  const nodes: Node[] = []
  const edges: Edge[] = []

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

    edges.push(createVerticalEdge({
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
      edges.push(createVerticalEdge({
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

  // Determine the default upstream source for decisions without signal connections
  // Prefer first signal group if exists, otherwise use last global plugin
  const signalGroupIds = activeSignalTypes.map(t => `signal-group-${t}`)
  const defaultUpstream = signalGroupIds.length > 0 ? signalGroupIds[0] : lastSourceId

  // Create a set of existing signal group IDs for quick lookup
  const existingSignalGroups = new Set(signalGroupIds)

  topology.decisions.forEach(decision => {
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
        edges.push(createVerticalEdge({
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
      edges.push(createVerticalEdge({
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

      edges.push(createVerticalEdge({
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

      edges.push(createVerticalEdge({
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
    edges.push(createVerticalEdge({
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
        edges.push(createVerticalEdge({
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
      edges.push(createVerticalEdge({
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

  topology.decisions.forEach(decision => {
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
      
      edges.push(createVerticalEdge({
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

      edges.push(createVerticalEdge({
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

      edges.push(createVerticalEdge({
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
      edges.push(createVerticalEdge({
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
        edges.push(createVerticalEdge({
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
    rankdir: 'TB',           // Top to Bottom (changed from LR)
    nodesep: 80,             // Horizontal spacing between nodes in same rank
    ranksep: 100,            // Vertical spacing between ranks/rows
    marginx: 80,
    marginy: 80,
    ranker: 'network-simplex', // Best for DAGs
    align: 'UL',             // Align nodes to upper-left for better distribution
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

  // ============== Neural Network Layer Structure ==============
  // Define fixed Y positions for each layer to create clear neural network structure
  const LAYER_Y_POSITIONS = {
    client: 0,           // Layer 1: User Query
    signals: 280,        // Layer 2: Signals (increased from 200)
    decisions: 680,      // Layer 3: Decisions (increased from 500)
    algorithms: 900,     // Layer 3.5: Algorithms (between decisions and plugin chains)
    pluginChains: 1100,  // Layer 4: Plugin Chains (increased from 1000)
    models: 1400,        // Layer 5: Models (increased from 1300)
  }

  // Group nodes by layer
  const nodesByLayer: Record<string, Node[]> = {
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
    } else if (node.id.startsWith('decision-') || node.id === 'default-route' || node.id === 'fallback-decision') {
      nodesByLayer.decisions.push(node)
    } else if (node.id.startsWith('algorithm-')) {
      nodesByLayer.algorithms.push(node)
    } else if (node.id.startsWith('plugin-chain-')) {
      nodesByLayer.pluginChains.push(node)
    } else if (node.id.startsWith('model-')) {
      nodesByLayer.models.push(node)
    }
  })

  // Apply positions with layer-based Y and centered X
  Object.entries(nodesByLayer).forEach(([layerName, layerNodes]) => {
    if (layerNodes.length === 0) return

    const layerY = LAYER_Y_POSITIONS[layerName as keyof typeof LAYER_Y_POSITIONS]

    // Calculate total width of all nodes in this layer
    const totalWidth = layerNodes.reduce((sum, node) => {
      const dim = nodeDimensions.get(node.id) || { width: 150, height: 80 }
      return sum + dim.width
    }, 0)

    // Different spacing for different layers
    let spacing = 80  // Default spacing
    if (layerName === 'pluginChains') {
      spacing = 150  // More spacing for plugin chains
    } else if (layerName === 'models') {
      spacing = 120  // More spacing for models
    } else if (layerName === 'decisions') {
      spacing = 100  // Slightly more spacing for decisions
    } else if (layerName === 'algorithms') {
      spacing = 100  // Same spacing as decisions
    }

    const totalSpacing = (layerNodes.length - 1) * spacing
    const layerTotalWidth = totalWidth + totalSpacing

    // Start X position to center the layer
    let currentX = -layerTotalWidth / 2

    // Position each node in the layer
    layerNodes.forEach(node => {
      const dim = nodeDimensions.get(node.id) || { width: 150, height: 80 }

      node.position = {
        x: currentX,
        y: layerY,
      }

      currentX += dim.width + spacing
    })
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

  return { nodes, edges }
}
