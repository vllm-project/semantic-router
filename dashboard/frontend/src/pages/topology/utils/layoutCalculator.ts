// topology/utils/layoutCalculator.ts - Layout calculation for Full View using Dagre

import { Node, Edge, MarkerType } from 'reactflow'
import Dagre from '@dagrejs/dagre'
import {
  ParsedTopology,
  CollapseState,
  TestQueryResult,
} from '../types'
import {
  TOPOLOGY_LAYER_LAYOUT,
  EDGE_COLORS,
  MODEL_NODE_WIDTH,
} from '../constants'
import { buildLayoutGraph } from './layoutGraphBuilder'

interface LayoutResult {
  nodes: Node[]
  edges: Edge[]
  meta: LayoutMeta
}

type LayerName = keyof typeof TOPOLOGY_LAYER_LAYOUT.x
export type DecisionDensityMode = 'compact' | 'balanced' | 'cinematic'

interface LayoutMeta {
  hiddenDecisionCount: number
  visibleDecisionCount: number
  totalDecisionCount: number
}

interface LayerFrame {
  left: number
  center: number
  right: number
  width: number
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

const DENSITY_HORIZONTAL_GAP_SCALE: Record<DecisionDensityMode, number> = {
  compact: 0.92,
  balanced: 1.08,
  cinematic: 1.2,
}

const DENSITY_FRAME_PADDING_SCALE: Record<DecisionDensityMode, number> = {
  compact: 0.9,
  balanced: 1.04,
  cinematic: 1.12,
}

const ORDERED_LAYERS: LayerName[] = [
  'client',
  'signals',
  'projections',
  'decisions',
  'algorithms',
  'pluginChains',
  'models',
]

const HORIZONTAL_GAP_BY_LAYER: Record<LayerName, number> = {
  client: TOPOLOGY_LAYER_LAYOUT.horizontalGap.clientToSignals,
  signals: TOPOLOGY_LAYER_LAYOUT.horizontalGap.signalsToProjections,
  projections: TOPOLOGY_LAYER_LAYOUT.horizontalGap.projectionsToDecisions,
  decisions: TOPOLOGY_LAYER_LAYOUT.horizontalGap.decisionsToAlgorithms,
  algorithms: TOPOLOGY_LAYER_LAYOUT.horizontalGap.algorithmsToPluginChains,
  pluginChains: TOPOLOGY_LAYER_LAYOUT.horizontalGap.pluginChainsToModels,
  models: 0,
}

function getAdaptiveLayerSpacing(layerName: LayerName, nodeCount: number): number {
  const rule = TOPOLOGY_LAYER_LAYOUT.verticalSpacing[layerName]
  if (nodeCount <= rule.compactThreshold) return rule.base
  const overflow = nodeCount - rule.compactThreshold
  return Math.max(rule.min, rule.base - overflow * rule.compactStep)
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
  const {
    nodes,
    edges,
    nodeDimensions,
    hiddenDecisionCount,
    visibleDecisionCount,
  } = buildLayoutGraph(
    topology,
    collapseState,
    highlightedPath,
    testResult,
    layoutOptions,
    densityMode,
    isHighlighted
  )

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

  // Group nodes by layer
  const nodesByLayer: Record<LayerName, Node[]> = {
    client: [],
    signals: [],
    projections: [],
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
    } else if (node.id.startsWith('projection-group-')) {
      nodesByLayer.projections.push(node)
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

  const decisionLaneRule = TOPOLOGY_LAYER_LAYOUT.lanes.decisions
  const decisionMaxPerLane = Math.min(6, decisionLaneRule.maxPerLane)
  const regularDecisionCount = nodesByLayer.decisions.filter(node => node.id.startsWith('decision-')).length
  const decisionLaneCount = Math.max(
    1,
    Math.ceil(Math.max(regularDecisionCount, 1) / decisionMaxPerLane)
  )
  const decisionLaneSpan = Math.max(0, decisionLaneCount - 1) * decisionLaneRule.laneGap * laneGapScale

  const signalLaneRule = TOPOLOGY_LAYER_LAYOUT.lanes.signals
  const signalLaneCount = nodesByLayer.signals.length >= signalLaneRule.enableAt
    ? Math.max(
        1,
        Math.min(
          signalLaneRule.maxLanes,
          Math.ceil(nodesByLayer.signals.length / signalLaneRule.maxPerLane)
        )
      )
    : 1
  const signalLaneSpan = Math.max(0, signalLaneCount - 1) * signalLaneRule.laneGap * laneGapScale

  const projectionLaneRule = TOPOLOGY_LAYER_LAYOUT.lanes.projections
  const projectionLaneCount = nodesByLayer.projections.length >= projectionLaneRule.enableAt
    ? Math.max(
        1,
        Math.min(
          projectionLaneRule.maxLanes,
          Math.ceil(nodesByLayer.projections.length / projectionLaneRule.maxPerLane)
        )
      )
    : 1
  const projectionLaneSpan = Math.max(0, projectionLaneCount - 1) * projectionLaneRule.laneGap * laneGapScale

  const modelLaneRule = TOPOLOGY_LAYER_LAYOUT.lanes.models
  const modelLaneCount = nodesByLayer.models.length >= modelLaneRule.enableAt
    ? Math.max(
        1,
        Math.min(
          modelLaneRule.maxLanes,
          Math.ceil(nodesByLayer.models.length / modelLaneRule.maxPerLane)
        )
      )
    : 1
  const modelLaneSpan = Math.max(0, modelLaneCount - 1) * modelLaneRule.laneGap * laneGapScale

  const getLayerMaxNodeWidth = (layerName: LayerName): number => {
    const fallbackWidth = layerName === 'models' ? MODEL_NODE_WIDTH : 180
    return nodesByLayer[layerName].reduce((maxWidth, node) => {
      const dim = nodeDimensions.get(node.id)
      return Math.max(maxWidth, dim?.width ?? fallbackWidth)
    }, fallbackWidth)
  }

  const getLayerFrameWidth = (layerName: LayerName): number => {
    const baseWidth = getLayerMaxNodeWidth(layerName)
    const framePadding = TOPOLOGY_LAYER_LAYOUT.framePadding[layerName] * DENSITY_FRAME_PADDING_SCALE[densityMode]

    switch (layerName) {
      case 'signals':
        return baseWidth + framePadding * 2 + signalLaneSpan
      case 'projections':
        return baseWidth + framePadding * 2 + projectionLaneSpan
      case 'decisions':
        return baseWidth + framePadding * 2 + decisionLaneSpan
      case 'algorithms':
        return baseWidth + framePadding * 2 + Math.max(0, decisionLaneCount - 1) * TOPOLOGY_LAYER_LAYOUT.lanes.algorithms.laneGap * laneGapScale
      case 'pluginChains':
        return baseWidth + framePadding * 2 + Math.max(0, decisionLaneCount - 1) * TOPOLOGY_LAYER_LAYOUT.lanes.pluginChains.laneGap * laneGapScale
      case 'models':
        return baseWidth + framePadding * 2 + modelLaneSpan
      default:
        return baseWidth + framePadding * 2
    }
  }

  const layerFrames = ORDERED_LAYERS.reduce<Record<LayerName, LayerFrame>>((frames, layerName, index) => {
    const width = getLayerFrameWidth(layerName)
    const previousLayer = ORDERED_LAYERS[index - 1]
    const previousRight = previousLayer ? frames[previousLayer].right : 0
    const gap = previousLayer
      ? HORIZONTAL_GAP_BY_LAYER[previousLayer] * DENSITY_HORIZONTAL_GAP_SCALE[densityMode]
      : 0
    const left = previousRight + gap

    frames[layerName] = {
      left,
      center: left + width / 2,
      right: left + width,
      width,
    }

    return frames
  }, {} as Record<LayerName, LayerFrame>)

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
  const getIncomingBarycenter = (nodeId: string, currentLayerLeft: number): number | null => {
    const sourceIds = incomingSourcesByTarget.get(nodeId)
    if (!sourceIds || sourceIds.length === 0) return null

    const sourceCenters = sourceIds
      .map(sourceId => nodeById.get(sourceId))
      .filter((sourceNode): sourceNode is Node => Boolean(sourceNode))
      .filter(sourceNode => (sourceNode.position?.x ?? 0) < currentLayerLeft)
      .map(sourceNode => getNodeCenterY(sourceNode))
      .filter(centerY => Number.isFinite(centerY))

    if (sourceCenters.length === 0) return null

    const sum = sourceCenters.reduce((acc, centerY) => acc + centerY, 0)
    return sum / sourceCenters.length
  }

  const sortByBarycenter = (layerLeft: number) => (a: Node, b: Node) => {
    const aBarycenter = getIncomingBarycenter(a.id, layerLeft)
    const bBarycenter = getIncomingBarycenter(b.id, layerLeft)

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

  const placeStack = (orderedNodes: Node[], centerX: number, spacing: number): void => {
    if (orderedNodes.length === 0) return
    const totalHeight = orderedNodes.reduce((sum, node) => {
      const dim = nodeDimensions.get(node.id) || { width: 150, height: 80 }
      return sum + dim.height
    }, 0)
    const totalSpacing = Math.max(orderedNodes.length - 1, 0) * spacing
    let currentY = -(totalHeight + totalSpacing) / 2

    orderedNodes.forEach(node => {
      const dim = nodeDimensions.get(node.id) || { width: 150, height: 80 }
      node.position = { x: centerX - dim.width / 2, y: currentY }
      currentY += dim.height + spacing
    })
  }

  const decisionLaneByName = new Map<string, number>()
  const decisionCenterYByName = new Map<string, number>()
  const decisionLaneOffsets = getLaneOffsets(decisionLaneCount, decisionLaneRule.laneGap * laneGapScale)

  const placeDecisionLayer = (): void => {
    const layerNodes = nodesByLayer.decisions
    if (layerNodes.length === 0) return

    const layerFrame = layerFrames.decisions
    const spacing = Math.max(8, getAdaptiveLayerSpacing('decisions', layerNodes.length) * spacingScale)
    const orderedNodes = [...layerNodes].sort(sortByBarycenter(layerFrame.left))

    const regularDecisionNodes = orderedNodes.filter(node => node.id.startsWith('decision-'))
    const auxiliaryNodes = orderedNodes.filter(node => !node.id.startsWith('decision-'))

    const lanes: Node[][] = Array.from({ length: decisionLaneCount }, () => [])
    const laneChunkSize = Math.max(1, decisionMaxPerLane)

    regularDecisionNodes.forEach((node, index) => {
      const laneIndex = Math.min(decisionLaneCount - 1, Math.floor(index / laneChunkSize))
      lanes[laneIndex].push(node)
    })

    if (auxiliaryNodes.length > 0) {
      const centerLane = Math.floor(decisionLaneCount / 2)
      lanes[centerLane].push(...auxiliaryNodes)
    }

    lanes.forEach((laneNodes, laneIndex) => {
      const laneCenterX = layerFrame.center + decisionLaneOffsets[laneIndex]
      placeStack(laneNodes, laneCenterX, spacing)
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

    const layerFrame = layerFrames[layerName]
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
        x: layerFrame.center + alignedLaneOffsets[laneIndex] - dim.width / 2,
        y: decisionCenterY - dim.height / 2,
      }
    })

    if (fallbackNodes.length > 0) {
      const orderedFallback = [...fallbackNodes].sort(sortByBarycenter(layerFrame.left))
      placeStack(orderedFallback, layerFrame.center, spacing)
    }
  }

  const placeWrappedLayer = (layerName: 'models'): void => {
    const layerNodes = nodesByLayer[layerName]
    if (layerNodes.length === 0) return

    const layerFrame = layerFrames[layerName]
    const spacing = Math.max(8, getAdaptiveLayerSpacing(layerName, layerNodes.length) * spacingScale)
    const orderedNodes = [...layerNodes].sort(sortByBarycenter(layerFrame.left))
    const laneRule = TOPOLOGY_LAYER_LAYOUT.lanes.models
    const laneOffsets = getLaneOffsets(modelLaneCount, laneRule.laneGap * laneGapScale)
    const laneChunkSize = Math.max(1, Math.ceil(orderedNodes.length / modelLaneCount))

    const lanes: Node[][] = Array.from({ length: modelLaneCount }, () => [])
    orderedNodes.forEach((node, index) => {
      const laneIndex = Math.min(modelLaneCount - 1, Math.floor(index / laneChunkSize))
      lanes[laneIndex].push(node)
    })

    lanes.forEach((laneNodes, laneIndex) => {
      placeStack(laneNodes, layerFrame.center + laneOffsets[laneIndex], spacing)
    })
  }

  const placePackedLayer = (
    layerName: 'signals' | 'projections',
    laneCount: number,
    laneGap: number
  ): void => {
    const layerNodes = nodesByLayer[layerName]
    if (layerNodes.length === 0) return

    const layerFrame = layerFrames[layerName]
    const spacing = Math.max(8, getAdaptiveLayerSpacing(layerName, layerNodes.length) * spacingScale)
    const orderedNodes = [...layerNodes].sort(sortByBarycenter(layerFrame.left))

    if (laneCount <= 1) {
      placeStack(orderedNodes, layerFrame.center, spacing)
      return
    }

    const laneOffsets = getLaneOffsets(laneCount, laneGap * laneGapScale)
    const laneChunkSize = Math.max(1, Math.ceil(orderedNodes.length / laneCount))
    const lanes: Node[][] = Array.from({ length: laneCount }, () => [])

    orderedNodes.forEach((node, index) => {
      const laneIndex = Math.min(laneCount - 1, Math.floor(index / laneChunkSize))
      lanes[laneIndex].push(node)
    })

    lanes.forEach((laneNodes, laneIndex) => {
      placeStack(laneNodes, layerFrame.center + laneOffsets[laneIndex], spacing)
    })
  }

  const placedLayers = new Set<LayerName>()

  placeDecisionLayer()
  placedLayers.add('decisions')

  const projectionLayerNodes = nodesByLayer.projections
  if (projectionLayerNodes.length > 0) {
    placePackedLayer('projections', projectionLaneCount, projectionLaneRule.laneGap)
    placedLayers.add('projections')
  }

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
      const routeDim = nodeDimensions.get('default-route') || { width: 200, height: 140 }
      const routeCenterY = (defaultRouteNode.position?.y ?? 0) + routeDim.height / 2

      // Find the default model node (could be shared with a decision)
      const normalizedDefaultKey = topology.defaultModel.replace(/[^a-zA-Z0-9]/g, '-')
      const defaultModelId = `model-${normalizedDefaultKey}`
      const defaultModelNode = nodeById.get(defaultModelId)
        || nodes.find(n => n.type === 'modelNode' && n.data.modelRef?.model === topology.defaultModel)

      if (defaultModelNode) {
        const modelDim = nodeDimensions.get(defaultModelNode.id) || { width: MODEL_NODE_WIDTH, height: 80 }
        // Only reposition if the default model is NOT shared with other decisions
        // (i.e. only connected from default-route)
        const isShared = defaultModelNode.data.fromDecisions
          && defaultModelNode.data.fromDecisions.length > 1
          && defaultModelNode.data.fromDecisions.some((d: string) => d !== 'default')
        if (!isShared) {
          defaultModelNode.position = {
            x: defaultModelNode.position?.x ?? layerFrames.models.center - modelDim.width / 2,
            y: routeCenterY - modelDim.height / 2,
          }
        }
      }
    }
  }

  // Apply standard placement for the remaining layers.
  ;(Object.entries(nodesByLayer) as [LayerName, Node[]][]).forEach(([layerName, layerNodes]) => {
    if (layerNodes.length === 0 || placedLayers.has(layerName)) return

    if (layerName === 'signals') {
      placePackedLayer('signals', signalLaneCount, signalLaneRule.laneGap)
      return
    }

    const layerFrame = layerFrames[layerName]
    const orderedNodes = [...layerNodes].sort(sortByBarycenter(layerFrame.left))

    if (orderedNodes.length === 1 && layerName === 'client') {
      const dim = nodeDimensions.get(orderedNodes[0].id) || { width: 150, height: 80 }
      orderedNodes[0].position = { x: layerFrame.center - dim.width / 2, y: 0 }
      return
    }

    const spacing = Math.max(8, getAdaptiveLayerSpacing(layerName, orderedNodes.length) * spacingScale)
    placeStack(orderedNodes, layerFrame.center, spacing)
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
      visibleDecisionCount,
      totalDecisionCount: topology.decisions.length,
    },
  }
}
