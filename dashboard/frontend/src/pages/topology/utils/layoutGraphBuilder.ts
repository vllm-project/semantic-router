import { Edge, MarkerType, Node } from 'reactflow'
import {
  CollapseState,
  ParsedTopology,
  SignalType,
  TestQueryResult,
} from '../types'
import {
  EDGE_COLORS,
  MODEL_NODE_WIDTH,
  SIGNAL_LATENCY,
  SIGNAL_TYPES,
} from '../constants'
import {
  buildProjectionGroups,
  buildProjectionOutputNodeMap,
  formatProjectionInputLabel,
  formatProjectionOutputLabel,
  groupProjectionInputsBySignalType,
} from './layoutProjectionSupport'
import {
  createFlowEdge,
  DecisionDensityMode,
  DENSITY_VISIBLE_DECISION_LIMIT,
  getDecisionNodeHeight,
  getModelConfigKey,
  getPhysicalModelKey,
  getPluginChainHeight,
  getSignalGroupHeight,
  GraphBuildResult,
  LayoutInteractions,
  ModelConnection,
} from './layoutGraphBuilderSupport'
import { groupSignalsByType } from './topologyParser'
import { collectRuleConditions, collectRuleSignalTypes } from './ruleTree'

export function buildLayoutGraph(
  topology: ParsedTopology,
  collapseState: CollapseState,
  highlightedPath: string[],
  testResult: TestQueryResult | null | undefined,
  layoutOptions: LayoutInteractions | undefined,
  densityMode: DecisionDensityMode,
  isHighlighted: (id: string) => boolean
): GraphBuildResult {
  const nodes: Node[] = []
  const edges: Edge[] = []

  const signalGroups = groupSignalsByType(topology.signals)
  const activeSignalTypes = SIGNAL_TYPES.filter(type => type !== 'projection' && signalGroups[type].length > 0)
  const nodeDimensions: Map<string, { width: number; height: number }> = new Map()

  const clientId = 'client'
  nodeDimensions.set(clientId, { width: 120, height: 80 })
  nodes.push({
    id: clientId,
    type: 'clientNode',
    position: { x: 0, y: 0 },
    data: {
      label: 'User Query',
      isHighlighted: isHighlighted(clientId),
    },
  })

  const lastSourceId = clientId

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

  if (testResult?.matchedSignals?.length) {
    const existingGroupTypes = new Set(activeSignalTypes)
    const dynamicSignalsByType = new Map<SignalType, { name: string; confidence?: number }[]>()

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

    dynamicSignalsByType.forEach((signals, signalType) => {
      const signalGroupId = `signal-group-${signalType}`
      const syntheticSignals = signals.map(signal => ({
        type: signalType,
        name: signal.name,
        description: `Detected by ML model (confidence: ${signal.confidence ? (signal.confidence * 100).toFixed(0) + '%' : 'N/A'})`,
        latency: SIGNAL_LATENCY[signalType] || '~100ms',
        config: {},
        isDynamic: true,
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
          isHighlighted: true,
          isDynamic: true,
        },
      })

      edges.push(createFlowEdge({
        id: `e-${lastSourceId}-${signalGroupId}`,
        source: lastSourceId,
        target: signalGroupId,
        animated: true,
        style: {
          stroke: EDGE_COLORS.normal,
          strokeWidth: 2,
          strokeDasharray: '5, 5',
        },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: EDGE_COLORS.normal,
        },
      }))

      activeSignalTypes.push(signalType)
    })
  }

  const projectionGroups = buildProjectionGroups(signalGroups.projection)
  const projectionNodeIds = projectionGroups.map((group) => group.nodeId)
  const projectionOutputToNodeId = buildProjectionOutputNodeMap(projectionGroups)

  projectionGroups.forEach((group) => {
    const isCollapsed = collapseState.signalGroups.projection
    const nodeHeight = getSignalGroupHeight(group.outputs, isCollapsed) + 18

    nodeDimensions.set(group.nodeId, { width: 190, height: nodeHeight })
    nodes.push({
      id: group.nodeId,
      type: 'signalGroupNode',
      position: { x: 0, y: 0 },
      data: {
        signalType: 'projection',
        title: group.mappingName,
        subtitle: `score: ${group.sourceScore} · ${group.method}`,
        latencyLabel: SIGNAL_LATENCY.projection,
        signals: group.outputs,
        collapsed: isCollapsed,
        isHighlighted: isHighlighted(group.nodeId),
      },
    })

    groupProjectionInputsBySignalType(group).forEach((inputNames, signalType) => {
      const signalGroupId = `signal-group-${signalType}`
      if (!nodes.find(node => node.id === signalGroupId)) {
        return
      }

      edges.push(createFlowEdge({
        id: `e-${signalGroupId}-${group.nodeId}`,
        source: signalGroupId,
        target: group.nodeId,
        style: {
          stroke: EDGE_COLORS.normal,
          strokeWidth: 1.5,
        },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: EDGE_COLORS.normal,
        },
        label: formatProjectionInputLabel(inputNames),
        labelStyle: { fontSize: 9, fill: '#888' },
        labelBgStyle: { fill: '#1a1a2e', fillOpacity: 0.8 },
      }))
    })
  })

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

  const signalGroupIds = activeSignalTypes.map(type => `signal-group-${type}`)
  const defaultUpstream = signalGroupIds[0] ?? projectionNodeIds[0] ?? lastSourceId
  const configuredSignals = new Set(topology.signals.map(signal => `${signal.type}:${signal.name}`))

  visibleDecisions.forEach(decision => {
    const decisionId = `decision-${decision.name}`
    const isRulesCollapsed = collapseState.decisions[decision.name]
    const nodeHeight = getDecisionNodeHeight(decision, isRulesCollapsed)

    nodeDimensions.set(decisionId, { width: 200, height: nodeHeight })

    const leafConditions = collectRuleConditions(decision.rules)
    const hasConditions = leafConditions.length > 0
    const hasValidConditions = hasConditions && leafConditions.some(
      condition => configuredSignals.has(`${condition.type}:${condition.name}`)
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
        isUnreachable,
        unreachableReason: !hasConditions
          ? 'No conditions defined'
          : 'Referenced signals not configured',
      },
    })

    const connectedSignalTypes = new Set<SignalType>(collectRuleSignalTypes(decision.rules))
    const projectionConditionNamesByNode = new Map<string, string[]>()
    let hasConnection = false

    leafConditions
      .filter(condition => condition.type === 'projection')
      .forEach((condition) => {
        const projectionNodeId = projectionOutputToNodeId.get(condition.name)
        if (!projectionNodeId) {
          return
        }
        hasConnection = true
        if (!projectionConditionNamesByNode.has(projectionNodeId)) {
          projectionConditionNamesByNode.set(projectionNodeId, [])
        }
        projectionConditionNamesByNode.get(projectionNodeId)!.push(condition.name)
      })

    connectedSignalTypes.forEach(signalType => {
      if (signalType === 'projection') {
        return
      }
      const signalGroupId = `signal-group-${signalType}`
      if (nodes.find(node => node.id === signalGroupId)) {
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

    projectionConditionNamesByNode.forEach((projectionNames, projectionNodeId) => {
      edges.push(createFlowEdge({
        id: `e-${projectionNodeId}-${decisionId}`,
        source: projectionNodeId,
        target: decisionId,
        style: {
          stroke: EDGE_COLORS.normal,
          strokeWidth: 1.5,
        },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: EDGE_COLORS.normal,
        },
        label: formatProjectionOutputLabel(projectionNames),
        labelStyle: { fontSize: 9, fill: '#888' },
        labelBgStyle: { fill: '#1a1a2e', fillOpacity: 0.8 },
      }))
    })

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

    const hasPluginsOrRemom = (decision.plugins && decision.plugins.length > 0) || isRemomAlgorithm

    if (hasPluginsOrRemom) {
      const pluginChainId = `plugin-chain-${decision.name}`
      const isPluginCollapsed = collapseState.pluginChains[decision.name]
      const plugins = decision.plugins || []
      const baseHeight = isRemomAlgorithm ? 30 : 0
      const pluginHeight = getPluginChainHeight(plugins, isPluginCollapsed) + baseHeight

      nodeDimensions.set(pluginChainId, { width: 160, height: pluginHeight })

      const globalCachePlugin = topology.globalPlugins.find(plugin => plugin.type === 'semantic_cache')

      nodes.push({
        id: pluginChainId,
        type: 'pluginChainNode',
        position: { x: 0, y: 0 },
        data: {
          decisionName: decision.name,
          plugins,
          algorithm: isRemomAlgorithm ? decision.algorithm : undefined,
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

  const defaultRouteId = 'default-route'
  if (topology.defaultModel) {
    nodeDimensions.set(defaultRouteId, { width: 200, height: 140 })

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

    edges.push(createFlowEdge({
      id: `e-${clientId}-${defaultRouteId}`,
      source: clientId,
      target: defaultRouteId,
      style: {
        stroke: EDGE_COLORS.normal,
        strokeWidth: 1.5,
        strokeDasharray: '8, 4',
      },
      markerEnd: { type: MarkerType.ArrowClosed, color: EDGE_COLORS.normal },
      label: 'fallback',
      labelStyle: { fontSize: 9, fill: '#888' },
      labelBgStyle: { fill: '#1a1a2e', fillOpacity: 0.8 },
    }))
  }

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

    const matchedSignalTypes = new Set<SignalType>()
    testResult.matchedSignals?.forEach(signal => {
      matchedSignalTypes.add(signal.type)
    })

    let hasSignalConnection = false
    matchedSignalTypes.forEach(signalType => {
      const signalGroupId = `signal-group-${signalType}`
      if (nodes.find(node => node.id === signalGroupId)) {
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

  const modelConnections: Map<string, ModelConnection[]> = new Map()

  visibleDecisions.forEach(decision => {
    const finalSourceId = decisionFinalSources[decision.name]
    const hasAlgorithm = decision.algorithm && decision.algorithm.type !== 'static'
    const isMultiModel = decision.modelRefs.length > 1

    decision.modelRefs.forEach((modelRef, index) => {
      let modelKey: string
      if (hasAlgorithm && isMultiModel) {
        modelKey = `${decision.name}|${modelRef.model}|${index}`
      } else {
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

  modelConnections.forEach((connections, physicalKey) => {
    const modelId = `model-${physicalKey.replace(/[^a-zA-Z0-9]/g, '-')}`
    const primaryConnection = connections[0]
    const fromDecisions = connections.map(connection => connection.decisionName)
    const modes = connections.map(connection => ({
      decisionName: connection.decisionName,
      hasReasoning: connection.hasReasoning,
      reasoningEffort: connection.reasoningEffort,
    }))

    const uniqueModes = new Set(modes.map(mode => mode.hasReasoning ? 'reasoning' : 'standard'))
    const nodeHeight = 80 + (uniqueModes.size > 1 ? 30 : 0)

    nodeDimensions.set(modelId, { width: MODEL_NODE_WIDTH, height: nodeHeight })

    const configKeys = connections.map(connection => getModelConfigKey(connection.modelRef))
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
        modes,
        hasMultipleModes: uniqueModes.size > 1,
      },
    })

    connections.forEach(connection => {
      const configKey = getModelConfigKey(connection.modelRef)
      const configModelId = `model-${configKey.replace(/[^a-zA-Z0-9]/g, '-')}`
      const edgeId = `e-${connection.sourceId}-${modelId}-${connection.hasReasoning ? 'reasoning' : 'std'}`
      const edgeHighlighted = isHighlighted(connection.sourceId) && isHighlighted(configModelId)

      edges.push(createFlowEdge({
        id: edgeId,
        source: connection.sourceId,
        target: modelId,
        animated: connection.hasReasoning || edgeHighlighted,
        style: {
          stroke: edgeHighlighted
            ? EDGE_COLORS.highlighted
            : (connection.hasReasoning ? EDGE_COLORS.reasoning : EDGE_COLORS.normal),
          strokeWidth: edgeHighlighted ? 3 : (connection.hasReasoning ? 2.5 : 1.5),
          strokeDasharray: connection.hasReasoning ? '0' : '5, 5',
        },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: edgeHighlighted
            ? EDGE_COLORS.highlighted
            : (connection.hasReasoning ? EDGE_COLORS.reasoning : EDGE_COLORS.normal),
          width: 18,
          height: 18,
        },
        label: connection.hasReasoning
          ? `🧠${connection.reasoningEffort ? ` ${connection.reasoningEffort}` : ''}`
          : '',
        labelStyle: { fontSize: 9, fill: '#fff' },
        labelBgStyle: { fill: connection.hasReasoning ? '#9333ea' : 'transparent', fillOpacity: 0.8 },
        labelBgPadding: [4, 2] as [number, number],
        labelBgBorderRadius: 4,
      }))
    })
  })

  const referencedModelNames = new Set(
    Array.from(modelConnections.values()).flatMap((connections) =>
      connections.map((connection) => connection.modelRef.model)
    )
  )

  topology.models.forEach((model) => {
    if (referencedModelNames.has(model.name)) {
      return
    }

    const standaloneModelId = `model-${model.name.replace(/[^a-zA-Z0-9]/g, '-')}`
    if (nodes.some((node) => node.id === standaloneModelId)) {
      return
    }

    nodeDimensions.set(standaloneModelId, { width: MODEL_NODE_WIDTH, height: 80 })
    nodes.push({
      id: standaloneModelId,
      type: 'modelNode',
      position: { x: 0, y: 0 },
      data: {
        modelRef: {
          model: model.name,
          reasoning_family: model.reasoning_family,
        },
        decisionName: 'Not referenced',
        usageLabel: 'Not referenced by any decision',
        fromDecisions: [],
        isHighlighted: isHighlighted(standaloneModelId),
        modes: [],
        hasMultipleModes: false,
      },
    })
  })

  if (topology.defaultModel) {
    const defaultModelKey = topology.defaultModel
    const normalizedDefaultKey = defaultModelKey.replace(/[^a-zA-Z0-9]/g, '-')
    const defaultModelId = `model-${normalizedDefaultKey}`

    const existingModelNode = nodes.find(node => {
      if (node.type !== 'modelNode') return false
      if (node.id === defaultModelId) return true
      const nodeModelName = node.data.modelRef?.model
      return nodeModelName === topology.defaultModel
    })

    if (existingModelNode) {
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
      nodeDimensions.set(defaultModelId, { width: MODEL_NODE_WIDTH, height: 80 })

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

  if (fallbackDecisionSourceId && testResult?.matchedModels?.length) {
    const matchedModelName = testResult.matchedModels[0]
    const normalizedModelKey = matchedModelName.replace(/[^a-zA-Z0-9]/g, '-')
    const matchedModelId = `model-${normalizedModelKey}`
    const existingModelNode = nodes.find(node => node.id === matchedModelId)

    if (existingModelNode) {
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
      const defaultModelKey = topology.defaultModel.replace(/[^a-zA-Z0-9]/g, '-')
      const defaultModelId = `model-${defaultModelKey}`
      const defaultModelNode = nodes.find(node => node.id === defaultModelId)

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

  return {
    nodes,
    edges,
    nodeDimensions,
    hiddenDecisionCount,
    visibleDecisionCount: visibleDecisions.length,
  }
}
