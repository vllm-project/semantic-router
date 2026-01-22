// topology/TopologyPageEnhanced.tsx - Full Signal-Driven Decision Pipeline Visualization

import React, { useCallback, useEffect } from 'react'
import ReactFlow, {
  Node,
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  useReactFlow,
  ReactFlowProvider,
  ConnectionLineType,
} from 'reactflow'
import 'reactflow/dist/style.css'

import { useTopologyData, useCollapseState, useTestQuery } from './hooks'
import { useTheme } from '../../hooks'
import { customNodeTypes } from './components/CustomNodes'
import {
  TestQueryInput,
  CollapseControls,
  TopologyLegend,
} from './components/ControlPanel'
import { calculateFullLayout } from './utils/layoutCalculator'
import styles from './TopologyPageEnhanced.module.css'

// ============== Inner Flow Component ==============
const TopologyFlow: React.FC = () => {
  const { data, loading, error, refresh } = useTopologyData()
  const { collapseState, expandAll, collapseAll } = useCollapseState()
  const { isDark } = useTheme()
  const {
    testQuery,
    setTestQuery,
    testResult,
    isLoading: isTestLoading,
    runTest,
  } = useTestQuery(data)

  const [nodes, setNodes, onNodesChange] = useNodesState([])
  const [edges, setEdges, onEdgesChange] = useEdgesState([])
  const { fitView } = useReactFlow()

  // Generate full topology layout
  useEffect(() => {
    if (!data) return

    const highlightedPath = testResult?.highlightedPath || []
    const { nodes: newNodes, edges: newEdges } = calculateFullLayout(
      data,
      collapseState,
      highlightedPath,
      testResult
    )
    setNodes(newNodes)
    setEdges(newEdges)
  }, [data, collapseState, testResult, setNodes, setEdges])

  // Fit view after nodes change
  useEffect(() => {
    if (nodes.length > 0) {
      const timer = setTimeout(() => {
        fitView({ padding: 0.2, duration: 300 })
      }, 100)
      return () => clearTimeout(timer)
    }
  }, [nodes.length, fitView])

  const getNodeColor = useCallback((node: Node) => {
    const style = node.style as any
    return style?.background || '#ccc'
  }, [])

  if (loading) {
    return (
      <div className={styles.container}>
        <div className={styles.loading}>
          <div className={styles.spinner}></div>
          <p>Loading topology...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className={styles.container}>
        <div className={styles.error}>
          <span className={styles.errorIcon}>⚠️</span>
          <p>{error}</p>
          <button onClick={refresh} className={styles.retryButton}>
            Retry
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <h1 className={styles.title}>
          Mixture of Models: Signal-Driven Decision Topology
        </h1>
        <button onClick={refresh} className={styles.refreshButton}>
          Refresh
        </button>
      </div>

      <div className={styles.content}>
        {/* Control Panel */}
        <div className={styles.controlPanel}>
          <TestQueryInput
            value={testQuery}
            onChange={setTestQuery}
            onTest={runTest}
            isLoading={isTestLoading}
            result={testResult}
          />
          <CollapseControls onExpandAll={expandAll} onCollapseAll={collapseAll} />
          <TopologyLegend />
        </div>

        {/* Flow Canvas */}
        <div className={styles.flowContainer}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            nodeTypes={customNodeTypes}
            connectionLineType={ConnectionLineType.Bezier}
            defaultEdgeOptions={{
              type: 'bezier',
              style: { strokeWidth: 1.5 },
            }}
            fitView
            fitViewOptions={{ padding: 0.3, minZoom: 0.3, maxZoom: 1.5 }}
            defaultViewport={{ x: 0, y: 0, zoom: 0.6 }}
          >
            <Background 
              variant={BackgroundVariant.Dots}
              gap={20}
              size={1}
              color={isDark ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.08)'}
            />
            <Controls />
            <MiniMap 
              nodeColor={getNodeColor} 
              maskColor={isDark ? 'rgba(0, 0, 0, 0.2)' : 'rgba(255, 255, 255, 0.3)'}
              style={{
                backgroundColor: isDark ? '#141414' : '#ffffff',
              }}
              nodeStrokeWidth={2}
            />
          </ReactFlow>
        </div>
      </div>
    </div>
  )
}

// ============== Wrapper Component ==============
const TopologyPageEnhanced: React.FC = () => {
  return (
    <ReactFlowProvider>
      <TopologyFlow />
    </ReactFlowProvider>
  )
}

export default TopologyPageEnhanced
