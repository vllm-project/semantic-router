// topology/TopologyPageEnhanced.tsx - Full Signal-Driven Decision Pipeline Visualization

import React, { useCallback, useEffect, useState } from 'react'
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
import { TestQueryInput } from './components/ControlPanel'
import { ResultCard } from './components/ResultCard'
import { calculateFullLayout } from './utils/layoutCalculator'
import styles from './TopologyPageEnhanced.module.css'

// ============== Inner Flow Component ==============
const TopologyFlow: React.FC = () => {
  const { data, loading, error, refresh } = useTopologyData()
  const { collapseState } = useCollapseState()
  const { isDark } = useTheme()
  const {
    testQuery,
    setTestQuery,
    testResult,
    isLoading: isTestLoading,
    runTest,
    clearResult,
  } = useTestQuery(data)

  const [nodes, setNodes, onNodesChange] = useNodesState([])
  const [edges, setEdges, onEdgesChange] = useEdgesState([])
  const { fitView } = useReactFlow()
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)

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

  // Fit view after nodes change - with extra bottom padding for input panel
  useEffect(() => {
    if (nodes.length > 0) {
      const timer = setTimeout(() => {
        fitView({
          padding: 0.1,
          duration: 300,
          minZoom: 0.2,
          maxZoom: 1.0
        })
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
      <div className={styles.content}>
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
            fitViewOptions={{ padding: 0.1, minZoom: 0.2, maxZoom: 1.0 }}
            defaultViewport={{ x: 0, y: 0, zoom: 0.4 }}
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

        {/* Bottom Control Panel */}
        <div className={`${styles.bottomPanel} ${sidebarCollapsed ? styles.collapsed : ''}`}>
          {/* Toggle Button */}
          <button
            className={styles.bottomToggle}
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            title={sidebarCollapsed ? 'Expand Panel' : 'Collapse Panel'}
          >
            {sidebarCollapsed ? '▲' : '▼'}
          </button>

          {/* Panel Content */}
          <div className={styles.bottomPanelContent}>
            <TestQueryInput
              value={testQuery}
              onChange={setTestQuery}
              onTest={runTest}
              isLoading={isTestLoading}
            />
          </div>
        </div>

        {/* Result Card */}
        <ResultCard
          result={testResult}
          onClose={clearResult}
        />
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
