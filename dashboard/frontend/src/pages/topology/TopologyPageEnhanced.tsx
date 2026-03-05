// topology/TopologyPageEnhanced.tsx - Full Signal-Driven Decision Pipeline Visualization

import React, { useCallback, useEffect, useMemo, useState } from 'react'
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
import { calculateFullLayout, type DecisionDensityMode } from './utils/layoutCalculator'
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
  const [densityMode, setDensityMode] = useState<DecisionDensityMode>('balanced')
  const [expandHiddenDecisions, setExpandHiddenDecisions] = useState(false)
  const [focusMode, setFocusMode] = useState(true)
  const [focusedDecisionName, setFocusedDecisionName] = useState<string | null>(null)
  const [layoutMeta, setLayoutMeta] = useState({
    hiddenDecisionCount: 0,
    visibleDecisionCount: 0,
    totalDecisionCount: 0,
  })

  const handleExpandHiddenDecisions = useCallback(() => {
    setExpandHiddenDecisions(true)
  }, [])

  const handleDecisionFocus = useCallback((decisionName: string) => {
    setFocusedDecisionName(prev => (prev === decisionName ? null : decisionName))
    setFocusMode(true)
  }, [])

  const layoutOptions = useMemo(() => ({
    densityMode,
    expandHiddenDecisions,
    onExpandHiddenDecisions: handleExpandHiddenDecisions,
    focusMode,
    focusedDecisionName: focusMode ? focusedDecisionName : null,
    onFocusDecision: handleDecisionFocus,
  }), [
    densityMode,
    expandHiddenDecisions,
    handleExpandHiddenDecisions,
    focusMode,
    focusedDecisionName,
    handleDecisionFocus,
  ])

  useEffect(() => {
    setExpandHiddenDecisions(false)
  }, [densityMode])

  // Generate full topology layout
  useEffect(() => {
    if (!data) return

    const highlightedPath = testResult?.highlightedPath || []
    const { nodes: newNodes, edges: newEdges, meta } = calculateFullLayout(
      data,
      collapseState,
      highlightedPath,
      testResult,
      layoutOptions
    )
    setNodes(newNodes)
    setEdges(newEdges)
    setLayoutMeta(meta)
  }, [data, collapseState, testResult, layoutOptions, setNodes, setEdges])

  // Fit view after nodes change - with extra bottom padding for input panel
  useEffect(() => {
    if (nodes.length > 0) {
      const timer = setTimeout(() => {
        fitView({
          padding: 0.12,
          duration: 300,
          minZoom: 0.15,
          maxZoom: 1.0
        })
      }, 100)
      return () => clearTimeout(timer)
    }
  }, [nodes.length, fitView])

  const getNodeColor = useCallback((node: Node) => {
    const style = node.style as Record<string, string> | undefined
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
          <div className={styles.layoutToolbar}>
            <div className={styles.toolbarSection}>
              <span className={styles.toolbarLabel}>Density</span>
              <div className={styles.modeSwitch}>
                {(['compact', 'balanced', 'cinematic'] as DecisionDensityMode[]).map(mode => (
                  <button
                    key={mode}
                    className={`${styles.modeBtn} ${densityMode === mode ? styles.modeBtnActive : ''}`}
                    onClick={() => setDensityMode(mode)}
                    type="button"
                  >
                    {mode}
                  </button>
                ))}
              </div>
            </div>

            <div className={styles.toolbarSection}>
              <span className={styles.toolbarLabel}>Decisions</span>
              <div className={styles.toolbarRow}>
                <span className={styles.toolbarValue}>
                  {layoutMeta.visibleDecisionCount}/{layoutMeta.totalDecisionCount}
                </span>
                {layoutMeta.hiddenDecisionCount > 0 && !expandHiddenDecisions && (
                  <button
                    type="button"
                    className={styles.toolBtn}
                    onClick={() => setExpandHiddenDecisions(true)}
                  >
                    +{layoutMeta.hiddenDecisionCount} more
                  </button>
                )}
                {expandHiddenDecisions && layoutMeta.hiddenDecisionCount > 0 && (
                  <button
                    type="button"
                    className={styles.toolBtn}
                    onClick={() => setExpandHiddenDecisions(false)}
                  >
                    Fold Low Priority
                  </button>
                )}
              </div>
            </div>

            <div className={styles.toolbarSection}>
              <span className={styles.toolbarLabel}>Focus</span>
              <div className={styles.toolbarRow}>
                <button
                  type="button"
                  className={`${styles.toolBtn} ${focusMode ? styles.toolBtnActive : ''}`}
                  onClick={() => {
                    setFocusMode(prev => {
                      const next = !prev
                      if (!next) setFocusedDecisionName(null)
                      return next
                    })
                  }}
                >
                  {focusMode ? 'On' : 'Off'}
                </button>
                {focusedDecisionName && (
                  <button
                    type="button"
                    className={styles.toolBtn}
                    onClick={() => setFocusedDecisionName(null)}
                  >
                    Clear
                  </button>
                )}
              </div>
            </div>
          </div>

          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            nodeTypes={customNodeTypes}
            connectionLineType={ConnectionLineType.SmoothStep}
            defaultEdgeOptions={{
              type: 'smoothstep',
              style: { strokeWidth: 1.5 },
            }}
            fitView
            fitViewOptions={{ padding: 0.12, minZoom: 0.15, maxZoom: 1.0 }}
            defaultViewport={{ x: 0, y: 0, zoom: 0.32 }}
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
