import React, { useEffect, useState } from 'react'
import ReactFlow, {
  Node,
  Edge,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  useReactFlow,
  ReactFlowProvider,
  MarkerType,
  Position,
} from 'reactflow'
import 'reactflow/dist/style.css'
import styles from './TopologyPage.module.css'

interface ConfigData {
  bert_model?: {
    model_id?: string
    threshold?: number
    use_cpu?: boolean
  }
  prompt_guard?: {
    enabled: boolean
    model_id?: string
    use_modernbert?: boolean
  }
  classifier?: {
    category_model?: {
      model_id?: string
      use_modernbert?: boolean
      threshold?: number
    }
    pii_model?: {
      enabled?: boolean
      model_id?: string
      use_modernbert?: boolean
    }
  }
  semantic_cache?: {
    enabled: boolean
    backend_type?: string
    similarity_threshold?: number
  }
  // Legacy format
  categories?: Array<{
    name: string
    system_prompt?: string
    // model_scores can be array (Python CLI) or object (Legacy: {"gpt-4": 0.9})
    model_scores?: Array<{
      model: string
      score: number
      use_reasoning?: boolean
    }> | Record<string, number>
  }>
  model_config?: {
    [key: string]: {
      reasoning_family?: string
    }
  }
  // Python CLI format
  signals?: {
    domains?: Array<{
      name: string
      description?: string
    }>
  }
  decisions?: Array<{
    name: string
    description?: string
    priority?: number
    rules?: {
      operator?: string
      conditions?: Array<{
        type: string
        name: string
      }>
    }
    modelRefs?: Array<{
      model: string
      use_reasoning?: boolean
    }>
    plugins?: Array<{
      type: string
      configuration?: Record<string, unknown>
    }>
  }>
  providers?: {
    models?: Array<{
      name: string
      reasoning_family?: string
    }>
    default_model?: string
  }
}

// Helper: Normalize model_scores from object to array (Legacy format uses object)
interface NormalizedModelScore {
  model: string
  score: number
  use_reasoning?: boolean
}

const normalizeModelScores = (
  modelScores: Array<{ model: string; score: number; use_reasoning?: boolean }> | Record<string, number> | undefined
): NormalizedModelScore[] => {
  if (!modelScores) return []
  // Already an array
  if (Array.isArray(modelScores)) return modelScores
  // Object format (Legacy) - convert to array
  return Object.entries(modelScores).map(([model, score]) => ({
    model,
    score: typeof score === 'number' ? score : 0,
    use_reasoning: false,
  }))
}

// Inner component that uses useReactFlow (must be inside ReactFlowProvider)
const TopologyFlow: React.FC = () => {
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [nodes, setNodes, onNodesChange] = useNodesState([])
  const [edges, setEdges, onEdgesChange] = useEdgesState([])
  const { fitView } = useReactFlow()

  useEffect(() => {
    fetchConfig()
  }, [])

  const fetchConfig = async () => {
    try {
      setLoading(true)
      // Try the dashboard backend endpoint first
      const response = await fetch('/api/router/config/all')
      if (!response.ok) {
        throw new Error(`Failed to fetch config: ${response.statusText}`)
      }
      const data = await response.json()
      generateTopology(data)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load configuration')
      console.error('Error fetching config:', err)
    } finally {
      setLoading(false)
    }
  }

  const generateTopology = (configData: ConfigData) => {
    const newNodes: Node[] = []
    const newEdges: Edge[] = []

    // Layout parameters
    const nodeWidth = 220 // Fixed node width to prevent text overflow
    const horizontalSpacing = 150 // Spacing between nodes (from right edge to left edge)
    const verticalSpacing = 100

    // Unified node style with fixed width
    const nodeStyle = {
      padding: '14px 20px',
      borderRadius: '8px',
      minHeight: '80px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      textAlign: 'center' as const,
      width: `${nodeWidth}px`,
      boxSizing: 'border-box' as const,
    }

    let currentX = 50 // Starting position from the left
    const baseY = 300 // Unified Y coordinate to keep nodes on the same horizontal line

    // 1. User Query Node (starting point)
    newNodes.push({
      id: 'user-query',
      type: 'input',
      data: {
        label: (
          <div style={{ textAlign: 'center', whiteSpace: 'nowrap' }}>
            <div style={{ fontWeight: 'bold', fontSize: '14px' }}>👤 User Query</div>
          </div>
        )
      },
      position: { x: currentX, y: baseY },
      sourcePosition: Position.Right,
      style: {
        ...nodeStyle,
        background: '#4CAF50',
        color: 'white',
        border: '2px solid #45a049',
        fontWeight: 'bold',
      },
    })

    currentX += nodeWidth + horizontalSpacing

    // 2. Prompt Guard (Jailbreak Detection)
    const promptGuardEnabled = configData.prompt_guard?.enabled ?? false
    const promptGuardModel = 'vLLM-SR-Jailbreak'
    newNodes.push({
      id: 'prompt-guard',
      data: {
        label: (
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontWeight: 'bold', fontSize: '14px', marginBottom: '6px', whiteSpace: 'nowrap' }}>🛡️ Prompt Guard</div>
            <div style={{
              fontSize: '11px',
              marginTop: '4px',
              background: 'rgba(255, 255, 255, 0.2)',
              padding: '4px 8px',
              borderRadius: '4px',
              fontWeight: '600',
              whiteSpace: 'nowrap',
            }}>
              {promptGuardEnabled ? `✓ ${promptGuardModel}` : '✗ Disabled'}
            </div>
          </div>
        ),
      },
      position: { x: currentX, y: baseY },
      sourcePosition: Position.Right,
      targetPosition: Position.Left,
      style: {
        ...nodeStyle,
        background: promptGuardEnabled ? '#FF9800' : '#757575',
        color: 'white',
        border: `2px solid ${promptGuardEnabled ? '#F57C00' : '#616161'}`,
      },
    })
    newEdges.push({
      id: 'e-query-guard',
      source: 'user-query',
      target: 'prompt-guard',
      animated: true,
      style: { stroke: promptGuardEnabled ? '#FF9800' : '#999', strokeWidth: 2 },
      markerEnd: { type: MarkerType.ArrowClosed, color: promptGuardEnabled ? '#FF9800' : '#999' },
    })

    currentX += nodeWidth + horizontalSpacing

    // 3. PII Detection
    const piiEnabled = configData.classifier?.pii_model?.model_id ? true : false
    const piiModel = 'vLLM-SR-PII'
    newNodes.push({
      id: 'pii-detection',
      data: {
        label: (
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontWeight: 'bold', fontSize: '14px', marginBottom: '6px', whiteSpace: 'nowrap' }}>🔒 PII Detection</div>
            <div style={{
              fontSize: '11px',
              marginTop: '4px',
              background: 'rgba(255, 255, 255, 0.2)',
              padding: '4px 8px',
              borderRadius: '4px',
              fontWeight: '600',
              whiteSpace: 'nowrap',
            }}>
              {piiEnabled ? `✓ ${piiModel}` : '✗ Disabled'}
            </div>
          </div>
        ),
      },
      position: { x: currentX, y: baseY },
      sourcePosition: Position.Right,
      targetPosition: Position.Left,
      style: {
        ...nodeStyle,
        background: piiEnabled ? '#9C27B0' : '#757575',
        color: 'white',
        border: `2px solid ${piiEnabled ? '#7B1FA2' : '#616161'}`,
      },
    })
    newEdges.push({
      id: 'e-guard-pii',
      source: 'prompt-guard',
      target: 'pii-detection',
      animated: true,
      style: { stroke: piiEnabled ? '#9C27B0' : '#999', strokeWidth: 2 },
      markerEnd: { type: MarkerType.ArrowClosed, color: piiEnabled ? '#9C27B0' : '#999' },
    })

    currentX += nodeWidth + horizontalSpacing

    // 4. Semantic Cache
    const cacheEnabled = configData.semantic_cache?.enabled ?? false
    const cacheType = configData.semantic_cache?.backend_type || 'memory'
    const cacheThreshold = configData.semantic_cache?.similarity_threshold || 0.8
    const cacheBertModel = 'vLLM-SR-Similarity'
    newNodes.push({
      id: 'semantic-cache',
      data: {
        label: (
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontWeight: 'bold', fontSize: '14px', marginBottom: '6px', whiteSpace: 'nowrap' }}>⚡ Semantic Cache</div>
            <div style={{
              fontSize: '11px',
              marginTop: '4px',
              background: 'rgba(255, 255, 255, 0.2)',
              padding: '4px 8px',
              borderRadius: '4px',
              fontWeight: '600',
              whiteSpace: 'nowrap',
            }}>
              {cacheEnabled ? `✓ ${cacheBertModel}` : '✗ Disabled'}
            </div>
            <div style={{ fontSize: '10px', marginTop: '4px', opacity: 0.85, whiteSpace: 'nowrap' }}>
              {cacheEnabled ? `${cacheType} (${cacheThreshold})` : ''}
            </div>
          </div>
        ),
      },
      position: { x: currentX, y: baseY },
      sourcePosition: Position.Right,
      targetPosition: Position.Left,
      style: {
        ...nodeStyle,
        background: cacheEnabled ? '#00BCD4' : '#757575',
        color: 'white',
        border: `2px solid ${cacheEnabled ? '#0097A7' : '#616161'}`,
      },
    })
    newEdges.push({
      id: 'e-pii-cache',
      source: 'pii-detection',
      target: 'semantic-cache',
      animated: true,
      style: { stroke: cacheEnabled ? '#00BCD4' : '#999', strokeWidth: 2 },
      markerEnd: { type: MarkerType.ArrowClosed, color: cacheEnabled ? '#00BCD4' : '#999' },
    })

    currentX += nodeWidth + horizontalSpacing

    // 5. Classification Hub
    const classificationModel = 'vLLM-SR-Classify'
    const classificationThreshold = configData.classifier?.category_model?.threshold || 0.6

    newNodes.push({
      id: 'classification',
      data: {
        label: (
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontWeight: 'bold', fontSize: '14px', marginBottom: '6px', whiteSpace: 'nowrap' }}>🧠 Classification</div>
            <div style={{
              fontSize: '11px',
              marginTop: '4px',
              background: 'rgba(255, 255, 255, 0.2)',
              padding: '4px 8px',
              borderRadius: '4px',
              fontWeight: '600',
              whiteSpace: 'nowrap',
            }}>
              ✓ {classificationModel}
            </div>
            <div style={{ fontSize: '10px', marginTop: '4px', opacity: 0.85, whiteSpace: 'nowrap' }}>
              threshold: {classificationThreshold}
            </div>
          </div>
        )
      },
      position: { x: currentX, y: baseY },
      sourcePosition: Position.Right,
      targetPosition: Position.Left,
      style: {
        ...nodeStyle,
        minHeight: '90px',
        background: '#673AB7',
        color: 'white',
        border: '2px solid #512DA8',
        fontWeight: 'bold',
      },
    })
    newEdges.push({
      id: 'e-cache-classification',
      source: 'semantic-cache',
      target: 'classification',
      animated: true,
      style: { stroke: '#673AB7', strokeWidth: 2 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#673AB7' },
    })

    // Detect if this is Python CLI format (has decisions) or Legacy format (has categories)
    const isPythonCLI = !!(configData.decisions && configData.decisions.length > 0)
    const categories = configData.categories || []
    const decisions = configData.decisions || []

    currentX += nodeWidth + horizontalSpacing
    const categoryX = currentX
    const modelX = categoryX + nodeWidth + horizontalSpacing

    if (isPythonCLI) {
      // Python CLI format: Show decisions as routing rules
      const totalDecisionsHeight = decisions.length * (verticalSpacing + 20)
      let decisionY = baseY - (totalDecisionsHeight / 2) + 50

      decisions.forEach((decision) => {
        const decisionId = `decision-${decision.name.replace(/[^a-zA-Z0-9]/g, '-')}`
        const hasReasoning = decision.modelRefs?.some(m => m.use_reasoning) || false
        const conditionCount = decision.rules?.conditions?.length || 0
        const operator = decision.rules?.operator || 'AND'

        newNodes.push({
          id: decisionId,
          data: {
            label: (
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontWeight: 'bold', fontSize: '11px', whiteSpace: 'nowrap' }}>🔀 {decision.name}</div>
                <div style={{ fontSize: '9px', marginTop: '2px', opacity: 0.9 }}>
                  Priority: {decision.priority || 0}
                </div>
                <div style={{ fontSize: '9px', marginTop: '2px', opacity: 0.8 }}>
                  {conditionCount} rules ({operator})
                </div>
              </div>
            )
          },
          position: { x: categoryX, y: decisionY },
          sourcePosition: Position.Right,
          targetPosition: Position.Left,
          style: {
            background: hasReasoning ? '#E91E63' : '#3F51B5',
            color: 'white',
            border: `2px solid ${hasReasoning ? '#C2185B' : '#303F9F'}`,
            fontSize: '12px',
            padding: '10px 16px',
            borderRadius: '6px',
            minWidth: '160px',
            minHeight: '70px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          },
        })

        newEdges.push({
          id: `e-classification-${decision.name}`,
          source: 'classification',
          target: decisionId,
          style: { stroke: hasReasoning ? '#E91E63' : '#3F51B5', strokeWidth: 1.5 },
          markerEnd: { type: MarkerType.ArrowClosed, color: hasReasoning ? '#E91E63' : '#3F51B5' },
          label: `P${decision.priority || 0}`,
          labelStyle: { fontSize: '10px', fill: '#666' },
          labelBgStyle: { fill: 'white', fillOpacity: 0.9 },
        })

        // Models for each decision
        const modelRefs = decision.modelRefs || []
        modelRefs.forEach((modelRef, modelIndex) => {
          const modelId = `model-${decision.name}-${modelRef.model.replace(/[^a-zA-Z0-9]/g, '-')}`
          const modelYPos = decisionY + (modelIndex * 50) - ((modelRefs.length - 1) * 25)
          const hasReasoningModel = modelRef.use_reasoning || false
          const modelName = modelRef.model.split('/').pop() || modelRef.model

          // Find reasoning family from providers.models
          const modelConfig = configData.providers?.models?.find(m => m.name === modelRef.model)
          const reasoningFamily = modelConfig?.reasoning_family

          newNodes.push({
            id: modelId,
            type: 'output',
            data: {
              label: (
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontWeight: 'bold', fontSize: '11px' }}>
                    🤖 {modelName}
                  </div>
                  <div style={{ fontSize: '9px', marginTop: '2px' }}>
                    {hasReasoningModel ? '⚡ Reasoning' : '📝 Standard'}
                  </div>
                  {reasoningFamily && (
                    <div style={{ fontSize: '8px', marginTop: '1px', opacity: 0.8 }}>
                      {reasoningFamily}
                    </div>
                  )}
                </div>
              ),
            },
            position: { x: modelX, y: modelYPos },
            targetPosition: Position.Left,
            style: {
              background: hasReasoningModel ? '#E91E63' : '#607D8B',
              color: 'white',
              border: `2px solid ${hasReasoningModel ? '#C2185B' : '#455A64'}`,
              fontSize: '11px',
              padding: '8px 12px',
              borderRadius: '6px',
              minWidth: '140px',
            },
          })

          newEdges.push({
            id: `e-${decisionId}-${modelId}`,
            source: decisionId,
            target: modelId,
            animated: hasReasoningModel,
            style: {
              stroke: hasReasoningModel ? '#E91E63' : '#607D8B',
              strokeWidth: hasReasoningModel ? 3 : 2,
              strokeDasharray: hasReasoningModel ? '0' : '5, 5',
            },
            markerEnd: {
              type: MarkerType.ArrowClosed,
              color: hasReasoningModel ? '#E91E63' : '#607D8B',
              width: hasReasoningModel ? 25 : 20,
              height: hasReasoningModel ? 25 : 20,
            },
            label: hasReasoningModel ? '🧠' : '',
            labelStyle: { fontSize: '11px' },
            labelBgStyle: { fill: 'white', fillOpacity: 0.9 },
          })
        })

        decisionY += verticalSpacing + 20
      })
    } else {
      // Legacy format: Show categories with model_scores
      // Calculate total height for center alignment
      const totalCategoriesHeight = categories.length * verticalSpacing
      let categoryY = baseY - (totalCategoriesHeight / 2) + 50

      categories.forEach((category) => {
        const categoryId = `category-${category.name}`
        const hasSystemPrompt = category.system_prompt && category.system_prompt.length > 0

        newNodes.push({
          id: categoryId,
          data: {
            label: (
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontWeight: 'bold', fontSize: '12px' }}>📁 {category.name}</div>
                <div style={{ fontSize: '10px', marginTop: '3px', opacity: 0.9 }}>
                  {hasSystemPrompt ? '✓ System Prompt' : '✗ No Prompt'}
                </div>
              </div>
            )
          },
          position: { x: categoryX, y: categoryY },
          sourcePosition: Position.Right,
          targetPosition: Position.Left,
          style: {
            background: '#3F51B5',
            color: 'white',
            border: '2px solid #303F9F',
            fontSize: '12px',
            padding: '10px 16px',
            borderRadius: '6px',
            minWidth: '140px',
            minHeight: '60px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          },
        })

        newEdges.push({
          id: `e-classification-${category.name}`,
          source: 'classification',
          target: categoryId,
          style: { stroke: '#3F51B5', strokeWidth: 1.5 },
          markerEnd: { type: MarkerType.ArrowClosed, color: '#3F51B5' },
        })

        // Models for each category
        const modelScores = normalizeModelScores(category.model_scores)
        modelScores.forEach((modelScore, modelIndex) => {
          const modelId = `model-${category.name}-${modelScore.model.replace(/[^a-zA-Z0-9]/g, '-')}`
          const modelYPos = categoryY + (modelIndex * 50) - ((modelScores.length - 1) * 25)

          const reasoningFamily = configData.model_config?.[modelScore.model]?.reasoning_family
          const hasReasoning = modelScore.use_reasoning && reasoningFamily
          const modelName = modelScore.model.split('/').pop() || modelScore.model

          newNodes.push({
            id: modelId,
            type: 'output',
            data: {
              label: (
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontWeight: 'bold', fontSize: '11px' }}>
                    🤖 {modelName}
                  </div>
                  <div style={{ fontSize: '9px', marginTop: '2px' }}>
                    Score: {modelScore.score.toFixed(2)}
                  </div>
                </div>
              ),
            },
            position: { x: modelX, y: modelYPos },
            targetPosition: Position.Left,
            style: {
              background: '#607D8B',
              color: 'white',
              border: '2px solid #455A64',
              fontSize: '11px',
              padding: '8px 12px',
              borderRadius: '6px',
              minWidth: '140px',
            },
          })

          // Use different line styles to indicate reasoning enabled
          newEdges.push({
            id: `e-${categoryId}-${modelId}`,
            source: categoryId,
            target: modelId,
            animated: !!hasReasoning,
            style: {
              stroke: hasReasoning ? '#E91E63' : '#607D8B',
              strokeWidth: hasReasoning ? 3 : 2,
              strokeDasharray: hasReasoning ? '0' : '5, 5',
            },
            markerEnd: {
              type: MarkerType.ArrowClosed,
              color: hasReasoning ? '#E91E63' : '#607D8B',
              width: hasReasoning ? 25 : 20,
              height: hasReasoning ? 25 : 20,
            },
            label: `${(modelScore.score * 100).toFixed(0)}%${hasReasoning ? ' 🧠' : ''}`,
            labelStyle: {
              fontSize: '11px',
              fill: hasReasoning ? '#E91E63' : '#666',
              fontWeight: hasReasoning ? 'bold' : 'normal',
            },
            labelBgStyle: { fill: 'white', fillOpacity: 0.9 },
          })
        })

        categoryY += verticalSpacing
      })
    }

    setNodes(newNodes)
    setEdges(newEdges)
  }

  // Call fitView after nodes are updated
  useEffect(() => {
    if (nodes.length > 0) {
      // Small delay to ensure React Flow has rendered the nodes
      const timer = setTimeout(() => {
        fitView({ padding: 0.2, duration: 300 })
      }, 100)
      return () => clearTimeout(timer)
    }
  }, [nodes.length, fitView])

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
          <button onClick={fetchConfig} className={styles.retryButton}>
            Retry
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <h1 className={styles.title}>🗺️ Semantic Router Topology</h1>
        <p className={styles.subtitle}>
          Visualize the chain-of-thought flow from user query to model selection
        </p>
        <button onClick={fetchConfig} className={styles.refreshButton}>
          🔄 Refresh
        </button>
      </div>
      <div className={styles.flowContainer}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          fitView
          fitViewOptions={{
            padding: 0.3,
            minZoom: 0.5,
            maxZoom: 1.5,
          }}
          defaultViewport={{ x: 0, y: 0, zoom: 0.7 }}
          attributionPosition="bottom-left"
          style={{ width: '100%', height: '100%' }}
        >
          <Background />
          <Controls />
          <MiniMap
            nodeColor={(node) => {
              const style = node.style as any
              return style?.background || '#ccc'
            }}
            maskColor="rgba(0, 0, 0, 0.1)"
          />
        </ReactFlow>
      </div>
      <div className={styles.legend}>
        <h3>Legend</h3>
        <div className={styles.legendItems}>
          <div className={styles.legendItem}>
            <span className={styles.legendColor} style={{ background: '#4CAF50' }}></span>
            <span>User Input</span>
          </div>
          <div className={styles.legendItem}>
            <span className={styles.legendColor} style={{ background: '#FF9800' }}></span>
            <span>Security</span>
          </div>
          <div className={styles.legendItem}>
            <span className={styles.legendColor} style={{ background: '#00BCD4' }}></span>
            <span>Cache</span>
          </div>
          <div className={styles.legendItem}>
            <span className={styles.legendColor} style={{ background: '#673AB7' }}></span>
            <span>Classification</span>
          </div>
          <div className={styles.legendItem}>
            <span className={styles.legendColor} style={{ background: '#3F51B5' }}></span>
            <span>Category/Decision</span>
          </div>
          <div className={styles.legendItem}>
            <span className={styles.legendColor} style={{ background: '#607D8B' }}></span>
            <span>Model</span>
          </div>
          <div className={styles.legendItem}>
            <div style={{
              width: '30px',
              height: '3px',
              background: '#E91E63',
              borderRadius: '2px',
              marginRight: '8px'
            }}></div>
            <span>Reasoning (solid)</span>
          </div>
          <div className={styles.legendItem}>
            <div style={{
              width: '30px',
              height: '2px',
              background: '#607D8B',
              borderRadius: '2px',
              marginRight: '8px',
              backgroundImage: 'repeating-linear-gradient(90deg, #607D8B 0, #607D8B 5px, transparent 5px, transparent 10px)'
            }}></div>
            <span>Standard (dashed)</span>
          </div>
        </div>
      </div>
    </div>
  )
}

// Wrapper component that provides ReactFlow context
const TopologyPage: React.FC = () => {
  return (
    <ReactFlowProvider>
      <TopologyFlow />
    </ReactFlowProvider>
  )
}

export default TopologyPage

