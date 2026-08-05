import React, { useState, useCallback, useMemo, useRef, useEffect } from 'react'
import ReactFlow, {
  useReactFlow,
  useNodesState,
  useEdgesState,
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  ConnectionLineType,
  type Node,
} from 'reactflow'
import 'reactflow/dist/style.css'
import styles from './ExpressionBuilder.module.css'
import {
  DRAG_MIME,
  addChildAtPath,
  decodeDrag,
  getNodeAtPath,
  insertAtPath,
  isLeaf,
  isOperator,
  makeDragNode,
  parseExprText,
  removeAtPath,
  replaceAtPath,
  serializeNode,
  validateTree,
  type DragData,
  type NodePath,
  type RuleNode,
  type SignalDescriptor,
} from './ExpressionBuilderSupport'
import { applyDagreLayout, treeToFlowElements, type FlowNodeData } from './ExpressionBuilderFlow'
import ExpressionBuilderCanvasEmptyState from './ExpressionBuilderCanvasEmptyState'
import ExpressionBuilderContextMenu from './ExpressionBuilderContextMenu'
import { AddChildPicker, EditSignalDialog } from './ExpressionBuilderDialogs'
import { type BuilderTemplate, nodeTypes } from './ExpressionBuilderNodes'
import ExpressionBuilderToolbox from './ExpressionBuilderToolbox'

export interface ExpressionBuilderInnerProps {
  tree: RuleNode | null
  setTree: React.Dispatch<React.SetStateAction<RuleNode | null>>
  rawText: string
  setRawText: React.Dispatch<React.SetStateAction<string>>
  isRawMode: boolean
  setIsRawMode: React.Dispatch<React.SetStateAction<boolean>>
  maximized: boolean
  setMaximized: React.Dispatch<React.SetStateAction<boolean>>
  availableSignals: SignalDescriptor[]
  onChange: (expr: string) => void
  pushHistory: (prev: RuleNode | null) => void
  canUndo: boolean
  canRedo: boolean
  handleUndo: () => void
  handleRedo: () => void
  selectedPath: NodePath | null
  setSelectedPath: React.Dispatch<React.SetStateAction<NodePath | null>>
  internalChangeRef: React.MutableRefObject<boolean>
}

/**
 * Canvas + toolbox + raw/validation UI. Must render under `ReactFlowProvider`.
 */
const ExpressionBuilderInner: React.FC<ExpressionBuilderInnerProps> = ({
  tree, setTree, rawText, setRawText, isRawMode, setIsRawMode,
  maximized, setMaximized, availableSignals, onChange, pushHistory,
  canUndo, canRedo, handleUndo, handleRedo, selectedPath, setSelectedPath,
  internalChangeRef,
}) => {
  const { fitView } = useReactFlow()
  const [nodes, setNodes, onNodesChange] = useNodesState<FlowNodeData>([])
  const [edges, setEdges, onEdgesChange] = useEdgesState([])
  const [signalSearch, setSignalSearch] = useState('')
  const [toolboxCollapsed, setToolboxCollapsed] = useState(false)
  const [collapsedGroups, setCollapsedGroups] = useState<Set<string>>(() => {
    const keys = new Set<string>()
    for (const s of availableSignals) keys.add(s.signalType.toUpperCase())
    return keys
  })
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; path: NodePath } | null>(null)
  const [editingNode, setEditingNode] = useState<{ path: NodePath; signalType: string; signalName: string } | null>(null)
  const [addingToPath, setAddingToPath] = useState<NodePath | null>(null)
  const [toast, setToast] = useState<string | null>(null)
  const toastTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const [insertSiblingTarget, setInsertSiblingTarget] = useState<{ parentPath: NodePath; index: number } | null>(null)

  const reactFlowRef = useRef<HTMLDivElement>(null)
  const suppressSyncRef = useRef(false)

  const showToast = useCallback((msg: string) => {
    if (toastTimerRef.current) clearTimeout(toastTimerRef.current)
    setToast(msg)
    toastTimerRef.current = setTimeout(() => setToast(null), 2000)
  }, [])

  const handleInsertSiblingPick = useCallback((newNode: RuleNode) => {
    if (!insertSiblingTarget || !tree) { setInsertSiblingTarget(null); return }
    pushHistory(tree)
    setTree(insertAtPath(tree, insertSiblingTarget.parentPath, insertSiblingTarget.index, newNode))
    const label = isLeaf(newNode) ? `${newNode.signalType}("${newNode.signalName}")` : (newNode as Exclude<RuleNode, { signalType: string }>).operator
    showToast(`Inserted ${label}`)
    setInsertSiblingTarget(null)
  }, [insertSiblingTarget, tree, pushHistory, setTree, showToast])

  useEffect(() => {
    if (suppressSyncRef.current) { suppressSyncRef.current = false; return }
    const text = tree ? serializeNode(tree) : ''
    setRawText(text)
    internalChangeRef.current = true
    onChange(text)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tree])

  useEffect(() => {
    if (!contextMenu) return
    const handler = () => setContextMenu(null)
    window.addEventListener('click', handler)
    return () => window.removeEventListener('click', handler)
  }, [contextMenu])

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && maximized) setMaximized(false)
      const tag = (e.target as HTMLElement)?.tagName?.toLowerCase()
      const inInput = tag === 'input' || tag === 'textarea' || tag === 'select' || (e.target as HTMLElement)?.isContentEditable
      if (inInput) return
      if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) { e.preventDefault(); handleUndo() }
      if ((e.ctrlKey || e.metaKey) && (e.key === 'y' || (e.key === 'z' && e.shiftKey))) { e.preventDefault(); handleRedo() }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [maximized, setMaximized, handleUndo, handleRedo])

  const signalGroups = useMemo(() => {
    const groups: Record<string, SignalDescriptor[]> = {}
    for (const s of availableSignals) {
      const key = s.signalType.toUpperCase()
      if (!groups[key]) groups[key] = []
      groups[key].push(s)
    }
    return Object.entries(groups).sort((a, b) => a[0].localeCompare(b[0]))
  }, [availableSignals])

  const filteredGroups = useMemo(() => {
    if (!signalSearch.trim()) return signalGroups
    const q = signalSearch.toLowerCase()
    return signalGroups
      .map(([type, signals]) => [type, signals.filter(s => s.name.toLowerCase().includes(q) || s.signalType.toLowerCase().includes(q))] as [string, SignalDescriptor[]])
      .filter(([, signals]) => signals.length > 0)
  }, [signalGroups, signalSearch])

  const toggleGroup = useCallback((group: string) => {
    setCollapsedGroups(prev => { const n = new Set(prev); if (n.has(group)) n.delete(group); else n.add(group); return n })
  }, [])

  const handleClear = useCallback(() => { pushHistory(tree); setTree(null); setSelectedPath(null); showToast('Expression cleared') }, [tree, pushHistory, setTree, setSelectedPath, showToast])

  const handleDeleteNode = useCallback((path: NodePath) => {
    if (!tree) return
    const node = getNodeAtPath(tree, path)
    const label = node ? (isLeaf(node) ? `${node.signalType}("${node.signalName}")` : node.operator) : 'node'
    pushHistory(tree)
    setTree(prev => prev ? removeAtPath(prev, path) : null)
    setSelectedPath(null)
    showToast(`Deleted ${label}`)
  }, [tree, pushHistory, setTree, setSelectedPath, showToast])

  const handleNodeDoubleClick = useCallback((path: NodePath) => {
    if (!tree) return
    const node = getNodeAtPath(tree, path)
    if (!node) return
    if (isLeaf(node)) {
      setEditingNode({ path, signalType: node.signalType, signalName: node.signalName })
    }
  }, [tree])

  const handleEditSave = useCallback((signalType: string, signalName: string) => {
    if (!editingNode || !tree) { setEditingNode(null); return }
    pushHistory(tree)
    setTree(replaceAtPath(tree, editingNode.path, { signalType, signalName }))
    setEditingNode(null)
    showToast(`Updated to ${signalType}("${signalName}")`)
  }, [editingNode, tree, pushHistory, setTree, showToast])

  const handleDropOnNode = useCallback((targetPath: NodePath, dragData: DragData) => {
    if (!tree) return
    const targetNode = getNodeAtPath(tree, targetPath)
    if (!targetNode) return
    if (isLeaf(targetNode)) return
    const newNode = makeDragNode(dragData)
    if (!newNode) return
    pushHistory(tree)
    setTree(prev => prev ? addChildAtPath(prev, targetPath, newNode) : newNode)
    showToast('Added to operator node')
  }, [tree, pushHistory, setTree, showToast])

  const handleAddChild = useCallback((targetPath: NodePath) => {
    setAddingToPath(targetPath)
  }, [])

  const handleAddChildPick = useCallback((newNode: RuleNode) => {
    if (!addingToPath || !tree) { setAddingToPath(null); return }
    pushHistory(tree)
    setTree(prev => prev ? addChildAtPath(prev, addingToPath, newNode) : newNode)
    const label = isLeaf(newNode) ? `${newNode.signalType}("${newNode.signalName}")` : (newNode as Exclude<RuleNode, { signalType: string }>).operator
    showToast(`Added ${label}`)
    setAddingToPath(null)
  }, [addingToPath, tree, pushHistory, setTree, showToast])

  useEffect(() => {
    if (!tree) {
      setNodes([])
      setEdges([])
      return
    }
    const { nodes: rawNodes, edges: newEdges } = treeToFlowElements(tree, handleNodeDoubleClick, handleDropOnNode, handleAddChild)
    const layoutNodes = applyDagreLayout(rawNodes, newEdges)
    setNodes(layoutNodes)
    setEdges(newEdges)
    setTimeout(() => fitView({ padding: 0.15, duration: 200 }), 50)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tree, handleNodeDoubleClick, handleDropOnNode, handleAddChild])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    const raw = e.dataTransfer.getData(DRAG_MIME)
    if (!raw) return
    const data = decodeDrag(raw)
    if (!data || data.kind === 'tree-node') return

    const newNode = makeDragNode(data)
    if (!newNode) return

    if (!tree) {
      pushHistory(tree)
      setTree(newNode)
      showToast('Created root node')
      return
    }

    if (data.kind === 'operator') {
      pushHistory(tree)
      if (data.operator === 'NOT') {
        setTree({ operator: 'NOT', conditions: [tree] })
      } else {
        setTree({ operator: data.operator, conditions: [tree] })
      }
      showToast(`Wrapped tree with ${data.operator}`)
      return
    }

    if (isOperator(tree) && (tree.operator === 'AND' || tree.operator === 'OR')) {
      pushHistory(tree)
      setTree({ ...tree, conditions: [...tree.conditions, newNode] })
      showToast(`Added to root ${tree.operator}`)
      return
    }

    showToast('Drop onto an operator node instead')
  }, [tree, pushHistory, setTree, showToast])

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'move'
  }, [])

  const onNodeClick = useCallback((_: React.MouseEvent, node: Node<FlowNodeData>) => {
    setSelectedPath(node.data.path)
  }, [setSelectedPath])

  const onNodeContextMenu = useCallback((e: React.MouseEvent, node: Node<FlowNodeData>) => {
    e.preventDefault()
    e.stopPropagation()
    setContextMenu({ x: e.clientX, y: e.clientY, path: node.data.path })
  }, [])

  const onPaneClick = useCallback(() => {
    setSelectedPath(null)
    setContextMenu(null)
  }, [setSelectedPath])

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName?.toLowerCase()
      if (tag === 'input' || tag === 'textarea' || tag === 'select' || (e.target as HTMLElement)?.isContentEditable) return
      if ((e.key === 'Delete' || e.key === 'Backspace') && selectedPath && selectedPath.length > 0) {
        e.preventDefault()
        handleDeleteNode(selectedPath)
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [selectedPath, handleDeleteNode])

  const handleWrap = useCallback((path: NodePath, op: 'AND' | 'OR' | 'NOT') => {
    if (!tree) return
    const node = getNodeAtPath(tree, path)
    if (!node) return
    pushHistory(tree)
    const wrapped: RuleNode = op === 'NOT'
      ? { operator: 'NOT', conditions: [node] }
      : { operator: op, conditions: [node] }
    setTree(replaceAtPath(tree, path, wrapped))
    setContextMenu(null)
    showToast(`Wrapped with ${op}`)
  }, [tree, pushHistory, setTree, showToast])

  const handleUnwrap = useCallback((path: NodePath) => {
    if (!tree) return
    const node = getNodeAtPath(tree, path)
    if (!node || !isOperator(node) || node.conditions.length === 0) return
    pushHistory(tree)
    setTree(replaceAtPath(tree, path, node.conditions[0]))
    setContextMenu(null)
    showToast('Unwrapped node')
  }, [tree, pushHistory, setTree, showToast])

  const handleChangeOp = useCallback((path: NodePath, newOp: 'AND' | 'OR') => {
    if (!tree) return
    const node = getNodeAtPath(tree, path)
    if (!node || !isOperator(node)) return
    pushHistory(tree)
    setTree(replaceAtPath(tree, path, { ...node, operator: newOp, conditions: node.conditions } as RuleNode))
    setContextMenu(null)
    showToast(`Changed to ${newOp}`)
  }, [tree, pushHistory, setTree, showToast])

  const handleRawChange = useCallback((text: string) => {
    setRawText(text)
    suppressSyncRef.current = true
    setTree(parseExprText(text))
    onChange(text)
  }, [onChange, setRawText, setTree])

  const validationIssues = useMemo(() => validateTree(tree, availableSignals), [tree, availableSignals])

  const applyTemplate = useCallback((tpl: BuilderTemplate) => {
    pushHistory(tree)
    setTree(tpl.build())
  }, [tree, pushHistory, setTree])

  return (
    <div className={`${styles.container} ${maximized ? styles.containerMaximized : ''}`}
      onClick={() => { setSelectedPath(null); setContextMenu(null) }}>
      <div className={styles.mainLayout}>
        <div className={styles.canvasWrapper}>
          <div className={styles.zoomBar}>
            <div className={styles.toolbarGroup}>
              <button type="button" className={styles.zoomBtn} onClick={handleUndo} disabled={!canUndo} title="Undo (Ctrl+Z)">↩</button>
              <button type="button" className={styles.zoomBtn} onClick={handleRedo} disabled={!canRedo} title="Redo (Ctrl+Y)">↪</button>
            </div>
            <div className={styles.toolbarSep} />
            <div className={styles.toolbarGroup}>
              <button type="button" className={styles.zoomBtn} onClick={() => fitView({ padding: 0.15, duration: 200 })} title="Fit to view">Fit</button>
            </div>
            {selectedPath && (
              <span className={styles.zoomHint}>
                Selected: {(() => { const n = tree && getNodeAtPath(tree, selectedPath); return n ? (isLeaf(n) ? `${n.signalType}("${n.signalName}")` : n.operator) : '—' })()}
              </span>
            )}
            <div className={styles.toolbarSpacer} />
            <button
              type="button"
              className={`${styles.zoomBtn} ${styles.maximizeBtn}`}
              onClick={(e) => { e.stopPropagation(); setMaximized(!maximized) }}
              title={maximized ? 'Exit fullscreen (Esc)' : 'Fullscreen'}
            >
              {maximized ? '⊗' : '⛶'}
            </button>
          </div>

          <div className={styles.rfCanvas}
            ref={reactFlowRef}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
          >
            {tree ? (
              <ReactFlow
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                nodeTypes={nodeTypes}
                onNodeClick={onNodeClick}
                onNodeContextMenu={onNodeContextMenu}
                onPaneClick={onPaneClick}
                fitView
                fitViewOptions={{ padding: 0.15 }}
                minZoom={0.05}
                maxZoom={4}
                connectionLineType={ConnectionLineType.Bezier}
                defaultEdgeOptions={{
                  type: 'default',
                  style: { strokeWidth: 2 },
                }}
                nodesDraggable={false}
                nodesConnectable={false}
                edgesFocusable={false}
                proOptions={{ hideAttribution: true }}
              >
                <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="rgba(143, 148, 156, 0.15)" />
                <Controls
                  showInteractive={false}
                  position="bottom-left"
                  className={styles.rfControls}
                />
                <MiniMap
                  nodeColor={(n) => n.type === 'operatorNode' ? 'rgba(99, 102, 241, 0.7)' : 'rgba(143, 148, 156, 0.7)'}
                  maskColor="rgba(0, 0, 0, 0.15)"
                  className={styles.rfMinimap}
                  pannable
                  zoomable
                  position="bottom-right"
                  style={{ width: 120, height: 80 }}
                />
              </ReactFlow>
            ) : (
              <ExpressionBuilderCanvasEmptyState onApplyTemplate={applyTemplate} />
            )}
          </div>
        </div>

        <ExpressionBuilderToolbox
          collapsedGroups={collapsedGroups}
          filteredGroups={filteredGroups}
          signalCount={availableSignals.length}
          signalSearch={signalSearch}
          toolboxCollapsed={toolboxCollapsed}
          onClear={handleClear}
          onSignalSearchChange={setSignalSearch}
          onToggleCollapsed={() => setToolboxCollapsed(!toolboxCollapsed)}
          onToggleGroup={toggleGroup}
        />
      </div>

      <div className={styles.result}>
        <span className={styles.resultLabel}>Result:</span>
        {tree ? <code className={styles.resultExpr}>{rawText}</code>
             : <span className={styles.resultPlaceholder}>No condition — route matches all requests</span>}
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <label className={styles.rawToggle} htmlFor="expr-raw-toggle">
          <input id="expr-raw-toggle" type="checkbox" checked={isRawMode} onChange={e => setIsRawMode(e.target.checked)} style={{ accentColor: 'var(--color-primary)' }} />
          Edit raw expression
        </label>
      </div>
      {isRawMode && (
        <input className={styles.rawInput} value={rawText} onChange={e => handleRawChange(e.target.value)}
          placeholder='e.g. domain("math") AND complexity("hard")' onClick={e => e.stopPropagation()} />
      )}

      {validationIssues.length > 0 && <div>{validationIssues.map((w, i) => <div key={i} className={styles.validationWarn}>⚠ {w}</div>)}</div>}
      {tree && validationIssues.length === 0 && <div className={styles.validationOk}>✓ All referenced signals exist</div>}

      {contextMenu && tree ? (
        <ExpressionBuilderContextMenu
          contextMenu={contextMenu}
          tree={tree}
          onAddChild={(path) => {
            setAddingToPath(path)
            setContextMenu(null)
          }}
          onChangeOp={handleChangeOp}
          onDeleteNode={(path) => {
            handleDeleteNode(path)
            setContextMenu(null)
          }}
          onEditSignal={(path, signalType, signalName) => {
            setEditingNode({ path, signalType, signalName })
            setContextMenu(null)
          }}
          onInsertSibling={(target) => {
            setInsertSiblingTarget(target)
            setContextMenu(null)
          }}
          onUnwrap={handleUnwrap}
          onWrap={handleWrap}
        />
      ) : null}

      {editingNode && (
        <EditSignalDialog
          signalType={editingNode.signalType}
          signalName={editingNode.signalName}
          availableSignals={availableSignals}
          onSave={handleEditSave}
          onCancel={() => setEditingNode(null)}
        />
      )}

      {addingToPath && (
        <AddChildPicker
          availableSignals={availableSignals}
          onPick={handleAddChildPick}
          onCancel={() => setAddingToPath(null)}
        />
      )}

      {insertSiblingTarget && (
        <AddChildPicker
          availableSignals={availableSignals}
          onPick={handleInsertSiblingPick}
          onCancel={() => setInsertSiblingTarget(null)}
        />
      )}

      {toast && <div className={styles.toast}>{toast}</div>}
    </div>
  )
}

export default ExpressionBuilderInner
