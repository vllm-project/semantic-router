import React, { useState, useCallback, useMemo, useRef, useEffect } from 'react'
import { createPortal } from 'react-dom'
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
  removeAtPath,
  newRuleNodeId,
  replaceAtPath,
  type DragData,
  type NodePath,
  type RuleIssue,
  type RuleLeaf,
  type RuleNode,
  type SignalDescriptor,
} from './ExpressionBuilderSupport'
import { formatRuleIssuePath } from '@/lib/decisionRuleIssues'
import { applyDagreLayout, treeToFlowElements, type FlowNodeData } from './ExpressionBuilderFlow'
import ExpressionBuilderCanvasEmptyState from './ExpressionBuilderCanvasEmptyState'
import ExpressionBuilderContextMenu from './ExpressionBuilderContextMenu'
import { AddChildPicker, EditSignalDialog } from './ExpressionBuilderDialogs'
import { type BuilderTemplate, nodeTypes } from './ExpressionBuilderNodes'
import ExpressionBuilderToolbox from './ExpressionBuilderToolbox'

export interface ExpressionBuilderInnerProps {
  tree: RuleNode | null
  /** Reports a tree change to the owning editor; the editor serializes it. */
  setTree: (action: React.SetStateAction<RuleNode | null>) => void
  /** DSL text of the current tree, for the Result line. */
  resultText: string
  /** The raw input's text: the draft while raw mode is on. */
  rawText: string
  onRawChange: (text: string) => void
  rawError: string | null
  isRawMode: boolean
  setIsRawMode: (on: boolean) => void
  rawDisabledReason: string | null
  maximized: boolean
  setMaximized: React.Dispatch<React.SetStateAction<boolean>>
  availableSignals: SignalDescriptor[]
  issues: RuleIssue[]
  pushHistory: (prev: RuleNode | null) => void
  canUndo: boolean
  canRedo: boolean
  handleUndo: () => void
  handleRedo: () => void
  selectedPath: NodePath | null
  setSelectedPath: React.Dispatch<React.SetStateAction<NodePath | null>>
  idPrefix: string
}

function isTextEntry(target: EventTarget | null): boolean {
  const element = target as HTMLElement | null
  const tag = element?.tagName?.toLowerCase()
  return tag === 'input' || tag === 'textarea' || tag === 'select' || Boolean(element?.isContentEditable)
}

/**
 * Canvas + toolbox + raw/validation UI. Must render under `ReactFlowProvider`.
 */
const ExpressionBuilderInner: React.FC<ExpressionBuilderInnerProps> = ({
  tree, setTree, resultText, rawText, onRawChange, rawError, isRawMode, setIsRawMode,
  rawDisabledReason, maximized, setMaximized, availableSignals, issues, pushHistory,
  canUndo, canRedo, handleUndo, handleRedo, selectedPath, setSelectedPath, idPrefix,
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
  const [editingPath, setEditingPath] = useState<NodePath | null>(null)
  const [addingToPath, setAddingToPath] = useState<NodePath | null>(null)
  const [toast, setToast] = useState<string | null>(null)
  const toastTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const [insertSiblingTarget, setInsertSiblingTarget] = useState<{ parentPath: NodePath; index: number } | null>(null)

  const reactFlowRef = useRef<HTMLDivElement>(null)

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
    if (!contextMenu) return
    const handler = () => setContextMenu(null)
    window.addEventListener('click', handler)
    return () => window.removeEventListener('click', handler)
  }, [contextMenu])

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
    if (node && isLeaf(node)) setEditingPath(path)
  }, [tree])

  const editingLeaf = useMemo(() => {
    if (!tree || !editingPath) return null
    const node = getNodeAtPath(tree, editingPath)
    return node && isLeaf(node) ? node : null
  }, [tree, editingPath])

  const handleEditSave = useCallback((edit: Partial<RuleLeaf>) => {
    if (!editingPath || !tree || !editingLeaf) { setEditingPath(null); return }
    pushHistory(tree)
    // Merge into the existing leaf so its id and unknown fields survive; an
    // edit field set to undefined clears that field.
    const merged: Record<string, unknown> = { ...editingLeaf, ...edit }
    for (const key of Object.keys(merged)) if (merged[key] === undefined) delete merged[key]
    const next = merged as unknown as RuleLeaf
    setTree(replaceAtPath(tree, editingPath, next))
    setEditingPath(null)
    showToast(`Updated to ${next.signalType}("${next.signalName}")`)
  }, [editingPath, editingLeaf, tree, pushHistory, setTree, showToast])

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

  const issueSeverity = useMemo(() => {
    const severity = new Map<string, RuleIssue['severity']>()
    for (const issue of issues) {
      if (!issue.nodeId || severity.get(issue.nodeId) === 'error') continue
      severity.set(issue.nodeId, issue.severity)
    }
    return severity
  }, [issues])

  useEffect(() => {
    if (!tree) {
      setNodes([])
      setEdges([])
      return
    }
    const { nodes: rawNodes, edges: newEdges } = treeToFlowElements(tree, handleNodeDoubleClick, handleDropOnNode, handleAddChild, issueSeverity)
    const layoutNodes = applyDagreLayout(rawNodes, newEdges)
    setNodes(layoutNodes)
    setEdges(newEdges)
    // The fullscreen switch remounts this canvas; a fit on the unmounted one yields NaN coordinates.
    const fitTimer = setTimeout(() => fitView({ padding: 0.15, duration: 200 }), 50)
    return () => clearTimeout(fitTimer)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tree, handleNodeDoubleClick, handleDropOnNode, handleAddChild, issueSeverity])

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
      setTree(data.operator === 'NOT'
        ? { id: newRuleNodeId(), operator: 'NOT', conditions: [tree] }
        : { id: newRuleNodeId(), operator: data.operator, conditions: [tree] })
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

  const onNodeMenuKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLDivElement>) => {
      if (event.key !== 'ContextMenu' && !(event.shiftKey && event.key === 'F10')) return
      const element = (event.target as HTMLElement).closest<HTMLElement>('.react-flow__node')
      const node = nodes.find((candidate) => candidate.id === element?.dataset.id)
      if (!element || !node) return
      event.preventDefault()
      event.stopPropagation()
      const bounds = element.getBoundingClientRect()
      setContextMenu({ x: bounds.left, y: bounds.bottom, path: node.data.path })
    },
    [nodes],
  )

  const onPaneClick = useCallback(() => {
    setSelectedPath(null)
    setContextMenu(null)
  }, [setSelectedPath])

  // Keyboard shortcuts are scoped to this editor's element, so two mounted
  // editors never act on each other's tree or history.
  const handleContainerKeyDown = useCallback((e: React.KeyboardEvent<HTMLDivElement>) => {
    // A dialog opened by this editor (it may be portaled) handles its own keys;
    // a dialog that hosts the editor, such as ConfigPage's form, does not.
    const dialog = (e.target as HTMLElement).closest?.('[role="dialog"]')
    if (dialog && !dialog.contains(e.currentTarget)) return
    // Enter in the editor's own inputs never submits a host form.
    if (e.key === 'Enter' && (e.target as HTMLElement).tagName?.toLowerCase() === 'input') e.preventDefault()
    if (isTextEntry(e.target)) return
    const modifier = e.ctrlKey || e.metaKey
    if (modifier && e.key === 'z' && !e.shiftKey) { e.preventDefault(); handleUndo(); return }
    if (modifier && (e.key === 'y' || (e.key === 'z' && e.shiftKey))) { e.preventDefault(); handleRedo(); return }
    if ((e.key === 'Delete' || e.key === 'Backspace') && selectedPath && selectedPath.length > 0) {
      e.preventDefault()
      handleDeleteNode(selectedPath)
    }
  }, [handleUndo, handleRedo, selectedPath, handleDeleteNode])

  const handleWrap = useCallback((path: NodePath, op: 'AND' | 'OR' | 'NOT') => {
    if (!tree) return
    const node = getNodeAtPath(tree, path)
    if (!node) return
    pushHistory(tree)
    const wrapped: RuleNode = op === 'NOT'
      ? { id: newRuleNodeId(), operator: 'NOT', conditions: [node] }
      : { id: newRuleNodeId(), operator: op, conditions: [node] }
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

  const rawToggleId = `${idPrefix}-raw-toggle`
  const rawErrorId = `${idPrefix}-raw-error`

  const applyTemplate = useCallback((tpl: BuilderTemplate) => {
    pushHistory(tree)
    setTree(tpl.build())
  }, [tree, pushHistory, setTree])

  return (
    <div className={`${styles.container} ${maximized ? styles.containerMaximized : ''}`}
      tabIndex={-1}
      data-rule-editor-container
      onKeyDown={handleContainerKeyDown}
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
            onKeyDown={onNodeMenuKeyDown}
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
                // d3-zoom stops dblclick propagation, which would swallow the
                // node's double-click-to-edit before React sees it.
                zoomOnDoubleClick={false}
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
        {resultText ? <code className={styles.resultExpr}>{resultText}</code>
             : <span className={styles.resultPlaceholder}>No condition — matches all requests</span>}
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <label className={styles.rawToggle} htmlFor={rawToggleId}>
          <input id={rawToggleId} type="checkbox" checked={isRawMode && !rawDisabledReason} disabled={Boolean(rawDisabledReason)}
            onChange={e => setIsRawMode(e.target.checked)} style={{ accentColor: 'var(--color-primary)' }} />
          Edit raw expression
        </label>
        {rawDisabledReason ? <span className={styles.resultPlaceholder}>{rawDisabledReason}</span> : null}
      </div>
      {isRawMode && !rawDisabledReason && (
        <>
          <input className={styles.rawInput} value={rawText} onChange={e => onRawChange(e.target.value)}
            aria-label="Raw expression" aria-invalid={rawError ? true : undefined}
            aria-describedby={rawError ? rawErrorId : undefined}
            placeholder='e.g. domain("math") AND complexity("hard")' onClick={e => e.stopPropagation()} />
          {rawError ? (
            <div id={rawErrorId} role="alert" className={styles.validationError}>
              ✕ {rawError}. Saving is blocked until the expression is complete; turning raw editing off restores the expression from before it.
            </div>
          ) : null}
        </>
      )}

      {issues.length > 0 && (
        <ul className={styles.issueList} aria-label="Rule issues">
          {issues.map((issue, i) => (
            <li key={`${issue.nodeId ?? 'root'}-${i}`} className={issue.severity === 'error' ? styles.validationError : styles.validationWarn}>
              {issue.severity === 'error' ? '✕' : '⚠'} {issue.path.length > 0 ? `${formatRuleIssuePath(issue.path)}: ` : ''}{issue.message}
            </li>
          ))}
        </ul>
      )}

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
          onEditSignal={(path) => {
            setEditingPath(path)
            setContextMenu(null)
          }}
          onInsertSibling={(target) => {
            setInsertSiblingTarget(target)
            setContextMenu(null)
          }}
          onClose={() => setContextMenu(null)}
          onUnwrap={handleUnwrap}
          onWrap={handleWrap}
        />
      ) : null}

      {/* Portaled so its form never nests inside a host form, such as
          ConfigPage's decision editor. */}
      {editingLeaf && createPortal(
        <EditSignalDialog
          initial={editingLeaf}
          availableSignals={availableSignals}
          onSave={handleEditSave}
          onCancel={() => setEditingPath(null)}
        />,
        document.body,
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
