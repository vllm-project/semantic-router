/**
 * ExpressionBuilder — ReactFlow-based drag-and-drop boolean expression editor (v4).
 *
 * Built on ReactFlow + Dagre for:
 *   - Infinite canvas with smooth zoom/pan (GPU-accelerated)
 *   - Automatic tree layout via dagre
 *   - MiniMap + background grid
 *   - Drag from toolbox to canvas
 *   - Node selection, context menu, undo/redo
 *   - Full backward compatibility with v3 props interface
 */

import React, { useState, useCallback, useMemo, useRef, useEffect, memo } from 'react'
import { createPortal } from 'react-dom'
import ReactFlow, {
  ReactFlowProvider,
  useReactFlow,
  useNodesState,
  useEdgesState,
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  Handle,
  Position,
  ConnectionLineType,
  MarkerType,
  type Node,
  type Edge,
  type NodeProps,
  type NodeTypes,
} from 'reactflow'
import 'reactflow/dist/style.css'
import Dagre from '@dagrejs/dagre'
import styles from './ExpressionBuilder.module.css'

// ─── Data Model ──────────────────────────────────────────────

export type RuleNode =
  | { operator: 'AND' | 'OR'; conditions: RuleNode[] }
  | { operator: 'NOT'; conditions: [RuleNode] }
  | { signalType: string; signalName: string }

function isLeaf(n: RuleNode): n is { signalType: string; signalName: string } {
  return 'signalType' in n
}

function isOperator(n: RuleNode): n is Exclude<RuleNode, { signalType: string }> {
  return 'operator' in n
}

// ─── Serialization: RuleNode → DSL text ──────────────────────

function serializeNode(n: RuleNode): string {
  if (isLeaf(n)) return `${n.signalType}("${n.signalName}")`
  if (n.operator === 'NOT') {
    const child = n.conditions[0]
    if (!child) return 'NOT (?)'
    const s = serializeNode(child)
    return isOperator(child) && child.operator !== 'NOT' ? `NOT (${s})` : `NOT ${s}`
  }
  if (n.conditions.length === 0) return `(? ${n.operator} ?)`
  if (n.conditions.length === 1) return serializeNode(n.conditions[0])
  const parts = n.conditions.map((c) => {
    const s = serializeNode(c)
    if (isOperator(c) && (c.operator === 'AND' || c.operator === 'OR') && c.operator !== n.operator) {
      return `(${s})`
    }
    return s
  })
  return parts.join(` ${n.operator} `)
}

// ─── Parsing: DSL text → RuleNode (best-effort) ─────────────

function parseExprText(text: string): RuleNode | null {
  const trimmed = text.trim()
  if (!trimmed) return null
  try {
    return parseOr(trimmed, { pos: 0 })
  } catch {
    return null
  }
}

interface ParseCtx { pos: number }

function skipWs(src: string, ctx: ParseCtx) {
  while (ctx.pos < src.length && src[ctx.pos] === ' ') ctx.pos++
}

function parseOr(src: string, ctx: ParseCtx): RuleNode {
  let left = parseAnd(src, ctx)
  while (true) {
    skipWs(src, ctx)
    if (src.slice(ctx.pos, ctx.pos + 2).toUpperCase() === 'OR' && /\s/.test(src[ctx.pos + 2] ?? '')) {
      ctx.pos += 2; skipWs(src, ctx)
      const right = parseAnd(src, ctx)
      if (isOperator(left) && left.operator === 'OR') {
        left = { operator: 'OR', conditions: [...left.conditions, right] }
      } else {
        left = { operator: 'OR', conditions: [left, right] }
      }
    } else break
  }
  return left
}

function parseAnd(src: string, ctx: ParseCtx): RuleNode {
  let left = parseNot(src, ctx)
  while (true) {
    skipWs(src, ctx)
    if (src.slice(ctx.pos, ctx.pos + 3).toUpperCase() === 'AND' && /[\s(]/.test(src[ctx.pos + 3] ?? '')) {
      ctx.pos += 3; skipWs(src, ctx)
      const right = parseNot(src, ctx)
      if (isOperator(left) && left.operator === 'AND') {
        left = { operator: 'AND', conditions: [...left.conditions, right] }
      } else {
        left = { operator: 'AND', conditions: [left, right] }
      }
    } else break
  }
  return left
}

function parseNot(src: string, ctx: ParseCtx): RuleNode {
  skipWs(src, ctx)
  if (src.slice(ctx.pos, ctx.pos + 3).toUpperCase() === 'NOT' && /[\s(]/.test(src[ctx.pos + 3] ?? '')) {
    ctx.pos += 3; skipWs(src, ctx)
    const child = parseNot(src, ctx)
    return { operator: 'NOT', conditions: [child] }
  }
  return parseAtom(src, ctx)
}

function parseAtom(src: string, ctx: ParseCtx): RuleNode {
  skipWs(src, ctx)
  if (src[ctx.pos] === '(') {
    ctx.pos++; skipWs(src, ctx)
    const inner = parseOr(src, ctx)
    skipWs(src, ctx)
    if (src[ctx.pos] === ')') ctx.pos++
    return inner
  }
  const m = src.slice(ctx.pos).match(/^(\w+)\("([^"]*)"\)/)
  if (m) { ctx.pos += m[0].length; return { signalType: m[1], signalName: m[2] } }
  const w = src.slice(ctx.pos).match(/^\w+/)
  if (w) { ctx.pos += w[0].length; return { signalType: w[0], signalName: '' } }
  throw new Error('Unexpected token')
}

// ─── BoolExprNode (AST) → RuleNode conversion ───────────────

export function boolExprToRuleNode(expr: Record<string, unknown> | null): RuleNode | null {
  if (!expr) return null
  const type = expr.type as string
  switch (type) {
    case 'signal_ref':
      return { signalType: expr.signalType as string, signalName: expr.signalName as string }
    case 'and': {
      const left = boolExprToRuleNode(expr.left as Record<string, unknown>)
      const right = boolExprToRuleNode(expr.right as Record<string, unknown>)
      if (!left || !right) return left || right
      return { operator: 'AND', conditions: [left, right] }
    }
    case 'or': {
      const left = boolExprToRuleNode(expr.left as Record<string, unknown>)
      const right = boolExprToRuleNode(expr.right as Record<string, unknown>)
      if (!left || !right) return left || right
      return { operator: 'OR', conditions: [left, right] }
    }
    case 'not': {
      const child = boolExprToRuleNode(expr.expr as Record<string, unknown>)
      if (!child) return null
      return { operator: 'NOT', conditions: [child] }
    }
    default: return null
  }
}

// ─── Immutable tree helpers ──────────────────────────────────

type NodePath = number[]

// eslint-disable-next-line @typescript-eslint/no-unused-vars
export function pathEq(a: NodePath, b: NodePath): boolean {
  return a.length === b.length && a.every((v, i) => v === b[i])
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars
export function pathStartsWith(path: NodePath, prefix: NodePath): boolean {
  if (prefix.length > path.length) return false
  return prefix.every((v, i) => v === path[i])
}

function getNodeAtPath(root: RuleNode, path: NodePath): RuleNode | null {
  if (path.length === 0) return root
  if (!isOperator(root)) return null
  const [head, ...tail] = path
  if (head < 0 || head >= root.conditions.length) return null
  return getNodeAtPath(root.conditions[head], tail)
}

function replaceAtPath(root: RuleNode, path: NodePath, replacement: RuleNode): RuleNode {
  if (path.length === 0) return replacement
  if (!isOperator(root)) return root
  const [head, ...tail] = path
  const newChild = replaceAtPath(root.conditions[head], tail, replacement)
  const newConditions = root.conditions.map((c, i) => (i === head ? newChild : c))
  return { ...root, conditions: newConditions } as RuleNode
}

function removeAtPath(root: RuleNode, path: NodePath): RuleNode | null {
  if (path.length === 0) return null
  if (!isOperator(root)) return root
  const [head, ...tail] = path
  if (tail.length === 0) {
    const newConditions = root.conditions.filter((_, i) => i !== head)
    if (newConditions.length === 0) return null
    if (root.operator !== 'NOT' && newConditions.length === 1) return newConditions[0]
    return { ...root, conditions: newConditions } as RuleNode
  }
  const newChild = removeAtPath(root.conditions[head], tail)
  if (!newChild) {
    const newConditions = root.conditions.filter((_, i) => i !== head)
    if (newConditions.length === 0) return null
    if (root.operator !== 'NOT' && newConditions.length === 1) return newConditions[0]
    return { ...root, conditions: newConditions } as RuleNode
  }
  const newConditions = root.conditions.map((c, i) => (i === head ? newChild : c))
  return { ...root, conditions: newConditions } as RuleNode
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars
export function insertAtPath(root: RuleNode, path: NodePath, insertIdx: number, node: RuleNode): RuleNode {
  if (path.length === 0) {
    if (isOperator(root)) {
      const conds = [...root.conditions]
      conds.splice(insertIdx, 0, node)
      return { ...root, conditions: conds } as RuleNode
    }
    return { operator: 'AND', conditions: [root, node] }
  }
  if (!isOperator(root)) return root
  const [head, ...tail] = path
  const newChild = insertAtPath(root.conditions[head], tail, insertIdx, node)
  const newConditions = root.conditions.map((c, i) => (i === head ? newChild : c))
  return { ...root, conditions: newConditions } as RuleNode
}

function addChildAtPath(root: RuleNode, path: NodePath, child: RuleNode): RuleNode {
  const target = getNodeAtPath(root, path)
  if (!target) return root
  if (isOperator(target)) {
    const newTarget = { ...target, conditions: [...target.conditions, child] } as RuleNode
    return replaceAtPath(root, path, newTarget)
  }
  const wrapped: RuleNode = { operator: 'AND', conditions: [target, child] }
  return replaceAtPath(root, path, wrapped)
}

// ─── Drag data ──────────────────────────────────────────────

interface DragDataSignal { kind: 'signal'; signalType: string; signalName: string }
interface DragDataOperator { kind: 'operator'; operator: 'AND' | 'OR' | 'NOT' }
interface DragDataTreeNode { kind: 'tree-node'; path: NodePath }
type DragData = DragDataSignal | DragDataOperator | DragDataTreeNode

const DRAG_MIME = 'application/x-expr-builder'
function encodeDrag(data: DragData): string { return JSON.stringify(data) }
function decodeDrag(raw: string): DragData | null {
  try { return JSON.parse(raw) as DragData } catch { return null }
}

function makeDragNode(data: DragData): RuleNode | null {
  if (data.kind === 'signal') return { signalType: data.signalType, signalName: data.signalName }
  if (data.kind === 'operator') {
    return data.operator === 'NOT'
      ? { operator: 'NOT', conditions: [] as unknown as [RuleNode] }
      : { operator: data.operator, conditions: [] }
  }
  return null
}

// ─── Validation ─────────────────────────────────────────────

function validateTree(node: RuleNode | null, signals: { signalType: string; name: string }[]): string[] {
  const w: string[] = []
  if (!node) return w
  if (isLeaf(node)) {
    if (!signals.some(s => s.signalType === node.signalType && s.name === node.signalName))
      w.push(`Signal ${node.signalType}("${node.signalName}") is not defined`)
  } else {
    if (node.operator === 'NOT' && node.conditions.length !== 1)
      w.push('NOT must have exactly one child')
    if ((node.operator === 'AND' || node.operator === 'OR') && node.conditions.length < 2)
      w.push(`${node.operator} needs at least 2 children`)
    for (const c of node.conditions) w.push(...validateTree(c, signals))
  }
  return w
}

// ═══════════════════════════════════════════════════════════════
// ReactFlow: RuleNode ↔ Nodes/Edges conversion + Dagre layout
// ═══════════════════════════════════════════════════════════════

const OPERATOR_W = 88
const OPERATOR_H = 78
const SIGNAL_W = 200
const SIGNAL_H = 40

interface FlowNodeData {
  ruleNode: RuleNode
  path: NodePath
  label: string
  isOperator: boolean
  depth?: number
  onDoubleClick?: (path: NodePath) => void
  onDropOnNode?: (targetPath: NodePath, data: DragData) => void
  onAddChild?: (targetPath: NodePath) => void
}

let _idCounter = 0
function nextId(): string { return `n${++_idCounter}` }

function treeToFlowElements(
  root: RuleNode,
  onDoubleClick?: (path: NodePath) => void,
  onDropOnNode?: (targetPath: NodePath, data: DragData) => void,
  onAddChild?: (targetPath: NodePath) => void,
): { nodes: Node<FlowNodeData>[]; edges: Edge[] } {
  _idCounter = 0
  const nodes: Node<FlowNodeData>[] = []
  const edges: Edge[] = []

  function walk(node: RuleNode, path: NodePath, parentId?: string, depth = 0) {
    const id = nextId()
    const isOp = isOperator(node)
    const label = isOp ? node.operator : `${node.signalType}("${node.signalName}")`

    nodes.push({
      id,
      type: isOp ? 'operatorNode' : 'signalNode',
      position: { x: 0, y: 0 },
      data: { ruleNode: node, path, label, isOperator: isOp, onDoubleClick, onDropOnNode, onAddChild, depth },
    })

    if (parentId) {
      edges.push({
        id: `e-${parentId}-${id}`,
        source: parentId,
        target: id,
        type: 'default',
        style: { stroke: 'rgba(118, 185, 0, 0.5)', strokeWidth: 2 },
        markerEnd: { type: MarkerType.ArrowClosed, color: 'rgba(118, 185, 0, 0.55)', width: 12, height: 12 },
      })
    }

    if (isOp) {
      const opNode = node as Exclude<RuleNode, { signalType: string }>
      opNode.conditions.forEach((child, idx) => {
        walk(child, [...path, idx], id, depth + 1)
      })
    }
  }

  walk(root, [])
  return { nodes, edges }
}

// Estimate signal node width based on text content
function estimateSignalWidth(node: RuleNode): number {
  if (!isLeaf(node)) return OPERATOR_W
  const text = `${node.signalType}  ${node.signalName}`
  // ~7px per character for 12px monospace font + 24px padding
  const estimated = Math.max(SIGNAL_W, text.length * 7 + 32)
  return Math.min(estimated, 320) // cap max width
}

function applyDagreLayout(nodes: Node<FlowNodeData>[], edges: Edge[]): Node<FlowNodeData>[] {
  const g = new Dagre.graphlib.Graph().setDefaultEdgeLabel(() => ({}))
  g.setGraph({
    rankdir: 'TB',
    nodesep: 52,
    ranksep: 68,
    marginx: 24,
    marginy: 24,
  })

  for (const node of nodes) {
    const w = node.data.isOperator ? OPERATOR_W : estimateSignalWidth(node.data.ruleNode)
    const h = node.data.isOperator ? OPERATOR_H : SIGNAL_H
    g.setNode(node.id, { width: w, height: h })
  }
  for (const edge of edges) {
    g.setEdge(edge.source, edge.target)
  }

  Dagre.layout(g)

  return nodes.map(node => {
    const { x, y } = g.node(node.id)
    const w = node.data.isOperator ? OPERATOR_W : estimateSignalWidth(node.data.ruleNode)
    const h = node.data.isOperator ? OPERATOR_H : SIGNAL_H
    return {
      ...node,
      position: { x: x - w / 2, y: y - h / 2 },
    }
  })
}

// ═══════════════════════════════════════════════════════════════
// Logic Gate SVG shapes — AND, OR, NOT (IEEE/ANSI style)
// ═══════════════════════════════════════════════════════════════

// Gate dimensions: 72w x 52h viewBox, drawn top-down (input top, output bottom)
// Rotated 90° from traditional left-to-right gates so they flow top→bottom

const GateAND: React.FC<{ color: string; opacity?: number }> = ({ color, opacity = 1 }) => (
  <svg viewBox="0 0 84 64" width="84" height="64" className={styles.gateSvg} style={{ opacity }}>
    {/* Body: flat top + rounded bottom (D-shape rotated) */}
    <path
      d="M 6 4 L 78 4 L 78 30 Q 78 60 42 60 Q 6 60 6 30 Z"
      fill={color.replace(/[\d.]+\)$/, '0.08)')}
      stroke={color}
      strokeWidth="2.5"
      strokeLinejoin="round"
    />
  </svg>
)

const GateOR: React.FC<{ color: string; opacity?: number }> = ({ color, opacity = 1 }) => (
  <svg viewBox="0 0 84 64" width="84" height="64" className={styles.gateSvg} style={{ opacity }}>
    {/* Body: curved top + pointed bottom */}
    <path
      d="M 6 4 Q 42 18 78 4 Q 74 44 42 62 Q 10 44 6 4 Z"
      fill={color.replace(/[\d.]+\)$/, '0.08)')}
      stroke={color}
      strokeWidth="2.5"
      strokeLinejoin="round"
    />
  </svg>
)

const GateNOT: React.FC<{ color: string; opacity?: number }> = ({ color, opacity = 1 }) => (
  <svg viewBox="0 0 84 70" width="84" height="70" className={styles.gateSvg} style={{ opacity }}>
    {/* Triangle body */}
    <path
      d="M 10 4 L 74 4 L 42 52 Z"
      fill={color.replace(/[\d.]+\)$/, '0.08)')}
      stroke={color}
      strokeWidth="2.5"
      strokeLinejoin="round"
    />
    {/* Inversion bubble */}
    <circle
      cx="42" cy="60" r="6"
      fill={color.replace(/[\d.]+\)$/, '0.08)')}
      stroke={color}
      strokeWidth="2.5"
    />
  </svg>
)

// ═══════════════════════════════════════════════════════════════
// Custom ReactFlow Nodes
// ═══════════════════════════════════════════════════════════════

const OperatorNodeComponent = memo<NodeProps<FlowNodeData>>(({ data, selected }) => {
  const [dragOver, setDragOver] = useState(false)
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    e.dataTransfer.dropEffect = 'copy'
    setDragOver(true)
  }, [])
  const handleDragLeave = useCallback(() => setDragOver(false), [])
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragOver(false)
    const raw = e.dataTransfer.getData(DRAG_MIME)
    if (!raw) return
    const dragData = decodeDrag(raw)
    if (dragData && data.onDropOnNode) data.onDropOnNode(data.path, dragData)
  }, [data])
  const handleDoubleClick = useCallback((e: React.MouseEvent) => {
    e.stopPropagation()
    if (data.onDoubleClick) data.onDoubleClick(data.path)
  }, [data])
  const handleAddClick = useCallback((e: React.MouseEvent) => {
    e.stopPropagation()
    if (data.onAddChild) data.onAddChild(data.path)
  }, [data])
  const opNode = data.ruleNode as Exclude<RuleNode, { signalType: string }>
  const childCount = (opNode.conditions as RuleNode[]).length
  const showAddBtn = opNode.operator !== 'NOT' || childCount === 0

  // Operator-based colors matching Toolbox buttons
  const gateColorMap: Record<string, string> = {
    AND: 'rgba(129, 140, 248, 0.85)',   // indigo
    OR:  'rgba(52, 211, 153, 0.85)',    // emerald
    NOT: 'rgba(248, 113, 113, 0.85)',   // rose
  }
  const gateColor = gateColorMap[opNode.operator] ?? 'rgba(129, 140, 248, 0.85)'

  const GateShape = opNode.operator === 'AND' ? GateAND
    : opNode.operator === 'OR' ? GateOR
    : GateNOT

  return (
    <div className={styles.rfOperatorWrapper}>
      <div
        className={`${styles.rfGateNode} ${selected ? styles.rfGateSelected : ''} ${dragOver ? styles.rfGateDragOver : ''}`}
        onDoubleClick={handleDoubleClick}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        title={`${data.label} (${childCount} children)\nRight-click for options`}
      >
        <Handle type="target" position={Position.Top} className={styles.rfHandle} />
        <GateShape color={gateColor} />
        <span className={`${styles.rfGateLabel} ${opNode.operator === 'NOT' ? styles.rfGateLabelNot : ''}`} style={{ color: gateColor }}>
          {data.label}
          {childCount > 0 && <span className={styles.rfGateBadge}>{childCount}</span>}
        </span>
        <Handle type="source" position={Position.Bottom} className={styles.rfHandle} />
      </div>
      {showAddBtn && (
        <div
          className={`${styles.rfAddBtn} ${dragOver ? styles.rfAddBtnActive : ''}`}
          onClick={handleAddClick}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          title="Click to add child, or drag items here"
        >
          +
        </div>
      )}
    </div>
  )
})
OperatorNodeComponent.displayName = 'OperatorNode'

const SignalNodeComponent = memo<NodeProps<FlowNodeData>>(({ data, selected }) => {
  const handleDoubleClick = useCallback((e: React.MouseEvent) => {
    e.stopPropagation()
    if (data.onDoubleClick) data.onDoubleClick(data.path)
  }, [data])
  const node = data.ruleNode as { signalType: string; signalName: string }
  return (
    <div
      className={`${styles.rfSignalNode} ${selected ? styles.rfNodeSelected : ''}`}
      onDoubleClick={handleDoubleClick}
      title={`${node.signalType}("${node.signalName}")\nDouble-click to edit`}
    >
      <Handle type="target" position={Position.Top} className={styles.rfHandle} />
      <span className={styles.rfSignalType}>{node.signalType}</span>
      <span className={styles.rfSignalName}>{node.signalName}</span>
    </div>
  )
})
SignalNodeComponent.displayName = 'SignalNode'

const nodeTypes: NodeTypes = {
  operatorNode: OperatorNodeComponent,
  signalNode: SignalNodeComponent,
}

// ═══════════════════════════════════════════════════════════════
// Edit Signal Dialog
// ═══════════════════════════════════════════════════════════════

interface EditSignalDialogProps {
  signalType: string
  signalName: string
  availableSignals: { signalType: string; name: string }[]
  onSave: (signalType: string, signalName: string) => void
  onCancel: () => void
}

const EditSignalDialog: React.FC<EditSignalDialogProps> = memo(({
  signalType: initType, signalName: initName, availableSignals, onSave, onCancel,
}) => {
  const [signalType, setSignalType] = useState(initType)
  const [signalName, setSignalName] = useState(initName)
  const [search, setSearch] = useState('')

  const types = useMemo(() => Array.from(new Set(availableSignals.map(s => s.signalType))).sort(), [availableSignals])
  const filteredSignals = useMemo(() => {
    let list = availableSignals.filter(s => s.signalType === signalType)
    if (search.trim()) {
      const q = search.toLowerCase()
      list = list.filter(s => s.name.toLowerCase().includes(q))
    }
    return list
  }, [availableSignals, signalType, search])

  const handleSubmit = useCallback((e: React.FormEvent) => {
    e.preventDefault()
    if (signalType.trim() && signalName.trim()) onSave(signalType, signalName)
  }, [signalType, signalName, onSave])

  // ESC to cancel
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onCancel() }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onCancel])

  return (
    <div className={styles.editOverlay} onClick={onCancel}>
      <div className={styles.editDialog} onClick={e => e.stopPropagation()}>
        <div className={styles.editDialogHeader}>
          <span>Edit Signal</span>
          <button className={styles.editDialogClose} onClick={onCancel}>×</button>
        </div>
        <form onSubmit={handleSubmit} className={styles.editDialogBody}>
          <label className={styles.editLabel}>
            Signal Type
            <select
              className={styles.editSelect}
              value={signalType}
              onChange={e => { setSignalType(e.target.value); setSignalName('') }}
            >
              {types.map(t => <option key={t} value={t}>{t}</option>)}
              {!types.includes(signalType) && <option value={signalType}>{signalType}</option>}
            </select>
          </label>
          <label className={styles.editLabel}>
            Signal Name
            <input
              className={styles.editInput}
              value={signalName}
              onChange={e => setSignalName(e.target.value)}
              placeholder="Enter signal name"
              autoFocus
            />
          </label>
          {filteredSignals.length > 0 && (
            <div className={styles.editSignalList}>
              <input
                className={styles.editSearchInput}
                value={search}
                onChange={e => setSearch(e.target.value)}
                placeholder="Filter signals..."
              />
              <div className={styles.editSignalOptions}>
                {filteredSignals.map(s => (
                  <div
                    key={s.name}
                    className={`${styles.editSignalOption} ${s.name === signalName ? styles.editSignalOptionActive : ''}`}
                    onClick={() => setSignalName(s.name)}
                  >
                    {s.name}
                  </div>
                ))}
              </div>
            </div>
          )}
          <div className={styles.editDialogActions}>
            <button type="button" className={styles.editBtnCancel} onClick={onCancel}>Cancel</button>
            <button type="submit" className={styles.editBtnSave} disabled={!signalType.trim() || !signalName.trim()}>Save</button>
          </div>
        </form>
      </div>
    </div>
  )
})
EditSignalDialog.displayName = 'EditSignalDialog'

// ═══════════════════════════════════════════════════════════════
// Add Child Picker — quick inline picker to add a child node
// ═══════════════════════════════════════════════════════════════

interface AddChildPickerProps {
  availableSignals: { signalType: string; name: string }[]
  onPick: (node: RuleNode) => void
  onCancel: () => void
}

const AddChildPicker: React.FC<AddChildPickerProps> = memo(({ availableSignals, onPick, onCancel }) => {
  const [search, setSearch] = useState('')

  const groups = useMemo(() => {
    const g: Record<string, { signalType: string; name: string }[]> = {}
    for (const s of availableSignals) {
      const key = s.signalType.toUpperCase()
      if (!g[key]) g[key] = []
      g[key].push(s)
    }
    return Object.entries(g).sort((a, b) => a[0].localeCompare(b[0]))
  }, [availableSignals])

  const filteredGroups = useMemo(() => {
    if (!search.trim()) return groups
    const q = search.toLowerCase()
    return groups
      .map(([type, signals]) => [type, signals.filter(s => s.name.toLowerCase().includes(q) || s.signalType.toLowerCase().includes(q))] as [string, typeof signals])
      .filter(([, signals]) => signals.length > 0)
  }, [groups, search])

  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onCancel() }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onCancel])

  return (
    <div className={styles.editOverlay} onClick={onCancel}>
      <div className={styles.addPickerDialog} onClick={e => e.stopPropagation()}>
        <div className={styles.editDialogHeader}>
          <span>Add Child Node</span>
          <button className={styles.editDialogClose} onClick={onCancel}>×</button>
        </div>
        <div className={styles.addPickerBody}>
          {/* Operator buttons */}
          <div className={styles.addPickerSection}>
            <div className={styles.addPickerSectionTitle}>Operators</div>
            <div className={styles.addPickerOps}>
              {(['AND', 'OR', 'NOT'] as const).map(op => (
                <button key={op} className={styles.addPickerOpBtn} onClick={() => {
                  onPick(op === 'NOT'
                    ? { operator: 'NOT', conditions: [] as unknown as [RuleNode] }
                    : { operator: op, conditions: [] })
                }}>{op}</button>
              ))}
            </div>
          </div>
          {/* Signal search + list */}
          <div className={styles.addPickerSection}>
            <div className={styles.addPickerSectionTitle}>Signals</div>
            <input
              className={styles.editInput}
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder="Search signals..."
              autoFocus
            />
            <div className={styles.addPickerSignalList}>
              {filteredGroups.map(([type, signals]) => (
                <div key={type}>
                  <div className={styles.addPickerGroupTitle}>{type}</div>
                  <div className={styles.addPickerGroupItems}>
                    {signals.map(s => (
                      <div
                        key={s.name}
                        className={styles.addPickerSignalItem}
                        onClick={() => onPick({ signalType: s.signalType, signalName: s.name })}
                      >
                        {s.name}
                      </div>
                    ))}
                  </div>
                </div>
              ))}
              {filteredGroups.length === 0 && <div className={styles.addPickerEmpty}>No matching signals</div>}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
})
AddChildPicker.displayName = 'AddChildPicker'

// ═══════════════════════════════════════════════════════════════
// Inner component (needs ReactFlowProvider context)
// ═══════════════════════════════════════════════════════════════

interface InnerProps {
  tree: RuleNode | null
  setTree: React.Dispatch<React.SetStateAction<RuleNode | null>>
  rawText: string
  setRawText: React.Dispatch<React.SetStateAction<string>>
  isRawMode: boolean
  setIsRawMode: React.Dispatch<React.SetStateAction<boolean>>
  maximized: boolean
  setMaximized: React.Dispatch<React.SetStateAction<boolean>>
  availableSignals: { signalType: string; name: string }[]
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

const ExpressionBuilderInner: React.FC<InnerProps> = ({
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
    const label = isLeaf(newNode) ? `${newNode.signalType}("${newNode.signalName}")` : (newNode as Exclude<RuleNode, {signalType: string}>).operator
    showToast(`Inserted ${label}`)
    setInsertSiblingTarget(null)
  }, [insertSiblingTarget, tree, pushHistory, setTree, showToast])

  // Sync tree → text → parent
  useEffect(() => {
    if (suppressSyncRef.current) { suppressSyncRef.current = false; return }
    const text = tree ? serializeNode(tree) : ''
    setRawText(text)
    internalChangeRef.current = true
    onChange(text)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tree])

  // Close context menu on outside click
  useEffect(() => {
    if (!contextMenu) return
    const handler = () => setContextMenu(null)
    window.addEventListener('click', handler)
    return () => window.removeEventListener('click', handler)
  }, [contextMenu])

  // ESC + Ctrl+Z/Y (only when not in an input element)
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

  // Signal grouping
  const signalGroups = useMemo(() => {
    const groups: Record<string, { signalType: string; name: string }[]> = {}
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
      .map(([type, signals]) => [type, signals.filter(s => s.name.toLowerCase().includes(q) || s.signalType.toLowerCase().includes(q))] as [string, typeof signals])
      .filter(([, signals]) => signals.length > 0)
  }, [signalGroups, signalSearch])

  const toggleGroup = useCallback((group: string) => {
    setCollapsedGroups(prev => { const n = new Set(prev); if (n.has(group)) n.delete(group); else n.add(group); return n })
  }, [])

  // ── Tree mutations ──

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

  // ── Double-click on node: edit signal only (operators use context menu) ──
  const handleNodeDoubleClick = useCallback((path: NodePath) => {
    if (!tree) return
    const node = getNodeAtPath(tree, path)
    if (!node) return
    if (isLeaf(node)) {
      setEditingNode({ path, signalType: node.signalType, signalName: node.signalName })
    }
    // Operator double-click intentionally disabled to prevent accidental toggles.
    // Use right-click context menu → "Toggle AND/OR" instead.
  }, [tree])

  const handleEditSave = useCallback((signalType: string, signalName: string) => {
    if (!editingNode || !tree) { setEditingNode(null); return }
    pushHistory(tree)
    setTree(replaceAtPath(tree, editingNode.path, { signalType, signalName }))
    setEditingNode(null)
    showToast(`Updated to ${signalType}("${signalName}")`)
  }, [editingNode, tree, pushHistory, setTree, showToast])

  // ── Drop on a specific node (from toolbox or tree) ──
  // Only allow drop on operator nodes — dropping on signal nodes is disabled
  // to prevent accidentally wrapping signals in an AND.
  const handleDropOnNode = useCallback((targetPath: NodePath, dragData: DragData) => {
    if (!tree) return
    const targetNode = getNodeAtPath(tree, targetPath)
    if (!targetNode) return
    // Block drop on leaf nodes to prevent accidental wrapping
    if (isLeaf(targetNode)) return
    const newNode = makeDragNode(dragData)
    if (!newNode) return
    pushHistory(tree)
    setTree(prev => prev ? addChildAtPath(prev, targetPath, newNode) : newNode)
    showToast('Added to operator node')
  }, [tree, pushHistory, setTree, showToast])

  // ── Click "+" button on operator node → open Add Child picker ──
  const handleAddChild = useCallback((targetPath: NodePath) => {
    setAddingToPath(targetPath)
  }, [])

  const handleAddChildPick = useCallback((newNode: RuleNode) => {
    if (!addingToPath || !tree) { setAddingToPath(null); return }
    pushHistory(tree)
    setTree(prev => prev ? addChildAtPath(prev, addingToPath, newNode) : newNode)
    const label = isLeaf(newNode) ? `${newNode.signalType}("${newNode.signalName}")` : (newNode as Exclude<RuleNode, {signalType: string}>).operator
    showToast(`Added ${label}`)
    setAddingToPath(null)
  }, [addingToPath, tree, pushHistory, setTree, showToast])

  // Sync tree → ReactFlow nodes/edges (must be after handlers)
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

  // ── Drop on canvas ──
  // When tree is empty: create a new root node
  // When tree exists and root is AND/OR: add as sibling to root's children
  // When tree exists and dropping an operator: wrap root with that operator
  // Otherwise: reject drop (user should drop onto a specific operator node)
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    const raw = e.dataTransfer.getData(DRAG_MIME)
    if (!raw) return
    const data = decodeDrag(raw)
    if (!data || data.kind === 'tree-node') return

    const newNode = makeDragNode(data)
    if (!newNode) return

    // If no tree yet, just set as root
    if (!tree) {
      pushHistory(tree)
      setTree(newNode)
      showToast('Created root node')
      return
    }

    // Dropping an operator wraps the whole tree
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

    // Dropping a signal: only auto-add if root is AND/OR
    if (isOperator(tree) && (tree.operator === 'AND' || tree.operator === 'OR')) {
      pushHistory(tree)
      setTree({ ...tree, conditions: [...tree.conditions, newNode] })
      showToast(`Added to root ${tree.operator}`)
      return
    }

    // Otherwise reject — don't silently wrap in AND
    showToast('Drop onto an operator node instead')
  }, [tree, pushHistory, setTree, showToast])

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'move'
  }, [])

  // ── Node click → select + context menu ──
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

  // ── Node delete via backspace/delete (only when canvas is focused, not in inputs) ──
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      // Don't interfere with inputs, textareas, selects, or contenteditable
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

  // ── Context menu actions ──
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

  // Raw text editing
  const handleRawChange = useCallback((text: string) => {
    setRawText(text)
    suppressSyncRef.current = true
    setTree(parseExprText(text))
    onChange(text)
  }, [onChange, setRawText, setTree])

  const validationIssues = useMemo(() => validateTree(tree, availableSignals), [tree, availableSignals])

  // Templates
  const TEMPLATES = useMemo(() => [
    { name: 'AND Gate', op: 'AND' as const, desc: 'A AND B', build: () => ({ operator: 'AND' as const, conditions: [] }) },
    { name: 'OR Gate', op: 'OR' as const, desc: 'A OR B', build: () => ({ operator: 'OR' as const, conditions: [] }) },
    { name: 'NOT Gate', op: 'NOT' as const, desc: 'NOT A', build: () => ({ operator: 'NOT' as const, conditions: [] as unknown as [RuleNode] }) },
  ], [])

  const applyTemplate = useCallback((tpl: typeof TEMPLATES[0]) => {
    pushHistory(tree)
    setTree(tpl.build())
  }, [tree, pushHistory, setTree, TEMPLATES])

  // ── Render ──
  return (
    <div className={`${styles.container} ${maximized ? styles.containerMaximized : ''}`}
      onClick={() => { setSelectedPath(null); setContextMenu(null) }}>
      <div className={styles.mainLayout}>
        {/* ReactFlow Canvas */}
        <div className={styles.canvasWrapper}>
          <div className={styles.zoomBar}>
            <div className={styles.toolbarGroup}>
              <button className={styles.zoomBtn} onClick={handleUndo} disabled={!canUndo} title="Undo (Ctrl+Z)">↩</button>
              <button className={styles.zoomBtn} onClick={handleRedo} disabled={!canRedo} title="Redo (Ctrl+Y)">↪</button>
            </div>
            <div className={styles.toolbarSep} />
            <div className={styles.toolbarGroup}>
              <button className={styles.zoomBtn} onClick={() => fitView({ padding: 0.15, duration: 200 })} title="Fit to view">Fit</button>
            </div>
            {selectedPath && (
              <span className={styles.zoomHint}>
                Selected: {(() => { const n = tree && getNodeAtPath(tree, selectedPath); return n ? (isLeaf(n) ? `${n.signalType}("${n.signalName}")` : n.operator) : '—' })()}
              </span>
            )}
            <div className={styles.toolbarSpacer} />
            <button
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
                <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="rgba(118, 185, 0, 0.15)" />
                <Controls
                  showInteractive={false}
                  position="bottom-left"
                  className={styles.rfControls}
                />
                <MiniMap
                  nodeColor={(n) => n.type === 'operatorNode' ? 'rgba(99, 102, 241, 0.7)' : 'rgba(118, 185, 0, 0.7)'}
                  maskColor="rgba(0, 0, 0, 0.15)"
                  className={styles.rfMinimap}
                  pannable
                  zoomable
                  position="bottom-right"
                  style={{ width: 120, height: 80 }}
                />
              </ReactFlow>
            ) : (
              <div className={styles.canvasEmpty}>
                <div className={styles.canvasEmptyIcon}>⊕</div>
                <div>Drag signals or operators here</div>
                <div className={styles.canvasEmptyHint}>or start with a template</div>
                <div className={styles.canvasTemplates}>
                  {TEMPLATES.map(tpl => {
                    const TplGate = tpl.op === 'AND' ? GateAND : tpl.op === 'OR' ? GateOR : GateNOT
                    return (
                      <div key={tpl.name} className={styles.canvasTemplateCard}
                        onClick={(e) => { e.stopPropagation(); applyTemplate(tpl) }}>
                        <span className={styles.canvasTemplateGate}><TplGate color="rgba(99, 102, 241, 0.6)" /></span>
                        <span className={styles.canvasTemplateName}>{tpl.name}</span>
                        <span className={styles.canvasTemplateDesc}>{tpl.desc}</span>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Toolbox */}
        <div className={`${styles.toolbox} ${toolboxCollapsed ? styles.toolboxCollapsed : ''}`}>
          <div className={styles.toolboxHeader} onClick={() => setToolboxCollapsed(!toolboxCollapsed)}>
            <span className={styles.toolboxHeaderTitle}>{toolboxCollapsed ? '▶' : '▼'} Toolbox</span>
            <span className={styles.toolboxHeaderCount}>{availableSignals.length} signals</span>
          </div>

          {!toolboxCollapsed && (
            <div className={styles.toolboxContent}>
              <div className={styles.toolboxOperators}>
                {(['AND', 'OR', 'NOT'] as const).map(op => {
                  const opColor = op === 'AND' ? '#818cf8' : op === 'OR' ? '#34d399' : '#f87171'
                  const opIcon = op === 'AND' ? '∧' : op === 'OR' ? '∨' : '¬'
                  return (
                    <div
                      key={op}
                      className={`${styles.toolboxOp} ${styles[`toolboxOp${op}`]}`}
                      draggable
                      onDragStart={(e) => {
                        e.dataTransfer.setData(DRAG_MIME, encodeDrag({ kind: 'operator', operator: op }))
                        e.dataTransfer.effectAllowed = 'copyMove'
                      }}
                      onClick={(e) => { e.stopPropagation() }}
                      title={`Drag ${op} gate to canvas`}
                    >
                      <span className={styles.toolboxOpIcon} style={{ color: opColor }}>{opIcon}</span>
                      {op}
                    </div>
                  )
                })}
                <button className={styles.clearBtn} onClick={(e) => { e.stopPropagation(); handleClear() }}>Clear</button>
              </div>

              <div className={styles.toolboxSearch}>
                <input
                  className={styles.toolboxSearchInput}
                  value={signalSearch}
                  onChange={(e) => setSignalSearch(e.target.value)}
                  placeholder="Search signals..."
                  onClick={(e) => e.stopPropagation()}
                />
                {signalSearch && <button className={styles.toolboxSearchClear} onClick={() => setSignalSearch('')}>×</button>}
              </div>

              <div className={styles.toolboxSignals}>
                {filteredGroups.map(([type, signals]) => {
                  const collapsed = collapsedGroups.has(type)
                  return (
                    <div key={type} className={styles.signalGroup}>
                      <div className={styles.signalGroupHeader} onClick={(e) => { e.stopPropagation(); toggleGroup(type) }}>
                        <span className={styles.signalGroupToggle}>{collapsed ? '▶' : '▼'}</span>
                        <span className={styles.signalGroupName}>{type}</span>
                        <span className={styles.signalGroupCount}>{signals.length}</span>
                      </div>
                      {!collapsed && (
                        <div className={styles.signalGroupItems}>
                          {signals.map(s => (
                            <div
                              key={`${s.signalType}-${s.name}`}
                              className={styles.toolboxChip}
                              draggable
                              onDragStart={(e) => {
                                e.dataTransfer.setData(DRAG_MIME, encodeDrag({ kind: 'signal', signalType: s.signalType, signalName: s.name }))
                                e.dataTransfer.effectAllowed = 'copyMove'
                              }}
                              onClick={(e) => { e.stopPropagation() }}
                              title={`Drag to canvas to add ${s.signalType}("${s.name}")`}
                            >
                              {s.name}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )
                })}
                {filteredGroups.length === 0 && (
                  <span className={styles.toolboxEmpty}>{signalSearch ? 'No matching signals' : 'No signals defined'}</span>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Result */}
      <div className={styles.result}>
        <span className={styles.resultLabel}>Result:</span>
        {tree ? <code className={styles.resultExpr}>{rawText}</code>
             : <span className={styles.resultPlaceholder}>No condition — route matches all requests</span>}
      </div>

      {/* Raw text */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <label className={styles.rawToggle}>
          <input type="checkbox" checked={isRawMode} onChange={e => setIsRawMode(e.target.checked)} style={{ accentColor: 'var(--color-primary)' }} />
          Edit raw expression
        </label>
      </div>
      {isRawMode && (
        <input className={styles.rawInput} value={rawText} onChange={e => handleRawChange(e.target.value)}
          placeholder='e.g. domain("math") AND complexity("hard")' onClick={e => e.stopPropagation()} />
      )}

      {/* Validation */}
      {validationIssues.length > 0 && <div>{validationIssues.map((w, i) => <div key={i} className={styles.validationWarn}>⚠ {w}</div>)}</div>}
      {tree && validationIssues.length === 0 && <div className={styles.validationOk}>✓ All referenced signals exist</div>}

      {/* Context menu */}
      {contextMenu && tree && (() => {
        const node = getNodeAtPath(tree, contextMenu.path)
        if (!node) return null
        return (
          <div className={styles.ctxMenu} style={{ left: contextMenu.x, top: contextMenu.y }}
            onClick={e => e.stopPropagation()}>
            {isLeaf(node) && (
              <div className={styles.ctxMenuItem} onClick={() => {
                setEditingNode({ path: contextMenu.path, signalType: (node as { signalType: string; signalName: string }).signalType, signalName: (node as { signalType: string; signalName: string }).signalName })
                setContextMenu(null)
              }}>Edit Signal</div>
            )}
            {isOperator(node) && node.operator !== 'NOT' && (
              <div className={styles.ctxMenuItem} onClick={() => {
                handleChangeOp(contextMenu.path, node.operator === 'AND' ? 'OR' : 'AND')
              }}>Toggle to {node.operator === 'AND' ? 'OR' : 'AND'}</div>
            )}
            {isOperator(node) && (node.operator !== 'NOT' || (node.conditions as RuleNode[]).length === 0) && (
              <div className={styles.ctxMenuItem} onClick={() => {
                setAddingToPath(contextMenu.path)
                setContextMenu(null)
              }}>Add child...</div>
            )}
            <div className={styles.ctxMenuItem} onClick={() => handleWrap(contextMenu.path, 'AND')}>Wrap with AND</div>
            <div className={styles.ctxMenuItem} onClick={() => handleWrap(contextMenu.path, 'OR')}>Wrap with OR</div>
            <div className={styles.ctxMenuItem} onClick={() => handleWrap(contextMenu.path, 'NOT')}>Wrap with NOT</div>
            {contextMenu.path.length > 0 && (
              <>
                <div className={styles.ctxMenuDivider} />
                <div className={styles.ctxMenuItem} onClick={() => {
                  const parentPath = contextMenu.path.slice(0, -1)
                  const idx = contextMenu.path[contextMenu.path.length - 1]
                  setInsertSiblingTarget({ parentPath, index: idx })
                  setContextMenu(null)
                }}>Insert before...</div>
                <div className={styles.ctxMenuItem} onClick={() => {
                  const parentPath = contextMenu.path.slice(0, -1)
                  const idx = contextMenu.path[contextMenu.path.length - 1]
                  setInsertSiblingTarget({ parentPath, index: idx + 1 })
                  setContextMenu(null)
                }}>Insert after...</div>
              </>
            )}
            {isOperator(node) && node.conditions.length > 0 && (
              <div className={styles.ctxMenuItem} onClick={() => handleUnwrap(contextMenu.path)}>Unwrap (replace with first child)</div>
            )}
            {isOperator(node) && node.operator !== 'NOT' && (
              <>
                {node.operator !== 'AND' && <div className={styles.ctxMenuItem} onClick={() => handleChangeOp(contextMenu.path, 'AND')}>Change to AND</div>}
                {node.operator !== 'OR' && <div className={styles.ctxMenuItem} onClick={() => handleChangeOp(contextMenu.path, 'OR')}>Change to OR</div>}
              </>
            )}
            <div className={styles.ctxMenuDivider} />
            <div className={`${styles.ctxMenuItem} ${styles.ctxMenuDanger}`}
              onClick={() => { handleDeleteNode(contextMenu.path); setContextMenu(null) }}>Delete</div>
          </div>
        )
      })()}

      {/* Inline edit dialog for signal nodes */}
      {editingNode && (
        <EditSignalDialog
          signalType={editingNode.signalType}
          signalName={editingNode.signalName}
          availableSignals={availableSignals}
          onSave={handleEditSave}
          onCancel={() => setEditingNode(null)}
        />
      )}

      {/* Add child picker */}
      {addingToPath && (
        <AddChildPicker
          availableSignals={availableSignals}
          onPick={handleAddChildPick}
          onCancel={() => setAddingToPath(null)}
        />
      )}

      {/* Insert sibling picker */}
      {insertSiblingTarget && (
        <AddChildPicker
          availableSignals={availableSignals}
          onPick={handleInsertSiblingPick}
          onCancel={() => setInsertSiblingTarget(null)}
        />
      )}

      {/* Toast feedback */}
      {toast && <div className={styles.toast}>{toast}</div>}
    </div>
  )
}

// ═══════════════════════════════════════════════════════════════
// Outer Component (provides ReactFlowProvider)
// ═══════════════════════════════════════════════════════════════

interface ExpressionBuilderProps {
  value: string
  onChange: (expr: string) => void
  initialAstExpr?: Record<string, unknown> | null
  availableSignals: { signalType: string; name: string }[]
}

const ExpressionBuilder: React.FC<ExpressionBuilderProps> = ({
  value, onChange, initialAstExpr, availableSignals,
}) => {
  const [tree, setTree] = useState<RuleNode | null>(() => {
    if (initialAstExpr) { const n = boolExprToRuleNode(initialAstExpr); if (n) return n }
    return parseExprText(value)
  })

  const [rawText, setRawText] = useState(value)
  const [isRawMode, setIsRawMode] = useState(false)
  const [maximized, setMaximized] = useState(false)
  const [selectedPath, setSelectedPath] = useState<NodePath | null>(null)

  // Undo / Redo history
  const [history, setHistory] = useState<(RuleNode | null)[]>([])
  const [historyIdx, setHistoryIdx] = useState(-1)
  const skipHistoryRef = useRef(false)

  const pushHistory = useCallback((prev: RuleNode | null) => {
    if (skipHistoryRef.current) { skipHistoryRef.current = false; return }
    setHistory(h => {
      const trimmed = h.slice(0, historyIdx + 1)
      return [...trimmed, prev].slice(-50)
    })
    setHistoryIdx(i => Math.min(i + 1, 49))
  }, [historyIdx])

  const canUndo = historyIdx >= 0
  const canRedo = historyIdx < history.length - 1

  const handleUndo = useCallback(() => {
    if (!canUndo) return
    skipHistoryRef.current = true
    setHistory(h => {
      const trimmed = h.slice(0, historyIdx + 1)
      return [...trimmed, tree]
    })
    setTree(history[historyIdx])
    setHistoryIdx(i => i - 1)
  }, [canUndo, history, historyIdx, tree])

  const handleRedo = useCallback(() => {
    if (!canRedo) return
    skipHistoryRef.current = true
    setTree(history[historyIdx + 1])
    setHistoryIdx(i => i + 1)
  }, [canRedo, history, historyIdx])

  // Sync external value changes
  const prevValueRef = useRef(value)
  const internalChangeRef = useRef(false)
  useEffect(() => {
    if (value !== prevValueRef.current) {
      prevValueRef.current = value
      if (internalChangeRef.current) {
        internalChangeRef.current = false
        return
      }
      if (initialAstExpr) {
        const n = boolExprToRuleNode(initialAstExpr)
        if (n) { setTree(n); setRawText(value); return }
      }
      // Only re-parse into tree if the text actually parses successfully.
      // Incomplete expressions (e.g. containing '?') should not destroy the current tree.
      const parsed = parseExprText(value)
      if (parsed) {
        setTree(parsed)
      }
      setRawText(value)
    }
  }, [value, initialAstExpr])

  const innerProps: InnerProps = {
    tree, setTree, rawText, setRawText, isRawMode, setIsRawMode,
    maximized, setMaximized, availableSignals, onChange, pushHistory,
    canUndo, canRedo, handleUndo, handleRedo, selectedPath, setSelectedPath,
    internalChangeRef,
  }

  const content = (
    <ReactFlowProvider>
      <ExpressionBuilderInner {...innerProps} />
    </ReactFlowProvider>
  )

  if (maximized) {
    return createPortal(
      <div className={styles.fullscreenOverlay}>
        <div className={styles.fullscreenContainer}>
          <div className={styles.fullscreenHeader}>
            <span className={styles.fullscreenTitle}>Expression Builder</span>
            <button className={styles.fullscreenCloseBtn} onClick={() => setMaximized(false)} title="Exit fullscreen (Esc)">✕</button>
          </div>
          {content}
        </div>
      </div>,
      document.body
    )
  }

  return content
}

export default ExpressionBuilder
export { serializeNode, parseExprText }
export type { RuleNode as ExprRuleNode }
