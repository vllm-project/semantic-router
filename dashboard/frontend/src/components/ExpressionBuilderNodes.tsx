import React, { memo, useCallback, useState } from 'react'
import { Handle, Position, type NodeProps, type NodeTypes } from 'reactflow'

import styles from './ExpressionBuilder.module.css'
import { DRAG_MIME, decodeDrag, type RuleNode } from './ExpressionBuilderSupport'
import type { FlowNodeData } from './ExpressionBuilderFlow'

interface GateShapeProps {
  color: string
  opacity?: number
}

export type OperatorKind = 'AND' | 'OR' | 'NOT'

export const GateAND: React.FC<GateShapeProps> = ({ color, opacity = 1 }) => (
  <svg viewBox="0 0 84 64" width="84" height="64" className={styles.gateSvg} style={{ opacity }}>
    <path
      d="M 6 4 L 78 4 L 78 30 Q 78 60 42 60 Q 6 60 6 30 Z"
      fill={color.replace(/[\d.]+\)$/, '0.08)')}
      stroke={color}
      strokeWidth="2.5"
      strokeLinejoin="round"
    />
  </svg>
)

export const GateOR: React.FC<GateShapeProps> = ({ color, opacity = 1 }) => (
  <svg viewBox="0 0 84 64" width="84" height="64" className={styles.gateSvg} style={{ opacity }}>
    <path
      d="M 6 4 Q 42 18 78 4 Q 74 44 42 62 Q 10 44 6 4 Z"
      fill={color.replace(/[\d.]+\)$/, '0.08)')}
      stroke={color}
      strokeWidth="2.5"
      strokeLinejoin="round"
    />
  </svg>
)

export const GateNOT: React.FC<GateShapeProps> = ({ color, opacity = 1 }) => (
  <svg viewBox="0 0 84 70" width="84" height="70" className={styles.gateSvg} style={{ opacity }}>
    <path
      d="M 10 4 L 74 4 L 42 52 Z"
      fill={color.replace(/[\d.]+\)$/, '0.08)')}
      stroke={color}
      strokeWidth="2.5"
      strokeLinejoin="round"
    />
    <circle
      cx="42"
      cy="60"
      r="6"
      fill={color.replace(/[\d.]+\)$/, '0.08)')}
      stroke={color}
      strokeWidth="2.5"
    />
  </svg>
)

export const OPERATOR_ORDER: OperatorKind[] = ['AND', 'OR', 'NOT']

export const OPERATOR_META: Record<
  OperatorKind,
  {
    color: string
    description: string
    icon: string
    GateShape: React.FC<GateShapeProps>
  }
> = {
  AND: {
    color: '#818cf8',
    description: 'A AND B',
    icon: '∧',
    GateShape: GateAND,
  },
  OR: {
    color: '#34d399',
    description: 'A OR B',
    icon: '∨',
    GateShape: GateOR,
  },
  NOT: {
    color: '#f87171',
    description: 'NOT A',
    icon: '¬',
    GateShape: GateNOT,
  },
}

export interface BuilderTemplate {
  name: string
  op: OperatorKind
  desc: string
  build: () => RuleNode
}

export const BUILDER_TEMPLATES: BuilderTemplate[] = [
  {
    name: 'AND Gate',
    op: 'AND',
    desc: OPERATOR_META.AND.description,
    build: () => ({ operator: 'AND', conditions: [] }),
  },
  {
    name: 'OR Gate',
    op: 'OR',
    desc: OPERATOR_META.OR.description,
    build: () => ({ operator: 'OR', conditions: [] }),
  },
  {
    name: 'NOT Gate',
    op: 'NOT',
    desc: OPERATOR_META.NOT.description,
    build: () => ({ operator: 'NOT', conditions: [] as unknown as [RuleNode] }),
  },
]

const OperatorNodeComponent = memo<NodeProps<FlowNodeData>>(({ data, selected }) => {
  const [dragOver, setDragOver] = useState(false)

  const handleDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault()
    event.stopPropagation()
    event.dataTransfer.dropEffect = 'copy'
    setDragOver(true)
  }, [])

  const handleDragLeave = useCallback(() => setDragOver(false), [])

  const handleDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault()
      event.stopPropagation()
      setDragOver(false)
      const raw = event.dataTransfer.getData(DRAG_MIME)
      if (!raw) return
      const dragData = decodeDrag(raw)
      if (dragData && data.onDropOnNode) {
        data.onDropOnNode(data.path, dragData)
      }
    },
    [data]
  )

  const handleDoubleClick = useCallback(
    (event: React.MouseEvent) => {
      event.stopPropagation()
      data.onDoubleClick?.(data.path)
    },
    [data]
  )

  const handleAddClick = useCallback(
    (event: React.MouseEvent) => {
      event.stopPropagation()
      data.onAddChild?.(data.path)
    },
    [data]
  )

  const opNode = data.ruleNode as Exclude<RuleNode, { signalType: string }>
  const childCount = opNode.conditions.length
  const showAddBtn = opNode.operator !== 'NOT' || childCount === 0
  const gateMeta = OPERATOR_META[opNode.operator]

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
        <gateMeta.GateShape color={gateMeta.color} />
        <span
          className={`${styles.rfGateLabel} ${opNode.operator === 'NOT' ? styles.rfGateLabelNot : ''}`}
          style={{ color: gateMeta.color }}
        >
          {data.label}
          {childCount > 0 ? <span className={styles.rfGateBadge}>{childCount}</span> : null}
        </span>
        <Handle type="source" position={Position.Bottom} className={styles.rfHandle} />
      </div>
      {showAddBtn ? (
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
      ) : null}
    </div>
  )
})
OperatorNodeComponent.displayName = 'OperatorNode'

const SignalNodeComponent = memo<NodeProps<FlowNodeData>>(({ data, selected }) => {
  const handleDoubleClick = useCallback(
    (event: React.MouseEvent) => {
      event.stopPropagation()
      data.onDoubleClick?.(data.path)
    },
    [data]
  )

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

export const nodeTypes: NodeTypes = {
  operatorNode: OperatorNodeComponent,
  signalNode: SignalNodeComponent,
}
