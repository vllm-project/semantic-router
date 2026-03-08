import Dagre from '@dagrejs/dagre'
import { MarkerType, type Edge, type Node } from 'reactflow'

import { isLeaf, isOperator, type DragData, type NodePath, type RuleNode } from './ExpressionBuilderSupport'

export interface FlowNodeData {
  ruleNode: RuleNode
  path: NodePath
  label: string
  isOperator: boolean
  depth?: number
  onDoubleClick?: (path: NodePath) => void
  onDropOnNode?: (targetPath: NodePath, data: DragData) => void
  onAddChild?: (targetPath: NodePath) => void
}

const OPERATOR_W = 88
const OPERATOR_H = 78
const SIGNAL_W = 200
const SIGNAL_H = 40

let idCounter = 0

export function treeToFlowElements(
  root: RuleNode,
  onDoubleClick?: (path: NodePath) => void,
  onDropOnNode?: (targetPath: NodePath, data: DragData) => void,
  onAddChild?: (targetPath: NodePath) => void,
): { nodes: Node<FlowNodeData>[]; edges: Edge[] } {
  idCounter = 0
  const nodes: Node<FlowNodeData>[] = []
  const edges: Edge[] = []

  function walk(node: RuleNode, path: NodePath, parentId?: string, depth = 0) {
    const id = nextId()
    const operatorNode = isOperator(node)
    const label = operatorNode ? node.operator : `${node.signalType}("${node.signalName}")`

    nodes.push({
      id,
      type: operatorNode ? 'operatorNode' : 'signalNode',
      position: { x: 0, y: 0 },
      data: {
        ruleNode: node,
        path,
        label,
        isOperator: operatorNode,
        onDoubleClick,
        onDropOnNode,
        onAddChild,
        depth,
      },
    })

    if (parentId) {
      edges.push({
        id: `e-${parentId}-${id}`,
        source: parentId,
        target: id,
        type: 'default',
        style: { stroke: 'rgba(118, 185, 0, 0.5)', strokeWidth: 2 },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: 'rgba(118, 185, 0, 0.55)',
          width: 12,
          height: 12,
        },
      })
    }

    if (operatorNode) {
      node.conditions.forEach((child, index) => {
        walk(child, [...path, index], id, depth + 1)
      })
    }
  }

  walk(root, [])
  return { nodes, edges }
}

export function applyDagreLayout(
  nodes: Node<FlowNodeData>[],
  edges: Edge[],
): Node<FlowNodeData>[] {
  const graph = new Dagre.graphlib.Graph().setDefaultEdgeLabel(() => ({}))
  graph.setGraph({
    rankdir: 'TB',
    nodesep: 52,
    ranksep: 68,
    marginx: 24,
    marginy: 24,
  })

  for (const node of nodes) {
    const width = node.data.isOperator ? OPERATOR_W : estimateSignalWidth(node.data.ruleNode)
    const height = node.data.isOperator ? OPERATOR_H : SIGNAL_H
    graph.setNode(node.id, { width, height })
  }
  for (const edge of edges) {
    graph.setEdge(edge.source, edge.target)
  }

  Dagre.layout(graph)

  return nodes.map((node) => {
    const { x, y } = graph.node(node.id)
    const width = node.data.isOperator ? OPERATOR_W : estimateSignalWidth(node.data.ruleNode)
    const height = node.data.isOperator ? OPERATOR_H : SIGNAL_H
    return {
      ...node,
      position: { x: x - width / 2, y: y - height / 2 },
    }
  })
}

function nextId(): string {
  idCounter += 1
  return `n${idCounter}`
}

function estimateSignalWidth(node: RuleNode): number {
  if (!isLeaf(node)) return OPERATOR_W
  const text = `${node.signalType}  ${node.signalName}`
  const estimated = Math.max(SIGNAL_W, text.length * 7 + 32)
  return Math.min(estimated, 320)
}
