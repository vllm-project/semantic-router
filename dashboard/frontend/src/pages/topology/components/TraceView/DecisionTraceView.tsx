// DecisionTraceView.tsx - Recursive renderer for the router's exact eval trace
// (decision.DecisionTrace / decision.TraceNode), replacing the flat
// signal-name heuristic that could not represent nested rules.

import React from 'react'
import type { DecisionTrace, TraceNode } from '../../types'
import styles from './DecisionTraceView.module.css'

interface DecisionTraceViewProps {
  traces: DecisionTrace[]
  selectedDecisionName: string | null
}

function formatConfidence(confidence: number): string {
  return `${Math.round(confidence * 100)}%`
}

const TraceNodeView: React.FC<{ node: TraceNode; depth: number }> = ({ node, depth }) => {
  const isLeaf = node.node_type === 'leaf'
  return (
    <div className={styles.node} style={{ marginLeft: depth > 0 ? '0.75rem' : 0 }}>
      <div className={`${styles.nodeRow} ${node.matched ? styles.matched : styles.unmatched}`}>
        <span className={styles.nodeMark}>{node.matched ? '✓' : '✗'}</span>
        {isLeaf ? (
          <span className={styles.nodeLabel}>
            {node.signal_type}({node.signal_name})
            {node.label && <span className={styles.nodeBadge}>{node.label}</span>}
          </span>
        ) : (
          <span className={styles.nodeOperator}>{node.node_type}</span>
        )}
        <span className={styles.nodeConfidence}>{formatConfidence(node.confidence)}</span>
      </div>
      {node.children && node.children.length > 0 && (
        <div className={styles.children}>
          {node.children.map((child, index) => (
            <TraceNodeView key={index} node={child} depth={depth + 1} />
          ))}
        </div>
      )}
    </div>
  )
}

export const DecisionTraceView: React.FC<DecisionTraceViewProps> = ({
  traces,
  selectedDecisionName,
}) => {
  if (!traces || traces.length === 0) return null

  return (
    <div className={styles.trace}>
      {traces.map((trace) => {
        const isSelected = trace.decision_name === selectedDecisionName
        return (
          <div
            key={trace.decision_name}
            className={`${styles.decision} ${isSelected ? styles.decisionSelected : ''}`}
          >
            <div className={styles.decisionHeader}>
              <span className={styles.decisionName}>{trace.decision_name}</span>
              <span className={trace.matched ? styles.decisionMatched : styles.decisionUnmatched}>
                {trace.matched ? 'matched' : 'not matched'} · {formatConfidence(trace.confidence)}
              </span>
            </div>
            {trace.root_trace && <TraceNodeView node={trace.root_trace} depth={0} />}
          </div>
        )
      })}
    </div>
  )
}
