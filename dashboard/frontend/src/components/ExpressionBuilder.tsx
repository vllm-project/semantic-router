/**
 * ExpressionBuilder — ReactFlow-based drag-and-drop boolean expression editor (v4).
 */

import React, { useState, useCallback, useRef, useEffect } from 'react'
import { createPortal } from 'react-dom'
import { ReactFlowProvider } from 'reactflow'
import 'reactflow/dist/style.css'
import styles from './ExpressionBuilder.module.css'
import ExpressionBuilderInner, { type ExpressionBuilderInnerProps } from './ExpressionBuilderInner'
import {
  boolExprToRuleNode,
  parseExprText,
  type NodePath,
  type RuleNode,
  type SignalDescriptor,
} from './ExpressionBuilderSupport'

interface ExpressionBuilderProps {
  value: string
  onChange: (expr: string) => void
  initialAstExpr?: Record<string, unknown> | null
  availableSignals: SignalDescriptor[]
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
      const parsed = parseExprText(value)
      if (parsed) {
        setTree(parsed)
      }
      setRawText(value)
    }
  }, [value, initialAstExpr])

  const innerProps: ExpressionBuilderInnerProps = {
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
            <button type="button" className={styles.fullscreenCloseBtn} onClick={() => setMaximized(false)} title="Exit fullscreen (Esc)">✕</button>
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
