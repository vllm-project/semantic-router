/**
 * DecisionRuleEditor — the controlled, shared decision-rule editor.
 *
 * Builder routes (through the ExpressionBuilder DSL-string adapter) and
 * ConfigPage decisions (through the canonical-rule adapter) both render this
 * component, so there is one ReactFlow canvas, one undo/redo history and one
 * raw-expression implementation. The value is a DecisionRuleDocument; DSL text
 * is only a derived view that is parsed on an explicit, complete raw edit.
 */

import React, { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { ReactFlowProvider } from 'reactflow'
import 'reactflow/dist/style.css'

import {
  deriveRootShape,
  isRuleGroup,
  normalizeForDsl,
  ruleSubtreeEquals,
  assignRuleIds,
  type DecisionRuleDocument,
  type RuleIssue,
  type RuleNode,
  type RuleOnUnknown,
} from '@/lib/decisionRuleAst'
import { parseRuleExpr, serializeRuleExpr } from '@/lib/decisionRuleDsl'
import { trapDialogFocus } from '@/hooks/useAccessibleDialog'
import {
  RAW_EXPRESSION_INCOMPLETE,
  validateRuleDocument,
  type RuleSignalScope,
} from '@/lib/decisionRuleValidation'

import styles from './ExpressionBuilder.module.css'
import ExpressionBuilderInner, { type ExpressionBuilderInnerProps } from './ExpressionBuilderInner'
import type { NodePath } from './ExpressionBuilderSupport'

export interface DecisionRuleEditorProps {
  value: DecisionRuleDocument
  onChange: (next: DecisionRuleDocument) => void
  signalScope: RuleSignalScope
  /** Validation issues plus any raw-expression parse error. */
  onIssuesChange?: (issues: RuleIssue[]) => void
  /** Prefix for DOM ids, so several editors can share a page. */
  idPrefix?: string
  title?: string
  /** ConfigPage edits rules.on_unknown here; the Builder ROUTE header owns it. */
  showOnUnknown?: boolean
  /** Help rendered under the on_unknown select. */
  onUnknownHelp?: React.ReactNode
}

const HISTORY_LIMIT = 50
const ON_UNKNOWN_OPTIONS: Array<{ value: RuleOnUnknown; label: string }> = [
  { value: 'no_match', label: 'No match' },
  { value: 'match', label: 'Match' },
  { value: 'fail_request', label: 'Fail request' },
]

function treeHasExtra(node: RuleNode | null): boolean {
  if (!node) return false
  if (node.extra && Object.keys(node.extra).length > 0) return true
  return isRuleGroup(node) && node.conditions.some(treeHasExtra)
}

function issuesKey(issues: RuleIssue[]): string {
  return JSON.stringify(issues.map((issue) => [issue.nodeId, issue.path, issue.severity, issue.message]))
}

const DecisionRuleEditor: React.FC<DecisionRuleEditorProps> = ({
  value,
  onChange,
  signalScope,
  onIssuesChange,
  idPrefix,
  title = 'Expression Builder',
  showOnUnknown = false,
  onUnknownHelp,
}) => {
  const generatedId = useId()
  const prefix = idPrefix ?? `rule-editor${generatedId.replace(/:/g, '')}`

  const [tree, setTreeState] = useState<RuleNode | null>(() => assignRuleIds(value.tree))
  const treeRef = useRef(tree)
  const valueRef = useRef(value)
  valueRef.current = value
  const onChangeRef = useRef(onChange)
  onChangeRef.current = onChange

  const [past, setPast] = useState<(RuleNode | null)[]>([])
  const [future, setFuture] = useState<(RuleNode | null)[]>([])
  const [isRawMode, setIsRawModeState] = useState(false)
  const [rawDraft, setRawDraft] = useState('')
  const [rawError, setRawError] = useState<string | null>(null)
  const [maximized, setMaximized] = useState(false)
  const [selectedPath, setSelectedPath] = useState<NodePath | null>(null)
  // The tree when raw editing started; an unparsable draft restores it.
  const rawBaseRef = useRef<RuleNode | null>(null)
  const rawErrorRef = useRef<string | null>(null)
  rawErrorRef.current = rawError
  const shellRef = useRef<HTMLDivElement>(null)

  const serialized = useMemo(() => serializeRuleExpr(tree), [tree])

  // Every tree mutation goes through here: the canvas mirrors it and the
  // parent receives the whole document, never a partial tree.
  const commitTree = useCallback(
    (action: React.SetStateAction<RuleNode | null>, opts: { fromRaw?: boolean } = {}) => {
      const prev = treeRef.current
      const next = typeof action === 'function' ? action(prev) : action
      if (next === prev) return
      treeRef.current = next
      setTreeState(next)
      if (!opts.fromRaw) {
        setRawDraft(serializeRuleExpr(next).text)
        setRawError(null)
      }
      onChangeRef.current({ ...valueRef.current, tree: next, rootShape: deriveRootShape(next) })
    },
    [],
  )

  // Resync only when the parent's tree differs structurally; ids and explicit
  // nesting of the internal tree survive the parent's re-derivation. A
  // different tree is a new rule (the Builder reuses this editor across
  // routes), so raw mode and history from the old rule are dropped with it.
  useEffect(() => {
    if (ruleSubtreeEquals(value.tree, treeRef.current)) return
    const next = assignRuleIds(value.tree)
    treeRef.current = next
    setTreeState(next)
    setIsRawModeState(false)
    rawBaseRef.current = null
    setRawDraft(serializeRuleExpr(next).text)
    setRawError(null)
    setPast([])
    setFuture([])
    setSelectedPath(null)
  }, [value.tree])

  const pushHistory = useCallback((prev: RuleNode | null) => {
    setPast((stack) => [...stack, prev].slice(-HISTORY_LIMIT))
    setFuture([])
  }, [])

  const canUndo = past.length > 0
  const canRedo = future.length > 0

  const handleUndo = useCallback(() => {
    if (past.length === 0) return
    const previous = past[past.length - 1]
    setPast(past.slice(0, -1))
    setFuture((stack) => [treeRef.current, ...stack].slice(0, HISTORY_LIMIT))
    commitTree(previous)
  }, [past, commitTree])

  const handleRedo = useCallback(() => {
    if (future.length === 0) return
    const [next, ...rest] = future
    setFuture(rest)
    setPast((stack) => [...stack, treeRef.current].slice(-HISTORY_LIMIT))
    commitTree(next)
  }, [future, commitTree])

  const rawDisabledReason = treeHasExtra(tree)
    ? 'Raw editing is unavailable: a condition has fields the DSL cannot express.'
    : null

  const setIsRawMode = useCallback((on: boolean) => {
    setIsRawModeState(on)
    if (on) {
      rawBaseRef.current = treeRef.current
    } else if (rawErrorRef.current !== null) {
      // An unparsable draft is discarded whole: restore the tree from before
      // raw editing, never a complete prefix committed while typing.
      const base = rawBaseRef.current
      if (!ruleSubtreeEquals(base, treeRef.current)) {
        pushHistory(treeRef.current)
        commitTree(base)
      }
    }
    setRawDraft(serializeRuleExpr(treeRef.current).text)
    setRawError(null)
  }, [commitTree, pushHistory])

  const handleRawChange = useCallback(
    (text: string) => {
      setRawDraft(text)
      const parsed = parseRuleExpr(text)
      if (parsed.error) {
        setRawError(parsed.error.message)
        return
      }
      setRawError(null)
      const current = treeRef.current
      // Text that means the same tree keeps the explicit nesting, one-child
      // groups and node ids the DSL cannot express.
      if (ruleSubtreeEquals(normalizeForDsl(parsed.tree), normalizeForDsl(current))) return
      pushHistory(current)
      commitTree(parsed.tree, { fromRaw: true })
    },
    [commitTree, pushHistory],
  )

  const handleOnUnknownChange = useCallback((next: string) => {
    const doc = valueRef.current
    const updated: DecisionRuleDocument = { ...doc, tree: treeRef.current }
    if (next) updated.onUnknown = next as RuleOnUnknown
    else delete updated.onUnknown
    onChangeRef.current(updated)
  }, [])

  // The fullscreen overlay is portaled outside any host dialog, and switching
  // into or out of it remounts the editor, so focus falls to <body>. Escape is
  // consumed at the document before a host dialog (useAccessibleDialog) can
  // close, and focus returns to the editor so its own shortcuts keep working.
  useEffect(() => {
    if (!maximized) return
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key !== 'Escape' || event.defaultPrevented) return
      // A dialog this editor opened (the leaf editor) closes first.
      const target = event.target as HTMLElement | null
      const dialog = target?.closest?.('[role="dialog"]')
      if (dialog && !(shellRef.current && dialog.contains(shellRef.current))) return
      event.preventDefault()
      event.stopPropagation()
      setMaximized(false)
    }
    document.addEventListener('keydown', handleEscape, true)
    return () => document.removeEventListener('keydown', handleEscape, true)
  }, [maximized])

  // A host dialog's focus trap still sees itself as topmost and would pull Tab
  // focus behind the overlay, so the overlay traps Tab first (window capture
  // runs before the host's document capture). Dialogs opened from the editor
  // keep their own trap.
  const overlayRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!maximized) return
    const handleTab = (event: KeyboardEvent) => {
      if (event.key !== 'Tab' || event.defaultPrevented) return
      const active = document.activeElement
      if (active?.closest('[data-rule-editor-dialog]')) return
      event.stopPropagation()
      trapDialogFocus(overlayRef, event)
    }
    window.addEventListener('keydown', handleTab, true)
    return () => window.removeEventListener('keydown', handleTab, true)
  }, [maximized])

  const lastMaximizedRef = useRef(maximized)
  useEffect(() => {
    if (lastMaximizedRef.current === maximized) return
    lastMaximizedRef.current = maximized
    shellRef.current?.querySelector<HTMLElement>('[data-rule-editor-container]')?.focus({ preventScroll: true })
  }, [maximized])

  const treeIssues = useMemo(
    () => validateRuleDocument({ ...value, tree }, signalScope, { onUnknownManagedElsewhere: !showOnUnknown }),
    [value, tree, signalScope, showOnUnknown],
  )
  // The raw error is shown beside the raw input, so only the host sees it as an issue.
  const issues = useMemo(
    () =>
      rawError
        ? [{ path: [], severity: 'error' as const, message: `${RAW_EXPRESSION_INCOMPLETE}: ${rawError}` }, ...treeIssues]
        : treeIssues,
    [treeIssues, rawError],
  )

  const lastIssuesKeyRef = useRef<string | null>(null)
  useEffect(() => {
    if (!onIssuesChange) return
    const key = issuesKey(issues)
    if (key === lastIssuesKeyRef.current) return
    lastIssuesKeyRef.current = key
    onIssuesChange(issues)
  }, [issues, onIssuesChange])

  const innerProps: ExpressionBuilderInnerProps = {
    tree,
    setTree: commitTree,
    resultText: serialized.text,
    rawText: isRawMode ? rawDraft : serialized.text,
    onRawChange: handleRawChange,
    rawError,
    isRawMode,
    setIsRawMode,
    rawDisabledReason,
    maximized,
    setMaximized,
    availableSignals: signalScope.descriptors,
    issues: treeIssues,
    pushHistory,
    canUndo,
    canRedo,
    handleUndo,
    handleRedo,
    selectedPath,
    setSelectedPath,
    idPrefix: prefix,
  }

  const content = (
    <div ref={shellRef} className={styles.ruleEditorShell}>
      {showOnUnknown ? (
        <div className={styles.ruleEditorPolicy}>
          <label className={styles.ruleEditorPolicyLabel} htmlFor={`${prefix}-on-unknown`}>
            On unknown
          </label>
          <select
            id={`${prefix}-on-unknown`}
            className={styles.editSelect}
            value={value.onUnknown ?? ''}
            onChange={(event) => handleOnUnknownChange(event.target.value)}
          >
            <option value="">Use condition policy</option>
            {ON_UNKNOWN_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
          {onUnknownHelp}
        </div>
      ) : null}
      <ReactFlowProvider>
        <ExpressionBuilderInner {...innerProps} />
      </ReactFlowProvider>
    </div>
  )

  if (maximized) {
    return createPortal(
      <div className={styles.fullscreenOverlay}>
        <div ref={overlayRef} className={styles.fullscreenContainer}>
          <div className={styles.fullscreenHeader}>
            <span className={styles.fullscreenTitle}>{title}</span>
            <button type="button" className={styles.fullscreenCloseBtn} onClick={() => setMaximized(false)} title="Exit fullscreen (Esc)">✕</button>
          </div>
          {content}
        </div>
      </div>,
      document.body,
    )
  }

  return content
}

export default DecisionRuleEditor
