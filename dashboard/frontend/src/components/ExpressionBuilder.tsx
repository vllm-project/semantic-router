/**
 * ExpressionBuilder — the Builder's DSL-string adapter over DecisionRuleEditor.
 *
 * Builder routes carry their condition as WHEN text, so this adapter turns the
 * text (or the route's WASM AST) into a rule document and serializes every
 * edit back to WHEN text. The canvas, history and raw mode are the shared
 * DecisionRuleEditor that ConfigPage also uses.
 */

import React, { useCallback, useEffect, useRef, useState } from 'react'

import { deriveRootShape, type DecisionRuleDocument, type RuleIssue } from '@/lib/decisionRuleAst'
import { boolExprToRuleNode, parseRuleExpr, serializeRuleExpr } from '@/lib/decisionRuleDsl'

import DecisionRuleEditor from './DecisionRuleEditor'
import type { SignalDescriptor } from './ExpressionBuilderSupport'

interface ExpressionBuilderProps {
  value: string
  onChange: (expr: string) => void
  initialAstExpr?: Record<string, unknown> | null
  availableSignals: SignalDescriptor[]
  onIssuesChange?: (issues: RuleIssue[]) => void
}

function documentFromBuilder(value: string, initialAstExpr?: Record<string, unknown> | null): DecisionRuleDocument {
  const fromAst = initialAstExpr ? boolExprToRuleNode(initialAstExpr) : null
  const tree = fromAst ?? parseRuleExpr(value).tree
  return { tree, rootShape: deriveRootShape(tree) }
}

const ExpressionBuilder: React.FC<ExpressionBuilderProps> = ({
  value, onChange, initialAstExpr, availableSignals, onIssuesChange,
}) => {
  const [doc, setDoc] = useState<DecisionRuleDocument>(() => documentFromBuilder(value, initialAstExpr))
  const lastEmittedRef = useRef(value)
  useEffect(() => {
    if (value === lastEmittedRef.current) return
    lastEmittedRef.current = value
    setDoc(documentFromBuilder(value, initialAstExpr))
  }, [value, initialAstExpr])

  const handleChange = useCallback((next: DecisionRuleDocument) => {
    setDoc(next)
    const text = serializeRuleExpr(next.tree).text
    lastEmittedRef.current = text
    onChange(text)
  }, [onChange])

  return (
    <DecisionRuleEditor
      value={doc}
      onChange={handleChange}
      availableSignals={availableSignals}
      onIssuesChange={onIssuesChange}
    />
  )
}

export default ExpressionBuilder
