import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import { ruleSignalScope } from '../lib/decisionRuleValidation'
import ConfigPageDecisionRulesEditor from './ConfigPageDecisionRulesEditor'
import { decisionColumns } from './configPageDecisionTable'
import type { DecisionConfig, DecisionRuleSet } from './configPageSupport'

const singleCondition: DecisionRuleSet = {
  type: 'jailbreak',
  name: 'prompt_injection',
  on_unknown: 'no_match',
}

const renderRules = (value: DecisionRuleSet, readOnly = false) =>
  renderToStaticMarkup(
    createElement(ConfigPageDecisionRulesEditor, { value, readOnly, onChange: vi.fn() }),
  )

const renderConditionCount = (rules: DecisionRuleSet) => {
  const column = decisionColumns.find((candidate) => candidate.key === 'conditions')
  const row: DecisionConfig = { name: 'route', description: '', priority: 1, rules, modelRefs: [] }
  return renderToStaticMarkup(createElement('div', null, column?.render?.(row)))
}

describe('decision rules editor', () => {
  it('shows a single root condition instead of an unconditional match', () => {
    const view = renderRules(singleCondition, true)
    expect(view).toContain('Single condition')
    expect(view).toContain('jailbreak: prompt_injection')
    expect(view).toContain('On unknown: no_match')
    expect(view).not.toContain('Unconditional match')
  })

  it('renders predicates readably and keeps on_unknown on an unconditional view', () => {
    const view = renderRules(
      {
        operator: 'AND',
        conditions: [
          { type: 'classifier', name: 'intent', label: 'billing', predicate: { gte: 0.7, lt: 0.95 }, on_error: 'match' },
        ],
      },
      true,
    )
    expect(view).toContain('Label: billing')
    expect(view).toContain('Predicate: gte 0.7, lt 0.95')
    expect(view).toContain('On error: match')
    expect(view).not.toContain('{&quot;gte&quot;')
    expect(renderRules({ on_unknown: 'match' }, true)).toContain('On unknown: match')
  })

  it('keeps empty rules unconditional', () => {
    expect(renderRules({}, true)).toBe('<span>Unconditional match</span>')
  })

  it('edits through the shared decision rule editor with the on_unknown policy', () => {
    const editor = renderRules(singleCondition)
    // One canvas implementation: the shared editor's Result line and raw toggle.
    expect(editor).toContain('jailbreak(&quot;prompt_injection&quot;)')
    expect(editor).toContain('id="config-decision-rules-raw-toggle"')
    expect(editor).toContain('id="config-decision-rules-on-unknown"')
    expect(editor).toContain('<option value="no_match" selected="">No match</option>')
    expect(editor).toContain('When a signal evaluator fails')
    // The old per-condition form editor is gone.
    expect(editor).not.toContain('Root behavior')
    expect(editor).not.toContain('Add Condition')
  })

  it('shows an empty rule set as an unconditional canvas', () => {
    const editor = renderRules({})
    expect(editor).toContain('No condition — matches all requests')
    expect(editor).toContain('<option value="" selected="">Use condition policy</option>')
  })

  it('reports recipe-scope issues inline in the editor', () => {
    const editor = renderToStaticMarkup(
      createElement(ConfigPageDecisionRulesEditor, {
        value: { type: 'keyword', name: 'root-only' },
        onChange: vi.fn(),
        signalScope: ruleSignalScope([{ signalType: 'keyword', name: 'recipe-only' }], true),
      }),
    )
    expect(editor).toContain('signal keyword(&quot;root-only&quot;) is not declared in this recipe')
  })

  it('counts a single root condition in the decisions table', () => {
    expect(renderConditionCount(singleCondition)).toContain('1 condition')
    expect(renderConditionCount({})).toContain('0 conditions')
    expect(
      renderConditionCount({
        operator: 'OR',
        conditions: [
          { type: 'keyword', name: 'a' },
          { type: 'keyword', name: 'b' },
        ],
      }),
    ).toContain('2 conditions')
  })
})
