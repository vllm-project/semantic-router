import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

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

    const editor = renderRules(singleCondition)
    expect(editor).toContain('<option value="CONDITION" selected="">Single condition</option>')
    expect(editor).toContain('<option value="jailbreak" selected="">')
    expect(editor).toContain('value="prompt_injection"')
    expect(editor).toContain('<option value="no_match" selected="">No match</option>')
  })

  it('keeps empty rules unconditional', () => {
    expect(renderRules({}, true)).toBe('<span>Unconditional match</span>')
    const editor = renderRules({})
    expect(editor).toContain('<option value="" selected="">Unconditional match</option>')
    expect(editor).not.toContain('Single condition')
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
