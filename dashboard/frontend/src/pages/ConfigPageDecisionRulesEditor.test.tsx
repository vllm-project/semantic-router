import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import { TooltipProvider } from '../ui'
import ConfigPageDecisionRulesEditor from './ConfigPageDecisionRulesEditor'
import { countRuleConditions, ruleExpression, ruleLeaves } from './configPageDecisionRuleSupport'
import { decisionColumns } from './configPageDecisionTable'
import type { DecisionConfig, DecisionRuleSet } from './configPageSupport'

const singleCondition: DecisionRuleSet = {
  type: 'jailbreak',
  name: 'prompt_injection',
  on_unknown: 'no_match',
}

const renderRules = (value: DecisionRuleSet, readOnly = false) =>
  renderToStaticMarkup(
    createElement(
      TooltipProvider,
      null,
      createElement(ConfigPageDecisionRulesEditor, { value, readOnly, onChange: vi.fn() }),
    ),
  )

const renderWhen = (rules: DecisionRuleSet) => {
  const column = decisionColumns({ models: [] }).find((candidate) => candidate.id === 'when')
  const row: DecisionConfig = { name: 'route', description: '', priority: 1, rules, modelRefs: [] }
  return renderToStaticMarkup(
    createElement(TooltipProvider, null, createElement('div', null, column?.cell(row))),
  )
}

describe('decision rules editor', () => {
  it('keeps a single root condition instead of an unconditional match', () => {
    const view = renderRules(singleCondition, true)
    expect(view).toContain('Single condition')
    expect(view).toContain('value="prompt_injection"')
    expect(view).toContain('no_match')
    expect(view).not.toContain('Always matches')

    const editor = renderRules(singleCondition)
    expect(editor).toContain('Single condition')
    expect(editor).toContain('aria-label="Signal type"')
    expect(editor).toContain('value="prompt_injection"')
  })

  it('shows empty rules as always matching', () => {
    expect(renderRules({}, true)).toContain('Always matches')
    expect(renderRules({})).toContain('Add condition')
    expect(renderRules({})).not.toContain('Single condition')
  })

  it('renders nested All, Any and NOT groups with their conditions', () => {
    const markup = renderRules({
      operator: 'AND',
      conditions: [
        { type: 'domain', name: 'business' },
        {
          operator: 'OR',
          conditions: [
            { type: 'keyword', name: 'urgent_keywords' },
            { type: 'complexity', name: 'needs_reasoning', label: 'hard' },
          ],
        },
        { operator: 'NOT', conditions: [{ type: 'jailbreak', name: 'prompt_injection' }] },
      ],
    })
    expect(markup).toContain('All of')
    expect(markup).toContain('Any of')
    expect(markup).toContain('value="urgent_keywords"')
    expect(markup).toContain('value="hard"')
    expect(markup).toContain('aria-pressed="true"')
  })

  it('counts conditions through nested groups and shows them as chips in the table', () => {
    expect(countRuleConditions(singleCondition)).toBe(1)
    expect(countRuleConditions({})).toBe(0)
    expect(
      countRuleConditions({
        operator: 'OR',
        conditions: [
          { type: 'keyword', name: 'a' },
          { operator: 'NOT', conditions: [{ type: 'keyword', name: 'b' }] },
        ],
      }),
    ).toBe(2)
    expect(
      ruleLeaves({ operator: 'NOT', conditions: [{ type: 'keyword', name: 'b' }] })[0].negated,
    ).toBe(true)
    expect(renderWhen({})).toContain('Always')
    expect(
      renderWhen({ operator: 'AND', conditions: [{ type: 'domain', name: 'law' }] }),
    ).toContain('law')
  })

  it('writes the rule tree the way the DSL decompiler does', () => {
    expect(
      ruleExpression({
        operator: 'AND',
        conditions: [
          { type: 'domain', name: 'computer science' },
          { type: 'complexity', name: 'needs_reasoning', label: 'hard' },
          {
            operator: 'OR',
            conditions: [
              { type: 'keyword', name: 'a' },
              { type: 'systemone', name: 'risk', predicate: { gte: 0.8 }, on_error: 'match' },
            ],
          },
          {
            operator: 'NOT',
            conditions: [
              {
                operator: 'AND',
                conditions: [
                  { type: 'pii', name: 'p' },
                  { type: 'pii', name: 'q' },
                ],
              },
            ],
          },
        ],
      }),
    ).toBe(
      'domain("computer science") AND complexity("needs_reasoning", label: "hard") AND (keyword("a") OR systemone("risk", predicate: {gte: 0.8}, on_error: "match")) AND NOT (pii("p") AND pii("q"))',
    )
    expect(ruleExpression({})).toBe('always')
  })
})
