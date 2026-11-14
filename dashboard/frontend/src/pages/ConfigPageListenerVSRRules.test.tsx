import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import type { Listener } from '../types/config'
import { ListenerVSRRules } from './ConfigPageListenerVSRRules'
import {
  describeTry,
  vsrRuleBinding,
  vsrRuleBlocked,
  vsrRuleChoices,
  vsrRuleRecipes,
  withVSRRule,
  withVSRRulesRecipe,
} from './configPageListenerVSRSupport'
import type { ConfigData } from './configPageSupport'

const config = {
  version: 'v0.5',
  recipes: [
    {
      name: 'agents',
      routing: {
        signals: {
          keywords: [
            { name: 'prod_database', description: 'touches the production database', operator: 'OR', method: 'regex', keywords: ['prod-db'] },
            { name: 'billing', operator: 'AND', keywords: ['invoice', 'refund'] },
            { name: 'near', operator: 'OR', keywords: ['secret'], fuzzy_match: true },
            { name: 'employee_id', operator: 'OR', method: 'REGEX', keywords: ['E-\\d{6}'] },
            { name: 'loose', operator: 'any', keywords: ['secret'] },
          ],
          systemone: [
            { name: 'deletes_data', deployment: 'vela3', question: { type: 'noul', instructions: 'Deletes data?' } },
            { name: 'kind', deployment: 'vela3', question: { type: 'choice', instructions: 'Kind?' } },
            { name: 'account', deployment: 'vela3', question: { type: 'span', instructions: 'Accounts?' } },
          ],
        },
      },
    },
  ],
} as unknown as ConfigData

const listener: Listener = { name: 'agents', address: '0.0.0.0', port: 8899, apis: { systemone: {} } }
const names = (app: 'guard' | 'redact') => vsrRuleChoices(config, 'agents', app).map((choice) => choice.name)

describe('vSR rules of a listener', () => {
  it('offers guard a rule that says yes or no and redact one that finds text', () => {
    expect(vsrRuleRecipes(config)).toEqual(['agents'])
    expect(names('guard')).toEqual(['prod_database', 'billing', 'near', 'employee_id', 'loose', 'deletes_data'])
    expect(names('redact')).toEqual(['prod_database', 'employee_id', 'account'])
  })

  it('keeps a bound rule the recipe no longer offers, and blocks what the Router would refuse', () => {
    const gone = { recipe: 'agents', redact: [{ type: 'keyword' as const, name: 'retired' }] }
    expect(vsrRuleChoices(config, 'agents', 'redact', { rules: gone }).map((choice) => choice.name)).toContain('retired')
    const deletes = vsrRuleChoices(config, 'agents', 'guard').find((choice) => choice.name === 'deletes_data')!
    const sameName = { rules: { recipe: 'agents', guard: [{ type: 'keyword' as const, name: 'deletes_data' }] } }
    expect(vsrRuleBlocked(sameName, 'guard', deletes)).toBe('another rule named deletes_data is bound')
    const full = { rules: { recipe: 'agents', guard: Array.from({ length: 16 }, (_, i) => ({ type: 'keyword' as const, name: `r${i}` })) } }
    expect(vsrRuleBlocked(full, 'guard', deletes)).toBe('guard binds 16 rules at most')
    expect(vsrRuleBlocked(undefined, 'guard', deletes)).toBe(undefined)
  })

  it('binds, rebinds and lets go of a rule, keeping the other fields', () => {
    const prod = vsrRuleChoices(config, 'agents', 'guard')[0]
    const required = withVSRRule({ guard: true }, 'agents', 'guard', prod, 'required')
    expect(required).toEqual({ guard: true, rules: { recipe: 'agents', guard: [{ type: 'keyword', name: 'prod_database', enforce: 'required' }] } })
    expect(vsrRuleBinding(required, 'guard', prod)).toBe('required')
    const advisory = withVSRRule(required, 'agents', 'guard', prod, 'advisory')
    expect(advisory?.rules?.guard).toEqual([{ type: 'keyword', name: 'prod_database' }])
    expect(withVSRRule(advisory, 'agents', 'guard', prod, 'off')).toEqual({ guard: true })
    expect(withVSRRulesRecipe(required, '')).toEqual({ guard: true })
    expect(withVSRRulesRecipe(required, 'other')).toEqual({ guard: true, rules: { recipe: 'other' } })
  })

  it('says what a try found, in the sample\'s own characters', () => {
    expect(describeTry('guard', 'psql', { parts: [{ matches: ['prod_database'], spans: [], unchecked: [] }] })).toBe('Held by prod_database.')
    expect(
      describeTry('redact', '😀 E-104233', {
        parts: [{ matches: [], spans: [{ start: 2, end: 10, rule: 'employee_id' }], unchecked: [{ rule: 'account', reason: 'timed out' }] }],
      }),
    ).toBe('Hides "E-104233" (employee_id). Not checked: account (timed out).')
  })

  it('draws a row per rule with its binding pressed', () => {
    const markup = renderToStaticMarkup(
      createElement(ListenerVSRRules, {
        config,
        listener,
        vsr: { rules: { recipe: 'agents', guard: [{ type: 'keyword', name: 'prod_database', enforce: 'required' }] } },
        disabled: false,
        onChange: () => undefined,
      }),
    )
    expect(markup).toContain('Rules of your own')
    expect(markup).toContain('touches the production database')
    expect(markup).toMatch(/aria-label="Guard risky commands: prod_database">.*?aria-pressed="true"[^>]*>Required/)
    expect(markup).toContain('Try it')
  })

  it('folds the rules left Off once one is bound, and shows all while none is', () => {
    const markup = renderToStaticMarkup(
      createElement(ListenerVSRRules, {
        config,
        listener,
        vsr: { rules: { recipe: 'agents', guard: [{ type: 'keyword', name: 'prod_database', enforce: 'required' }] } },
        disabled: false,
        onChange: () => undefined,
      }),
    )
    expect(markup).toContain('aria-label="Guard risky commands: prod_database"')
    expect(markup).not.toContain('aria-label="Guard risky commands: billing"')
    expect(markup).toContain('Show 5 more rules of agents')
    // Nothing is bound to redact, so every rule that fits it is listed.
    expect(markup).toContain('aria-label="Hide personal data: account"')
  })
})
