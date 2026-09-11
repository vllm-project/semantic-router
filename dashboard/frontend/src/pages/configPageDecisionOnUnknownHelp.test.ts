import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

describe('decision editor on_unknown help', () => {
  it('explains every on_unknown option in plain language', () => {
    const source = readFileSync(
      new URL('./ConfigPageDecisionRulesEditor.tsx', import.meta.url),
      'utf8',
    )
    expect(source).toContain('When a signal evaluator fails')
    expect(source).toContain('no_match skips this decision')
    expect(source).toContain('match selects it')
    expect(source).toContain('fail_request rejects the request')
    expect(source).toContain('classifier condition&apos;s')
  })
})
