import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

describe('decision editor on_unknown help', () => {
  it('explains every on_unknown option in plain language', () => {
    const source = readFileSync(
      new URL('./ConfigPageDecisionsSection.tsx', import.meta.url),
      'utf8',
    )
    const field = source.slice(source.indexOf("name: 'on_unknown'"))
    const description = field.slice(0, field.indexOf('},'))
    expect(description).toContain('no_match skips this decision')
    expect(description).toContain('match selects it')
    expect(description).toContain('fail_request rejects the request')
    expect(description).toContain('Empty keeps condition on_error')
  })
})
