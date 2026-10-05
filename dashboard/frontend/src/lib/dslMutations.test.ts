import { describe, expect, it } from 'vitest'

import { serializeFields, updateRoute, updateSignal } from './dslMutations'

describe('DSL string literals written by Builder mutations', () => {
  it('escapes quotes, backslashes, and newlines in field values', () => {
    expect(
      serializeFields({
        description: 'Prompts with numbered list items such as "1. ..."',
        pattern: String.raw`(?m)^\s*\d+\.\s+\bstep\b`,
        note: 'first line\nsecond line',
      }),
    ).toBe(
      [
        String.raw`  description: "Prompts with numbered list items such as \"1. ...\""`,
        String.raw`  pattern: "(?m)^\\s*\\d+\\.\\s+\\bstep\\b"`,
        String.raw`  note: "first line\nsecond line"`,
      ].join('\n'),
    )
  })

  it('keeps an unchanged signal save parseable', () => {
    const source = [
      'SIGNAL structure numbered_steps {',
      String.raw`  description: "Prompts with numbered list items such as \"1. ...\""`,
      '}',
      '',
    ].join('\n')

    expect(
      updateSignal(source, 'structure', 'numbered_steps', {
        description: 'Prompts with numbered list items such as "1. ..."',
      }),
    ).toBe(source)
  })

  it('escapes route descriptions and model references', () => {
    const updated = updateRoute('ROUTE support {\n  PRIORITY 1\n}\n', 'support', {
      description: 'Answers "how do I" questions',
      priority: 5,
      models: [{ model: 'model "a"', effort: 'high' }],
      plugins: [],
    })

    expect(updated).toContain(
      String.raw`ROUTE support (description = "Answers \"how do I\" questions") {`,
    )
    expect(updated).toContain(String.raw`MODEL "model \"a\"" (effort = "high")`)
  })
})
