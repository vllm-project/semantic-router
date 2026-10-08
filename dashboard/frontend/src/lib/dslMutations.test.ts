import { describe, expect, it } from 'vitest'

import {
  deleteRoute,
  deleteSignal,
  serializeFields,
  updateModel,
  updatePlugin,
  updateRoute,
  updateSignal,
} from './dslMutations'

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

describe('Builder mutations of blocks whose headers contain DSL strings', () => {
  it('updates and deletes a signal whose name needs quotes', () => {
    const source =
      'SIGNAL domain "computer science" {\n  description: "Computer science prompts."\n}\n'

    expect(
      updateSignal(source, 'domain', 'computer science', { description: 'Programming prompts.' }),
    ).toBe('SIGNAL domain "computer science" {\n  description: "Programming prompts."\n}\n')
    expect(deleteSignal(source, 'domain', 'computer science')).toBe('')
  })

  it('updates and deletes a route whose description contains parentheses or braces', () => {
    const description = 'Answer coding questions (debugging and {reviews}).'
    const source = `ROUTE coding_help (description = "${description}") {\n  PRIORITY 100\n}\n`
    const updated = updateRoute(source, 'coding_help', {
      description,
      priority: 150,
      models: [],
      plugins: [],
    })

    expect(updated).toContain(
      `ROUTE coding_help (description = "${description}") {\n  PRIORITY 150`,
    )
    expect(updated).not.toContain('PRIORITY 100')
    expect(deleteRoute(source, 'coding_help')).toBe('')
  })

  it('updates a route and a plugin whose names need quotes', () => {
    expect(
      updateRoute('ROUTE "coding help" {\n  PRIORITY 1\n}\n', 'coding help', {
        priority: 2,
        models: [],
        plugins: [],
      }),
    ).toContain('ROUTE "coding help" {\n  PRIORITY 2')
    expect(
      updatePlugin('PLUGIN "team cache" semantic_cache {\n}\n', 'team cache', 'semantic_cache', {
        enabled: true,
      }),
    ).toBe('PLUGIN "team cache" semantic_cache {\n  enabled: true\n}\n')
  })

  it('quotes route plugin references whose names need quotes', () => {
    const updated = updateRoute('ROUTE support {\n  PRIORITY 1\n}\n', 'support', {
      priority: 1,
      models: [],
      plugins: [
        { name: 'team cache' },
        { name: 'team guard', fields: { enabled: true } },
        { name: 'semantic_cache' },
      ],
    })

    expect(updated).toContain('  PLUGIN "team cache"\n')
    expect(updated).toContain('  PLUGIN "team guard" {\n    enabled: true\n  }\n')
    expect(updated).toContain('  PLUGIN semantic_cache\n')
  })

  it('writes names that are DSL identifiers without quotes', () => {
    expect(
      updateSignal('SIGNAL keyword urgent-requests {\n}\n', 'keyword', 'urgent-requests', {
        keywords: ['urgent'],
      }),
    ).toBe('SIGNAL keyword urgent-requests {\n  keywords: ["urgent"]\n}\n')
    expect(updateModel('MODEL qwen3-8b {\n}\n', 'qwen3-8b', { modality: 'text' })).toBe(
      'MODEL qwen3-8b {\n  modality: "text"\n}\n',
    )
  })
})
