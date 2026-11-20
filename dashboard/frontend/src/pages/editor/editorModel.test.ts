import { describe, expect, it } from 'vitest'

import type { ASTProgram } from '@/types/dsl'
import {
  blockEndLine,
  entityAtLine,
  listEditorEntities,
  selectionKey,
  worstDiagnosticByEntity,
} from './editorModel'

const source = [
  'MODEL "qwen3-32b" {', // 1
  '  description: "Big {model}"', // 2
  '}', // 3
  '', // 4
  'SIGNAL keyword urgent {', // 5
  '  keywords: ["asap", "}"]', // 6
  '}', // 7
  '', // 8
  'ROUTE urgent_route {', // 9
  '  PRIORITY 200', // 10
  '  WHEN keyword("urgent")', // 11
  '  MODEL "qwen3-32b"', // 12
  '}', // 13
].join('\n')

const at = (Line: number) => ({ Line, Column: 1 })
const ast: ASTProgram = {
  models: [{ name: 'qwen3-32b', fields: {}, pos: at(1) }],
  signals: [{ signalType: 'keyword', name: 'urgent', fields: {}, pos: at(5) }],
  routes: [
    { name: 'urgent_route', priority: 200, when: null, models: [], plugins: [], pos: at(9) },
  ],
  plugins: [],
}

describe('config editor entities', () => {
  it('finds the end of a block while ignoring braces in strings', () => {
    const lines = source.split('\n')
    expect(blockEndLine(lines, 1)).toBe(3)
    expect(blockEndLine(lines, 5)).toBe(7)
    expect(blockEndLine(lines, 9)).toBe(13)
  })

  it('lists entities with the lines they span and stable keys', () => {
    const entities = listEditorEntities(ast, source)
    expect(entities.map((entity) => [entity.key, entity.line, entity.endLine])).toEqual([
      ['model::qwen3-32b', 1, 3],
      ['signal:keyword:urgent', 5, 7],
      ['route::urgent_route', 9, 13],
    ])
    expect(selectionKey({ kind: 'signal', name: 'urgent', signalType: 'keyword' })).toBe(
      'signal:keyword:urgent',
    )
  })

  it('maps diagnostics to the entity that owns their line', () => {
    const entities = listEditorEntities(ast, source)
    expect(entityAtLine(entities, 11)?.name).toBe('urgent_route')
    expect(entityAtLine(entities, 4)).toBeNull()
    const worst = worstDiagnosticByEntity(entities, [
      { level: 'warning', message: 'w', line: 10, column: 3 },
      { level: 'error', message: 'e', line: 12, column: 3 },
      { level: 'constraint', message: 'c', line: 6, column: 3 },
    ])
    expect(worst.get('route::urgent_route')).toBe('error')
    expect(worst.get('signal:keyword:urgent')).toBe('constraint')
    expect(worst.has('model::qwen3-32b')).toBe(false)
  })
})

describe('plain-text deploy diff', () => {
  it('shows the changed region with context and counts', async () => {
    const { changedRegion } = await import('./editorDiff')
    const before = ['a', 'b', 'c', 'd', 'e', 'f'].join('\n')
    const after = ['a', 'b', 'c', 'X', 'e', 'f'].join('\n')
    const diff = changedRegion(before, after, 1)
    expect(diff.added).toBe(1)
    expect(diff.removed).toBe(1)
    expect(diff.lines.map((line) => `${line.kind}:${line.text}`)).toEqual([
      'context:c',
      'remove:d',
      'add:X',
      'context:e',
    ])
    expect(changedRegion(before, before).lines).toEqual([])
  })
})
