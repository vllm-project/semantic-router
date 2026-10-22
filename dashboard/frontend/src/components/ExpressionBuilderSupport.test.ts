import { describe, expect, it } from 'vitest'

import {
  addChildAtPath,
  insertAtPath,
  makeDragNode,
  removeAtPath,
  replaceAtPath,
  type RuleLeaf,
  type RuleNode,
} from './ExpressionBuilderSupport'

const metadataLeaf: RuleLeaf = {
  id: 'leaf-a',
  signalType: 'classifier',
  signalName: 'intent',
  label: 'code',
  predicate: { gte: 0.7 },
  onError: 'match',
  extra: { x_note: 'kept' },
}

const tree = (): RuleNode => ({
  id: 'root',
  operator: 'AND',
  conditions: [
    metadataLeaf,
    { id: 'or', operator: 'OR', conditions: [{ id: 'b', signalType: 'keyword', signalName: 'b' }, { id: 'c', signalType: 'keyword', signalName: 'c' }] },
  ],
})

describe('ExpressionBuilderSupport tree operations', () => {
  it('keeps ids and extra fields on untouched nodes', () => {
    const replaced = replaceAtPath(tree(), [1, 0], { id: 'b2', signalType: 'keyword', signalName: 'b2' }) as {
      id: string
      conditions: RuleNode[]
    }
    expect(replaced.id).toBe('root')
    expect(replaced.conditions[0]).toBe(metadataLeaf)
    expect(replaced.conditions[1].id).toBe('or')

    const inserted = insertAtPath(tree(), [], 1, { signalType: 'keyword', signalName: 'd' }) as { conditions: RuleNode[] }
    expect(inserted.conditions[0]).toEqual(metadataLeaf)
  })

  it('does not collapse a group left with one child', () => {
    const removed = removeAtPath(tree(), [1, 0])
    expect(removed).toEqual({
      ...tree(),
      conditions: [metadataLeaf, { id: 'or', operator: 'OR', conditions: [{ id: 'c', signalType: 'keyword', signalName: 'c' }] }],
    })
    expect(removeAtPath(tree(), [0])).toMatchObject({ id: 'root', operator: 'AND', conditions: [{ id: 'or' }] })
  })

  it('removes a group that loses its last child', () => {
    const single: RuleNode = { operator: 'AND', conditions: [{ operator: 'NOT', conditions: [metadataLeaf] }, metadataLeaf] }
    expect(removeAtPath(single, [0, 0])).toEqual({ operator: 'AND', conditions: [metadataLeaf] })
    expect(removeAtPath(single, [])).toBeNull()
  })

  it('gives created nodes ids', () => {
    expect(makeDragNode({ kind: 'signal', signalType: 'keyword', signalName: 'x' })?.id).toMatch(/^r\d+$/)
    expect(makeDragNode({ kind: 'operator', operator: 'NOT' })?.id).toMatch(/^r\d+$/)
    const wrapped = addChildAtPath(metadataLeaf, [], { signalType: 'keyword', signalName: 'x' })
    expect(wrapped.id).toMatch(/^r\d+$/)
    expect((wrapped as { conditions: RuleNode[] }).conditions[0]).toBe(metadataLeaf)
  })
})
