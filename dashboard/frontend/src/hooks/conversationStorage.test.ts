import { describe, expect, it } from 'vitest'

import {
  normalizeStoredConversations,
  pruneStoredConversations,
  type StoredConversation,
} from './conversationStorage'

const conversation = (
  id: string,
  updatedAt: number,
  payload: string[] = [`message ${id}`]
): StoredConversation<string[]> => ({
  id,
  createdAt: updatedAt - 1,
  updatedAt,
  payload,
})

describe('conversationStorage', () => {
  it('filters invalid restored conversation entries', () => {
    const restored = normalizeStoredConversations([
      conversation('valid', 3),
      { id: '', createdAt: 1, updatedAt: 1, payload: [] },
      { id: 'missing-payload', createdAt: 1, updatedAt: 1 },
      { id: 'bad-created-at', createdAt: '1', updatedAt: 1, payload: [] },
      'not-a-conversation',
    ])

    expect(restored).toEqual([conversation('valid', 3)])
  })

  it('caps restored conversations by newest update time', () => {
    const restored = normalizeStoredConversations([
      conversation('old', 1),
      conversation('newest', 3),
      conversation('middle', 2),
    ], { maxConversations: 2 })

    expect(restored.map(item => item.id)).toEqual(['newest', 'middle'])
  })

  it('deduplicates restored conversations by id after trimming ids', () => {
    const restored = normalizeStoredConversations([
      conversation(' same ', 3),
      conversation('same', 4),
    ])

    expect(restored).toHaveLength(1)
    expect(restored[0].id).toBe('same')
    expect(restored[0].updatedAt).toBe(4)
  })

  it('accepts a payload validator for typed callers', () => {
    const restored = normalizeStoredConversations<string[]>(
      [
        conversation('messages', 2),
        { id: 'bad-payload', createdAt: 1, updatedAt: 1, payload: 'not-an-array' },
      ],
      {},
      (payload): payload is string[] => Array.isArray(payload)
    )

    expect(restored).toEqual([conversation('messages', 2)])
  })

  it('prunes saved conversations before persistence', () => {
    const pruned = pruneStoredConversations([
      conversation('first', 1),
      conversation('third', 3),
      conversation('second', 2),
    ], { maxConversations: 2 })

    expect(pruned.map(item => item.id)).toEqual(['third', 'second'])
  })
})
