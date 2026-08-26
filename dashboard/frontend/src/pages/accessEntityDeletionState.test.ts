import { describe, expect, it } from 'vitest'
import {
  createAccessEntityDeletionTombstones,
  omitDeletedAccessEntities,
  rememberDeletedAccessEntity,
} from './accessEntityDeletionState'

describe('access entity deletion tombstones', () => {
  it('keeps a successful deletion hidden when an older list response arrives later', () => {
    const tombstones = createAccessEntityDeletionTombstones()
    rememberDeletedAccessEntity(tombstones, 'user', 'user-deleted')

    expect(
      omitDeletedAccessEntities(tombstones, 'user', [
        { id: 'user-deleted' },
        { id: 'user-visible' },
      ]),
    ).toEqual([{ id: 'user-visible' }])
  })

  it('isolates tombstones by entity kind', () => {
    const tombstones = createAccessEntityDeletionTombstones()
    rememberDeletedAccessEntity(tombstones, 'dashboard-member', 'shared-id')

    expect(omitDeletedAccessEntities(tombstones, 'dashboard-member', [{ id: 'shared-id' }])).toEqual(
      [],
    )
    expect(omitDeletedAccessEntities(tombstones, 'user', [{ id: 'shared-id' }])).toEqual([
      { id: 'shared-id' },
    ])
  })

  it('keeps a deleted API key out of a stale replica response', () => {
    const tombstones = createAccessEntityDeletionTombstones()
    rememberDeletedAccessEntity(tombstones, 'key', 'key-deleted')

    expect(
      omitDeletedAccessEntities(tombstones, 'key', [
        { id: 'key-deleted' },
        { id: 'key-active' },
      ]),
    ).toEqual([{ id: 'key-active' }])
  })
})
