export type DeletableAccessEntityKind =
  | 'user'
  | 'team'
  | 'key'
  | 'group'
  | 'budget'
  | 'dashboard-member'

export type AccessEntityDeletionTombstones = Record<DeletableAccessEntityKind, Set<string>>

export function createAccessEntityDeletionTombstones(): AccessEntityDeletionTombstones {
  return {
    user: new Set(),
    team: new Set(),
    key: new Set(),
    group: new Set(),
    budget: new Set(),
    'dashboard-member': new Set(),
  }
}

export function rememberDeletedAccessEntity(
  tombstones: AccessEntityDeletionTombstones,
  kind: DeletableAccessEntityKind,
  id: string,
) {
  tombstones[kind].add(id)
}

export function omitDeletedAccessEntities<T extends { id: string }>(
  tombstones: AccessEntityDeletionTombstones,
  kind: DeletableAccessEntityKind,
  items: readonly T[],
): T[] {
  const deleted = tombstones[kind]
  return deleted.size ? items.filter((item) => !deleted.has(item.id)) : [...items]
}
