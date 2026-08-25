import { managementOperationRequest } from './managementApiContract'
import type { ManagementPage, ResourceDetail, SelfInferenceKey } from './routerManagementTypes'

export const SELF_KEY_PAGE_SIZE = 25
export const SELF_KEY_RENDER_LIMIT = 100
export const SELF_KEY_SEARCH_DEBOUNCE_MS = 250
export const SELF_KEY_SEARCH_LIMIT = 200

export interface SelfInferenceKeyPage {
  items: SelfInferenceKey[]
  nextCursor?: string
  hasMore: boolean
  pageSize: number
}

function assertSelfInferenceKey(value: unknown): SelfInferenceKey {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('Router returned an invalid inference key.')
  }
  const key = value as Partial<SelfInferenceKey>
  if (
    typeof key.keyId !== 'string' ||
    !key.keyId ||
    typeof key.name !== 'string' ||
    !key.owner ||
    (key.owner.type !== 'user' && key.owner.type !== 'team') ||
    typeof key.owner.id !== 'string' ||
    (key.contextTeamId !== undefined && typeof key.contextTeamId !== 'string') ||
    (key.expiresAt !== undefined && typeof key.expiresAt !== 'string')
  ) {
    throw new Error('Router returned an invalid inference key.')
  }
  return key as SelfInferenceKey
}

export function parseSelfInferenceKeyPage(payload: unknown): SelfInferenceKeyPage {
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    throw new Error('Router returned an invalid inference key list.')
  }
  const candidate = payload as Partial<ManagementPage<SelfInferenceKey>>
  if (
    !Array.isArray(candidate.data) ||
    !candidate.page ||
    typeof candidate.page !== 'object' ||
    typeof candidate.page.hasMore !== 'boolean' ||
    !Number.isSafeInteger(candidate.page.pageSize) ||
    (candidate.page.hasMore &&
      (typeof candidate.page.nextCursor !== 'string' || !candidate.page.nextCursor))
  ) {
    throw new Error('Router returned an invalid inference key list.')
  }
  return {
    items: activeSelfInferenceKeys(candidate.data.map(assertSelfInferenceKey)),
    nextCursor: candidate.page.nextCursor,
    hasMore: candidate.page.hasMore,
    pageSize: candidate.page.pageSize,
  }
}

function parseSelfInferenceKeyDetail(payload: unknown): SelfInferenceKey {
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    throw new Error('Router returned an invalid inference key.')
  }
  const candidate = payload as Partial<ResourceDetail<SelfInferenceKey>>
  return assertSelfInferenceKey(candidate.data)
}

export function activeSelfInferenceKeys(keys: SelfInferenceKey[]): SelfInferenceKey[] {
  const now = Date.now()
  return keys.filter((key) => {
    if (!key.expiresAt) return true
    const expiresAt = Date.parse(key.expiresAt)
    return Number.isFinite(expiresAt) && expiresAt > now
  })
}

export function selfInferenceKeyListQuery({
  cursor,
  search = '',
  pageSize = SELF_KEY_PAGE_SIZE,
}: {
  cursor?: string
  search?: string
  pageSize?: number
} = {}): URLSearchParams {
  const query = new URLSearchParams({
    pageSize: String(Math.max(1, Math.min(pageSize, SELF_KEY_PAGE_SIZE))),
  })
  const normalizedSearch = search.trim().slice(0, SELF_KEY_SEARCH_LIMIT)
  if (normalizedSearch) query.set('search', normalizedSearch)
  if (cursor) query.set('cursor', cursor)
  return query
}

export async function fetchSelfInferenceKeyPage(
  options: { cursor?: string; search?: string; pageSize?: number } = {},
  signal?: AbortSignal,
): Promise<SelfInferenceKeyPage> {
  return parseSelfInferenceKeyPage(
    await managementOperationRequest('getSelfInferenceKeys', {
      query: selfInferenceKeyListQuery(options),
      signal,
    }),
  )
}

export async function fetchSelfInferenceKey(
  keyId: string,
  signal?: AbortSignal,
): Promise<SelfInferenceKey> {
  return parseSelfInferenceKeyDetail(
    await managementOperationRequest('getSelfInferenceKeysByKeyId', {
      pathParameters: { keyId },
      signal,
    }),
  )
}

export async function restoreSelfInferenceKeySelection(
  initialKeys: SelfInferenceKey[],
  storedKeyId: string,
  lookup: (keyId: string) => Promise<SelfInferenceKey>,
  signal?: AbortSignal,
): Promise<SelfInferenceKey | null> {
  const stored = initialKeys.find((key) => key.keyId === storedKeyId)
  if (stored) return stored
  if (storedKeyId) {
    try {
      const resolved = await lookup(storedKeyId)
      if (activeSelfInferenceKeys([resolved]).length) return resolved
    } catch (cause) {
      if (signal?.aborted) throw cause
      // A stale, revoked, or out-of-scope saved key falls back to current eligibility.
    }
  }
  return initialKeys[0] ?? null
}

export function mergeSelfInferenceKeyPages(
  current: SelfInferenceKey[],
  incoming: SelfInferenceKey[],
  limit = SELF_KEY_RENDER_LIMIT,
): SelfInferenceKey[] {
  const byID = new Map(current.map((key) => [key.keyId, key]))
  for (const key of incoming) byID.set(key.keyId, key)
  return [...byID.values()].slice(0, limit)
}
