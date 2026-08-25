import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react'

import { useAuth } from './AuthContext'
import { managementOperationRequest } from '../utils/managementApiContract'
import type { ManagementPage, SelfInferenceKey } from '../utils/routerManagementTypes'
import {
  canReadKeyScopedRouting,
  canReadRouting,
  canUseDelegatedInference,
} from '../utils/accessControl'
import {
  fetchKeyScopedRoutingCatalog,
  keyScopedCatalogSnapshot,
  type KeyScopedRoutingCatalog,
} from '../utils/keyScopedRoutingCatalog'
import type { ManagedRoutingSnapshot } from '../utils/managedRoutingSnapshot'

export type InferenceRoutingAccessStatus = 'idle' | 'loading' | 'ready' | 'unavailable' | 'error'

interface InferenceRoutingAccessValue {
  keys: SelfInferenceKey[]
  keysStatus: InferenceRoutingAccessStatus
  keysError: string | null
  selectedKey: SelfInferenceKey | null
  selectedKeyId: string
  setSelectedKeyId: (keyId: string) => void
  retryKeys: () => void
  catalog: KeyScopedRoutingCatalog | null
  catalogSnapshot: ManagedRoutingSnapshot | null
  catalogStatus: InferenceRoutingAccessStatus
  catalogError: string | null
  retryCatalog: () => void
  usesKeyScopedCatalog: boolean
}

const InferenceRoutingAccessContext = createContext<InferenceRoutingAccessValue | undefined>(
  undefined,
)

function assertKeyPage(payload: unknown): ManagementPage<SelfInferenceKey> {
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    throw new Error('Router returned an invalid inference key list.')
  }
  const candidate = payload as Partial<ManagementPage<SelfInferenceKey>>
  if (
    !Array.isArray(candidate.data) ||
    !candidate.page ||
    typeof candidate.page !== 'object' ||
    typeof candidate.page.hasMore !== 'boolean' ||
    !Number.isSafeInteger(candidate.page.pageSize)
  ) {
    throw new Error('Router returned an invalid inference key list.')
  }
  for (const key of candidate.data) {
    if (
      !key ||
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
  }
  return candidate as ManagementPage<SelfInferenceKey>
}

function activeKeys(keys: SelfInferenceKey[]): SelfInferenceKey[] {
  const now = Date.now()
  return keys.filter((key) => {
    if (!key.expiresAt) return true
    const expiresAt = Date.parse(key.expiresAt)
    return Number.isFinite(expiresAt) && expiresAt > now
  })
}

function selectionStorageKey(namespaceId?: string, userId?: string): string {
  return namespaceId && userId ? `vllm-sr:routing-key:${namespaceId}:${userId}` : ''
}

function readStoredSelection(storageKey: string): string {
  if (!storageKey || typeof window === 'undefined') return ''
  try {
    return window.localStorage.getItem(storageKey) ?? ''
  } catch {
    return ''
  }
}

function writeStoredSelection(storageKey: string, keyId: string): void {
  if (!storageKey || typeof window === 'undefined') return
  try {
    window.localStorage.setItem(storageKey, keyId)
  } catch {
    // A blocked storage area must not prevent the in-memory selection.
  }
}

function errorMessage(cause: unknown, fallback: string): string {
  return cause instanceof Error && cause.message.trim() ? cause.message : fallback
}

export function InferenceRoutingAccessProvider({ children }: { children: ReactNode }) {
  const { user } = useAuth()
  const shouldLoadKeys = canUseDelegatedInference(user)
  const usesKeyScopedCatalog = canReadKeyScopedRouting(user) && !canReadRouting(user)
  const storageKey = selectionStorageKey(user?.managementNamespaceId, user?.managementUserId)
  const [keys, setKeys] = useState<SelfInferenceKey[]>([])
  const [selectedKeyId, setSelectedKeyIdState] = useState('')
  const [keysStatus, setKeysStatus] = useState<InferenceRoutingAccessStatus>('idle')
  const [keysError, setKeysError] = useState<string | null>(null)
  const [keysAttempt, setKeysAttempt] = useState(0)
  const [catalog, setCatalog] = useState<KeyScopedRoutingCatalog | null>(null)
  const [catalogStatus, setCatalogStatus] = useState<InferenceRoutingAccessStatus>('idle')
  const [catalogError, setCatalogError] = useState<string | null>(null)
  const [catalogAttempt, setCatalogAttempt] = useState(0)

  const retryKeys = useCallback(() => setKeysAttempt((current) => current + 1), [])
  const retryCatalog = useCallback(() => setCatalogAttempt((current) => current + 1), [])

  useEffect(() => {
    setKeys([])
    setSelectedKeyIdState('')
    setKeysError(null)
    setCatalog(null)
    setCatalogError(null)
    if (!shouldLoadKeys || user?.managementIdentityStatus !== 'ready') {
      setKeysStatus('idle')
      setCatalogStatus('idle')
      return
    }

    const controller = new AbortController()
    setKeysStatus('loading')
    void managementOperationRequest('getSelfInferenceKeys', {
      query: new URLSearchParams({ pageSize: '100' }),
      signal: controller.signal,
    })
      .then(assertKeyPage)
      .then((page) => {
        if (controller.signal.aborted) return
        const eligible = activeKeys(page.data)
        const stored = readStoredSelection(storageKey)
        const selected = eligible.find((key) => key.keyId === stored) ?? eligible[0] ?? null
        setKeys(eligible)
        setSelectedKeyIdState(selected?.keyId ?? '')
        if (selected) writeStoredSelection(storageKey, selected.keyId)
        setKeysStatus(selected ? 'ready' : 'unavailable')
      })
      .catch((cause: unknown) => {
        if (controller.signal.aborted) return
        setKeys([])
        setSelectedKeyIdState('')
        setKeysError(errorMessage(cause, 'API keys are unavailable.'))
        setKeysStatus('error')
      })
    return () => controller.abort()
  }, [keysAttempt, shouldLoadKeys, storageKey, user?.managementIdentityStatus])

  const setSelectedKeyId = useCallback(
    (keyId: string) => {
      if (!keys.some((key) => key.keyId === keyId)) return
      setSelectedKeyIdState(keyId)
      writeStoredSelection(storageKey, keyId)
    },
    [keys, storageKey],
  )

  const selectedKey = useMemo(
    () => keys.find((key) => key.keyId === selectedKeyId) ?? null,
    [keys, selectedKeyId],
  )

  useEffect(() => {
    setCatalog(null)
    setCatalogError(null)
    if (!usesKeyScopedCatalog) {
      setCatalogStatus('idle')
      return
    }
    if (keysStatus === 'loading' || keysStatus === 'idle') {
      setCatalogStatus('loading')
      return
    }
    if (!selectedKey) {
      if (keysStatus === 'error') {
        setCatalogError(keysError ?? 'API keys are unavailable.')
        setCatalogStatus('error')
      } else {
        setCatalogStatus('unavailable')
      }
      return
    }

    const controller = new AbortController()
    setCatalogStatus('loading')
    void fetchKeyScopedRoutingCatalog(selectedKey.keyId, controller.signal)
      .then((nextCatalog) => {
        if (controller.signal.aborted) return
        if (nextCatalog.keyId !== selectedKey.keyId) {
          throw new Error('Router returned routing access for another API key.')
        }
        setCatalog(nextCatalog)
        setCatalogStatus('ready')
      })
      .catch((cause: unknown) => {
        if (controller.signal.aborted) return
        setCatalogError(errorMessage(cause, 'Routing access is unavailable.'))
        setCatalogStatus('error')
      })
    return () => controller.abort()
  }, [catalogAttempt, keysError, keysStatus, selectedKey, usesKeyScopedCatalog])

  const catalogSnapshot = useMemo(
    () => (catalog ? keyScopedCatalogSnapshot(catalog) : null),
    [catalog],
  )

  const value = useMemo<InferenceRoutingAccessValue>(
    () => ({
      keys,
      keysStatus,
      keysError,
      selectedKey,
      selectedKeyId,
      setSelectedKeyId,
      retryKeys,
      catalog,
      catalogSnapshot,
      catalogStatus,
      catalogError,
      retryCatalog,
      usesKeyScopedCatalog,
    }),
    [
      catalog,
      catalogError,
      catalogSnapshot,
      catalogStatus,
      keys,
      keysError,
      keysStatus,
      retryCatalog,
      retryKeys,
      selectedKey,
      selectedKeyId,
      setSelectedKeyId,
      usesKeyScopedCatalog,
    ],
  )

  return (
    <InferenceRoutingAccessContext.Provider value={value}>
      {children}
    </InferenceRoutingAccessContext.Provider>
  )
}

export function useInferenceRoutingAccess(): InferenceRoutingAccessValue {
  const value = useContext(InferenceRoutingAccessContext)
  if (!value) {
    throw new Error('useInferenceRoutingAccess must be used within InferenceRoutingAccessProvider')
  }
  return value
}
