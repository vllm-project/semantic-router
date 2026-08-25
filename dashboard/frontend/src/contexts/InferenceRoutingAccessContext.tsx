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
import type { SelfInferenceKey } from '../utils/routerManagementTypes'
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
import {
  activeSelfInferenceKeys,
  fetchSelfInferenceKey,
  fetchSelfInferenceKeyPage,
  restoreSelfInferenceKeySelection,
} from '../utils/selfInferenceKeys'

export type InferenceRoutingAccessStatus = 'idle' | 'loading' | 'ready' | 'unavailable' | 'error'

interface InferenceRoutingAccessValue {
  keys: SelfInferenceKey[]
  keysStatus: InferenceRoutingAccessStatus
  keysError: string | null
  keysHasMore: boolean
  selectedKey: SelfInferenceKey | null
  selectedKeyId: string
  setSelectedKeyId: (keyId: string) => void
  selectKey: (key: SelfInferenceKey) => void
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
  const [keysHasMore, setKeysHasMore] = useState(false)
  const [selectedKey, setSelectedKey] = useState<SelfInferenceKey | null>(null)
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
    setKeysHasMore(false)
    setSelectedKey(null)
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
    void fetchSelfInferenceKeyPage({}, controller.signal)
      .then(async (page) => {
        if (controller.signal.aborted) return
        const stored = readStoredSelection(storageKey)
        const selected = await restoreSelfInferenceKeySelection(
          page.items,
          stored,
          (keyId) => fetchSelfInferenceKey(keyId, controller.signal),
          controller.signal,
        )
        if (controller.signal.aborted) return
        setKeys(page.items)
        setKeysHasMore(page.hasMore)
        setSelectedKey(selected)
        setSelectedKeyIdState(selected?.keyId ?? '')
        if (selected) writeStoredSelection(storageKey, selected.keyId)
        setKeysStatus(selected ? 'ready' : 'unavailable')
      })
      .catch((cause: unknown) => {
        if (controller.signal.aborted) return
        setKeys([])
        setKeysHasMore(false)
        setSelectedKey(null)
        setSelectedKeyIdState('')
        setKeysError(errorMessage(cause, 'API keys are unavailable.'))
        setKeysStatus('error')
      })
    return () => controller.abort()
  }, [keysAttempt, shouldLoadKeys, storageKey, user?.managementIdentityStatus])

  const setSelectedKeyId = useCallback(
    (keyId: string) => {
      const key = keys.find((candidate) => candidate.keyId === keyId)
      if (!key && selectedKey?.keyId !== keyId) return
      setSelectedKey(key ?? selectedKey)
      setSelectedKeyIdState(keyId)
      writeStoredSelection(storageKey, keyId)
    },
    [keys, selectedKey, storageKey],
  )

  const selectKey = useCallback(
    (key: SelfInferenceKey) => {
      if (!activeSelfInferenceKeys([key]).length) return
      setSelectedKey(key)
      setSelectedKeyIdState(key.keyId)
      writeStoredSelection(storageKey, key.keyId)
    },
    [storageKey],
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
      keysHasMore,
      selectedKey,
      selectedKeyId,
      setSelectedKeyId,
      selectKey,
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
      keysHasMore,
      keysStatus,
      retryCatalog,
      retryKeys,
      selectedKey,
      selectedKeyId,
      selectKey,
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
