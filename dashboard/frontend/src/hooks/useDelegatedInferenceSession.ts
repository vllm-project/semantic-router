import { useCallback, useEffect, useRef, useState } from 'react'

import { MANAGEMENT_API_HEADERS } from '../generated/managementApiContract'
import { managementOperationRequest } from '../utils/managementApiContract'
import type {
  ManagementPage,
  SecretEnvelope,
  SelfInferenceKey,
} from '../utils/routerManagementTypes'

const DELEGATED_SESSION_REFRESH_SKEW_MS = 30_000

interface CachedDelegatedSession {
  secret: string
  expiresAt: number
}

export type DelegatedInferenceStatus = 'loading' | 'ready' | 'unavailable' | 'error'

export interface DelegatedInferenceSessionState {
  getAccessToken: () => Promise<string>
  retry: () => void
  status: DelegatedInferenceStatus
}

function randomIdempotencyKey(): string {
  if (typeof crypto === 'undefined' || typeof crypto.randomUUID !== 'function') {
    throw new Error('This browser cannot create a secure Playground session.')
  }
  return crypto.randomUUID()
}

function sessionExpiry(expiresAt: string | undefined): number {
  if (!expiresAt) return 0
  const parsed = Date.parse(expiresAt)
  return Number.isFinite(parsed) ? parsed : 0
}

function assertKeyPage(payload: unknown): ManagementPage<SelfInferenceKey> {
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    throw new Error('Router returned an invalid inference key list.')
  }
  const candidate = payload as Partial<ManagementPage<SelfInferenceKey>>
  if (!Array.isArray(candidate.data) || !candidate.page || typeof candidate.page !== 'object') {
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
      typeof key.owner.id !== 'string'
    ) {
      throw new Error('Router returned an invalid inference key.')
    }
  }
  return candidate as ManagementPage<SelfInferenceKey>
}

function assertDelegatedSecret(payload: unknown): SecretEnvelope {
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    throw new Error('Router did not issue a Playground credential.')
  }
  const secret = payload as Partial<SecretEnvelope>
  if (
    secret.kind !== 'delegated_inference_credential' ||
    typeof secret.resourceId !== 'string' ||
    !secret.resourceId ||
    typeof secret.secret !== 'string' ||
    !secret.secret ||
    (secret.expiresAt !== undefined && typeof secret.expiresAt !== 'string')
  ) {
    throw new Error('Router did not issue a valid Playground credential.')
  }
  return secret as SecretEnvelope
}

// This memory-only credential is used only to discover authorized targets from
// /v1/models. Agent execution creates its own server-side delegation; the
// browser never owns a durable turn or tool-execution credential.
export function useDelegatedInferenceSession(): DelegatedInferenceSessionState {
  const [attempt, setAttempt] = useState(0)
  const [status, setStatus] = useState<DelegatedInferenceStatus>('loading')
  const keyRef = useRef<SelfInferenceKey | null>(null)
  const sessionRef = useRef<CachedDelegatedSession | null>(null)
  const issuanceRef = useRef<Promise<CachedDelegatedSession> | null>(null)
  const activeRef = useRef(true)

  const retry = useCallback(() => setAttempt((current) => current + 1), [])

  useEffect(() => {
    activeRef.current = true
    keyRef.current = null
    sessionRef.current = null
    issuanceRef.current = null
    setStatus('loading')
    const controller = new AbortController()

    void managementOperationRequest('getSelfInferenceKeys', {
      query: new URLSearchParams({ pageSize: '100' }),
      signal: controller.signal,
    })
      .then(assertKeyPage)
      .then((page) => {
        if (controller.signal.aborted || !activeRef.current) return
        const now = Date.now()
        const key = page.data.find((candidate) => {
          if (!candidate.expiresAt) return true
          const expiresAt = Date.parse(candidate.expiresAt)
          return Number.isFinite(expiresAt) && expiresAt > now
        })
        keyRef.current = key ?? null
        setStatus(key ? 'ready' : 'unavailable')
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted || !activeRef.current) return
        console.warn('Playground inference key discovery failed.', error)
        setStatus('error')
      })

    return () => {
      activeRef.current = false
      controller.abort()
    }
  }, [attempt])

  const getAccessToken = useCallback(async (): Promise<string> => {
    const now = Date.now()
    const cached = sessionRef.current
    if (cached && now + DELEGATED_SESSION_REFRESH_SKEW_MS < cached.expiresAt) {
      return cached.secret
    }
    if (!keyRef.current) {
      throw new Error('No inference key is available for this Playground session.')
    }
    if (!issuanceRef.current) {
      const keyId = keyRef.current.keyId
      issuanceRef.current = managementOperationRequest('postSelfInferenceSessions', {
        headers: { [MANAGEMENT_API_HEADERS.idempotencyKey]: randomIdempotencyKey() },
        body: { keyId },
      })
        .then(assertDelegatedSecret)
        .then((issued) => {
          const next: CachedDelegatedSession = {
            secret: issued.secret,
            expiresAt: sessionExpiry(issued.expiresAt),
          }
          if (!next.expiresAt || next.expiresAt <= Date.now()) {
            throw new Error('Router issued an expired Playground credential.')
          }
          sessionRef.current = next
          return next
        })
        .finally(() => {
          issuanceRef.current = null
        })
    }
    const issued = await issuanceRef.current
    return issued.secret
  }, [])

  return { getAccessToken, retry, status }
}
