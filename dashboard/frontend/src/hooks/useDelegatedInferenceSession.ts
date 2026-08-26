import { useCallback, useEffect, useRef } from 'react'

import { useInferenceRoutingAccess } from '../contexts/InferenceRoutingAccessContext'
import { MANAGEMENT_API_HEADERS } from '../generated/managementApiContract'
import { managementOperationRequest } from '../utils/managementApiContract'
import { DelegatedInferenceIssuanceIntents } from '../utils/delegatedInferenceIssuance'
import { OwnedDelegatedInferenceSessions } from '../utils/ownedDelegatedInferenceSessions'
import type { SecretEnvelope } from '../utils/routerManagementTypes'

const DELEGATED_SESSION_REFRESH_SKEW_MS = 30_000

interface CachedDelegatedSession {
  keyId: string
  resourceId: string
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

// This memory-only credential authorizes the browser's ordinary OpenAI-compatible
// model discovery and streaming inference calls. Durable API keys and Agent tool
// authority never enter browser storage.
export function useDelegatedInferenceSession(): DelegatedInferenceSessionState {
  const { keysStatus, retryKeys, selectedKey } = useInferenceRoutingAccess()
  const sessionRef = useRef<CachedDelegatedSession | null>(null)
  const selectedKeyIdRef = useRef(selectedKey?.keyId ?? '')
  selectedKeyIdRef.current = selectedKey?.keyId ?? ''
  const ownedSessionsRef = useRef<OwnedDelegatedInferenceSessions | null>(null)
  if (!ownedSessionsRef.current) {
    ownedSessionsRef.current = new OwnedDelegatedInferenceSessions((resourceId) => {
      void managementOperationRequest('deleteSelfInferenceSessionsBySessionId', {
        pathParameters: { sessionId: resourceId },
      }).catch(() => undefined)
    })
  }
  const issuanceIntentsRef = useRef<DelegatedInferenceIssuanceIntents | null>(null)
  if (!issuanceIntentsRef.current) {
    issuanceIntentsRef.current = new DelegatedInferenceIssuanceIntents(randomIdempotencyKey)
  }
  const issuanceRef = useRef<{
    keyId: string
    promise: Promise<CachedDelegatedSession>
  } | null>(null)

  useEffect(() => {
    const ownedSessions = ownedSessionsRef.current!
    ownedSessions.activate(selectedKey?.keyId ?? '')
    issuanceIntentsRef.current!.reset()
    sessionRef.current = null
    issuanceRef.current = null
    return () => {
      sessionRef.current = null
      issuanceRef.current = null
      issuanceIntentsRef.current!.reset()
      ownedSessions.deactivate()
    }
  }, [selectedKey?.keyId])

  const getAccessToken = useCallback(async (): Promise<string> => {
    const keyId = selectedKey?.keyId
    if (!keyId) {
      throw new Error('No inference key is available for this Playground session.')
    }
    const now = Date.now()
    const cached = sessionRef.current
    if (cached?.keyId === keyId && now + DELEGATED_SESSION_REFRESH_SKEW_MS < cached.expiresAt) {
      return cached.secret
    }
    if (issuanceRef.current?.keyId !== keyId) {
      const ownedSessions = ownedSessionsRef.current!
      const claim = ownedSessions.begin(keyId)
      if (!claim) {
        throw new DOMException('Playground credential issuance was superseded.', 'AbortError')
      }
      const previous = sessionRef.current
      const idempotencyKey = issuanceIntentsRef.current!.keyFor(keyId)
      const issuedPromise = managementOperationRequest('postSelfInferenceSessions', {
        headers: { [MANAGEMENT_API_HEADERS.idempotencyKey]: idempotencyKey },
        body: { keyId },
      })
        .then(assertDelegatedSecret)
        .then((issued) => {
          const next: CachedDelegatedSession = {
            keyId,
            resourceId: issued.resourceId,
            secret: issued.secret,
            expiresAt: sessionExpiry(issued.expiresAt),
          }
          if (!ownedSessions.claim(claim, next)) {
            throw new DOMException('Playground credential issuance was superseded.', 'AbortError')
          }
          if (!next.expiresAt || next.expiresAt <= Date.now()) {
            ownedSessions.retire(next.resourceId)
            throw new Error('Router issued an expired Playground credential.')
          }
          if (selectedKeyIdRef.current !== keyId) {
            ownedSessions.retire(next.resourceId)
            throw new DOMException('Playground credential issuance was superseded.', 'AbortError')
          }
          sessionRef.current = next
          issuanceIntentsRef.current!.complete(keyId, idempotencyKey)
          if (previous && previous.resourceId !== next.resourceId) {
            ownedSessions.retire(previous.resourceId)
          }
          return next
        })
      const promise: Promise<CachedDelegatedSession> = issuedPromise.finally(() => {
        if (issuanceRef.current?.promise === promise) issuanceRef.current = null
      })
      issuanceRef.current = { keyId, promise }
    }
    const issuance = issuanceRef.current
    if (!issuance || issuance.keyId !== keyId) {
      throw new DOMException('Playground credential issuance was superseded.', 'AbortError')
    }
    const issued = await issuance.promise
    return issued.secret
  }, [selectedKey?.keyId])

  const status: DelegatedInferenceStatus =
    keysStatus === 'ready'
      ? 'ready'
      : keysStatus === 'unavailable'
        ? 'unavailable'
        : keysStatus === 'error'
          ? 'error'
          : 'loading'

  return { getAccessToken, retry: retryKeys, status }
}
