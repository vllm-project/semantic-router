import { useCallback, useEffect, useRef } from 'react'

import { useInferenceRoutingAccess } from '../contexts/InferenceRoutingAccessContext'
import { MANAGEMENT_API_HEADERS } from '../generated/managementApiContract'
import { managementOperationRequest } from '../utils/managementApiContract'
import type { SecretEnvelope } from '../utils/routerManagementTypes'

const DELEGATED_SESSION_REFRESH_SKEW_MS = 30_000

interface CachedDelegatedSession {
  keyId: string
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

// This memory-only credential is used only to discover authorized targets from
// /v1/models. Agent execution creates its own server-side delegation; the
// browser never owns a durable turn or tool-execution credential.
export function useDelegatedInferenceSession(): DelegatedInferenceSessionState {
  const { keysStatus, retryKeys, selectedKey } = useInferenceRoutingAccess()
  const sessionRef = useRef<CachedDelegatedSession | null>(null)
  const selectedKeyIdRef = useRef(selectedKey?.keyId ?? '')
  selectedKeyIdRef.current = selectedKey?.keyId ?? ''
  const issuanceRef = useRef<{
    keyId: string
    promise: Promise<CachedDelegatedSession>
  } | null>(null)

  useEffect(() => {
    sessionRef.current = null
    issuanceRef.current = null
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
      const promise = managementOperationRequest('postSelfInferenceSessions', {
        headers: { [MANAGEMENT_API_HEADERS.idempotencyKey]: randomIdempotencyKey() },
        body: { keyId },
      })
        .then(assertDelegatedSecret)
        .then((issued) => {
          const next: CachedDelegatedSession = {
            keyId,
            secret: issued.secret,
            expiresAt: sessionExpiry(issued.expiresAt),
          }
          if (!next.expiresAt || next.expiresAt <= Date.now()) {
            throw new Error('Router issued an expired Playground credential.')
          }
          if (selectedKeyIdRef.current === keyId) sessionRef.current = next
          return next
        })
        .finally(() => {
          if (issuanceRef.current?.keyId === keyId) issuanceRef.current = null
        })
      issuanceRef.current = { keyId, promise }
    }
    const issued = await issuanceRef.current.promise
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
