import { useCallback, useEffect, useRef, useState } from 'react'

import {
  ModelVerificationApiError,
  verifyProviderModel,
  type ModelLiveVerification,
} from '../utils/modelVerificationApi'

export type ModelLiveVerificationState =
  | { status: 'idle' }
  | { status: 'pending' }
  | { status: 'verified'; evidence: ModelLiveVerification }
  | { status: 'failed'; message: string }

const IDLE_MODEL_VERIFICATION: ModelLiveVerificationState = { status: 'idle' }

type ModelVerificationStateUpdater = (
  update: (
    current: Map<string, ModelLiveVerificationState>,
  ) => Map<string, ModelLiveVerificationState>,
) => void

type ModelVerifier = typeof verifyProviderModel

export function modelLiveVerificationState(
  states: ReadonlyMap<string, ModelLiveVerificationState>,
  model: string,
): ModelLiveVerificationState {
  return states.get(model) ?? IDLE_MODEL_VERIFICATION
}

export function resetModelLiveVerification(
  activeRequests: Map<string, AbortController>,
  updateStates: ModelVerificationStateUpdater,
) {
  for (const controller of activeRequests.values()) controller.abort()
  activeRequests.clear()
  updateStates(() => new Map())
}

export async function runModelLiveVerification(
  model: string,
  activeRequests: Map<string, AbortController>,
  updateStates: ModelVerificationStateUpdater,
  verifier: ModelVerifier = verifyProviderModel,
) {
  const previous = activeRequests.get(model)
  if (previous) previous.abort()
  const controller = new AbortController()
  activeRequests.set(model, controller)
  updateStates((current) => new Map(current).set(model, { status: 'pending' }))

  try {
    const evidence = await verifier(model, controller.signal)
    if (activeRequests.get(model) !== controller) return
    activeRequests.delete(model)
    updateStates((current) => new Map(current).set(model, { status: 'verified', evidence }))
  } catch (error) {
    if (controller.signal.aborted || activeRequests.get(model) !== controller) return
    activeRequests.delete(model)
    const message =
      error instanceof ModelVerificationApiError
        ? error.message
        : 'Live model verification failed before the provider returned generated text.'
    updateStates((current) => new Map(current).set(model, { status: 'failed', message }))
  }
}

export function useModelLiveVerification(configRevision: unknown) {
  const [states, setStates] = useState<Map<string, ModelLiveVerificationState>>(new Map())
  const activeRequests = useRef(new Map<string, AbortController>())

  const abortAndClear = useCallback(() => {
    resetModelLiveVerification(activeRequests.current, setStates)
  }, [])

  useEffect(() => abortAndClear(), [abortAndClear, configRevision])
  useEffect(
    () => () => {
      for (const controller of activeRequests.current.values()) controller.abort()
      activeRequests.current.clear()
    },
    [],
  )

  const verify = useCallback(async (model: string) => {
    await runModelLiveVerification(model, activeRequests.current, setStates)
  }, [])

  return { states, verify }
}
