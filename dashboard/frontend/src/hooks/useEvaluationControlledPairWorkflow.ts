import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import type { EvaluationControlledPairExecution } from '../types/evaluationControlledPair'
import {
  createEvaluationControlledPair,
  EvaluationRequestError,
  getEvaluationControlledPair,
} from '../utils/evaluationPlaneApi'
import { buildCreateEvaluationControlledPairPayload } from '../utils/evaluationControlledPairContract'
import {
  controlledPairErrorMessage,
  controlledPairIsReady,
  controlledPairTerminalFailure,
  handoffEvaluationControlledPair,
  INITIAL_CONTROLLED_PAIR_STATE,
  type EvaluationControlledPairReadyHandler,
  type EvaluationControlledPairWorkflow,
} from './evaluationControlledPairHookSupport'

export function useControlledPairSession(
  onReady: EvaluationControlledPairReadyHandler,
  workflow: EvaluationControlledPairWorkflow,
) {
  const [state, setState] = useState(INITIAL_CONTROLLED_PAIR_STATE)
  const requestVersion = useRef(0)
  const mounted = useRef(false)
  const onReadyRef = useRef(onReady)
  const onPairIdentityRef = useRef(workflow.onPairIdentity)
  const creatingPairID = useRef<string | null>(null)
  const reconciledRoutePairID = useRef<string | null>(null)
  const deliveredExecutionID = useRef<string | null>(null)
  onReadyRef.current = onReady
  onPairIdentityRef.current = workflow.onPairIdentity

  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
      requestVersion.current += 1
      reconciledRoutePairID.current = null
    }
  }, [])

  const session = useMemo(
    () => ({
      setState,
      requestVersion,
      mounted,
      onReadyRef,
      onPairIdentityRef,
      creatingPairID,
      reconciledRoutePairID,
      deliveredExecutionID,
    }),
    [],
  )
  return { state, session }
}

export type ControlledPairSession = ReturnType<typeof useControlledPairSession>['session']

export function useDeliverControlledPairReady(session: ControlledPairSession) {
  return useCallback(
    async (execution: EvaluationControlledPairExecution, generation: number) => {
      const isCurrent = () =>
        session.mounted.current && generation === session.requestVersion.current
      if (!isCurrent() || session.deliveredExecutionID.current === execution.id) return
      session.setState((current) => ({
        ...current,
        status: 'assigning',
        execution,
        error: null,
      }))
      const handoffError = await handoffEvaluationControlledPair(
        execution,
        session.onReadyRef.current,
        isCurrent,
      )
      if (!isCurrent()) return
      if (handoffError) {
        session.setState((current) => ({
          ...current,
          status: 'error',
          execution,
          error: handoffError,
        }))
        return
      }
      session.deliveredExecutionID.current = execution.id
      session.setState((current) => ({ ...current, status: 'ready', execution, error: null }))
    },
    [session],
  )
}

async function createControlledPair(
  session: ControlledPairSession,
  deliverReady: (execution: EvaluationControlledPairExecution, generation: number) => Promise<void>,
  baselineSourceRunID: string,
  candidateSourceRunID: string,
): Promise<EvaluationControlledPairExecution | null> {
  const version = ++session.requestVersion.current
  session.deliveredExecutionID.current = null
  session.setState({
    status: 'creating',
    execution: null,
    error: null,
    sourceIDs: { baseline: baselineSourceRunID, candidate: candidateSourceRunID },
  })
  try {
    const request = buildCreateEvaluationControlledPairPayload(
      baselineSourceRunID,
      candidateSourceRunID,
    )
    session.creatingPairID.current = request.client_request_id
    await session.onPairIdentityRef.current(request.client_request_id)
    if (!session.mounted.current || version !== session.requestVersion.current) return null
    const execution = await createEvaluationControlledPair(request)
    if (!session.mounted.current || version !== session.requestVersion.current) return null
    session.creatingPairID.current = null
    const failure = controlledPairTerminalFailure(execution)
    if (failure) {
      session.setState({
        status: 'error',
        execution,
        error: failure,
        sourceIDs: { baseline: baselineSourceRunID, candidate: candidateSourceRunID },
      })
      return null
    }
    if (controlledPairIsReady(execution)) {
      session.setState({
        status: 'assigning',
        execution,
        error: null,
        sourceIDs: { baseline: baselineSourceRunID, candidate: candidateSourceRunID },
      })
      await deliverReady(execution, version)
      if (!session.mounted.current || version !== session.requestVersion.current) return null
    } else {
      session.setState({
        status: 'running',
        execution,
        error: null,
        sourceIDs: { baseline: baselineSourceRunID, candidate: candidateSourceRunID },
      })
    }
    return execution
  } catch (error) {
    if (!session.mounted.current || version !== session.requestVersion.current) return null
    session.creatingPairID.current = null
    if (error instanceof EvaluationRequestError && error.status >= 400 && error.status < 500) {
      await session.onPairIdentityRef.current(null)
      if (!session.mounted.current || version !== session.requestVersion.current) return null
    }
    session.setState({
      status: 'error',
      execution: null,
      error: controlledPairErrorMessage(error, 'Controlled AB/BA execution could not be created.'),
      sourceIDs: { baseline: baselineSourceRunID, candidate: candidateSourceRunID },
    })
    return null
  }
}

export function useCreateControlledPair(
  session: ControlledPairSession,
  deliverReady: (execution: EvaluationControlledPairExecution, generation: number) => Promise<void>,
) {
  return useCallback(
    (baselineSourceRunID: string, candidateSourceRunID: string) =>
      createControlledPair(session, deliverReady, baselineSourceRunID, candidateSourceRunID),
    [deliverReady, session],
  )
}

async function reconcileControlledPair(
  session: ControlledPairSession,
  deliverReady: (execution: EvaluationControlledPairExecution, generation: number) => Promise<void>,
  pairID: string,
): Promise<EvaluationControlledPairExecution | null> {
  const version = ++session.requestVersion.current
  session.deliveredExecutionID.current = null
  session.setState((current) => ({
    status: 'recovering',
    execution: current.execution?.id === pairID ? current.execution : null,
    error: null,
    sourceIDs: current.execution?.id === pairID ? current.sourceIDs : null,
  }))
  try {
    const execution = await getEvaluationControlledPair(pairID)
    if (!session.mounted.current || version !== session.requestVersion.current) return null
    const failure = controlledPairTerminalFailure(execution)
    if (failure) {
      session.setState((current) => ({ ...current, status: 'error', execution, error: failure }))
      return null
    }
    if (controlledPairIsReady(execution)) {
      await deliverReady(execution, version)
      if (!session.mounted.current || version !== session.requestVersion.current) return null
    } else {
      session.setState((current) => ({ ...current, status: 'running', execution, error: null }))
    }
    return execution
  } catch (error) {
    if (!session.mounted.current || version !== session.requestVersion.current) return null
    session.setState((current) => ({
      ...current,
      status: 'error',
      execution: null,
      error: controlledPairErrorMessage(
        error,
        'The saved controlled comparison could not be reconciled with the server.',
      ),
    }))
    return null
  }
}

export function useReconcileControlledPair(
  session: ControlledPairSession,
  deliverReady: (execution: EvaluationControlledPairExecution, generation: number) => Promise<void>,
) {
  return useCallback(
    (pairID: string) => reconcileControlledPair(session, deliverReady, pairID),
    [deliverReady, session],
  )
}

export function useControlledPairRouteReconciliation(
  session: ControlledPairSession,
  state: ReturnType<typeof useControlledPairSession>['state'],
  activePairID: string | null,
  reconcile: (pairID: string) => Promise<EvaluationControlledPairExecution | null>,
): void {
  useEffect(() => {
    if (!activePairID) {
      session.reconciledRoutePairID.current = null
      return
    }
    if (state.status === 'creating' && session.creatingPairID.current === activePairID) return
    if (
      state.execution?.id === activePairID ||
      session.reconciledRoutePairID.current === activePairID
    )
      return
    session.reconciledRoutePairID.current = activePairID
    void reconcile(activePairID)
  }, [activePairID, reconcile, session, state.execution?.id, state.status])
}

export function useControlledPairPolling(
  session: ControlledPairSession,
  state: ReturnType<typeof useControlledPairSession>['state'],
  deliverReady: (execution: EvaluationControlledPairExecution, generation: number) => Promise<void>,
): void {
  const pairID = state.execution?.id
  useEffect(() => {
    if (state.status !== 'running' || !pairID) return
    const version = session.requestVersion.current
    let stopped = false
    let timer: number | undefined
    let controller: AbortController | null = null
    const poll = async () => {
      controller?.abort()
      controller = new AbortController()
      try {
        const next = await getEvaluationControlledPair(pairID, controller.signal)
        if (stopped || !session.mounted.current || version !== session.requestVersion.current)
          return
        const failure = controlledPairTerminalFailure(next)
        if (failure) {
          session.setState((current) => ({
            ...current,
            status: 'error',
            execution: next,
            error: failure,
          }))
          return
        }
        if (controlledPairIsReady(next)) {
          await deliverReady(next, version)
          return
        }
        session.setState((current) => ({ ...current, execution: next, error: null }))
        timer = window.setTimeout(() => void poll(), 2_000)
      } catch (error) {
        if (
          stopped ||
          controller.signal.aborted ||
          !session.mounted.current ||
          version !== session.requestVersion.current
        )
          return
        session.setState((current) => ({
          ...current,
          status: 'error',
          error: controlledPairErrorMessage(
            error,
            'Controlled pair progress is temporarily unreachable.',
          ),
        }))
      }
    }
    timer = window.setTimeout(() => void poll(), 500)
    return () => {
      stopped = true
      controller?.abort()
      if (timer !== undefined) window.clearTimeout(timer)
    }
  }, [deliverReady, pairID, session, state.status])
}

export function useRetryControlledPair(
  session: ControlledPairSession,
  state: ReturnType<typeof useControlledPairSession>['state'],
  activePairID: string | null,
  create: (
    baseline: string,
    candidate: string,
  ) => Promise<EvaluationControlledPairExecution | null>,
  deliverReady: (execution: EvaluationControlledPairExecution, generation: number) => Promise<void>,
  reconcile: (pairID: string) => Promise<EvaluationControlledPairExecution | null>,
) {
  return useCallback(() => {
    const terminal = state.execution ? controlledPairTerminalFailure(state.execution) : null
    if (state.execution && controlledPairIsReady(state.execution) && !terminal) {
      const generation = ++session.requestVersion.current
      void deliverReady(state.execution, generation)
      return
    }
    if (activePairID) {
      void reconcile(activePairID)
      return
    }
    if (!state.sourceIDs) return
    if (!state.execution || terminal) {
      void create(state.sourceIDs.baseline, state.sourceIDs.candidate)
      return
    }
    session.requestVersion.current += 1
    session.setState((current) => ({ ...current, status: 'running', error: null }))
  }, [activePairID, create, deliverReady, reconcile, session, state.execution, state.sourceIDs])
}

export function useResetControlledPair(session: ControlledPairSession) {
  return useCallback(() => {
    session.requestVersion.current += 1
    session.deliveredExecutionID.current = null
    session.creatingPairID.current = null
    session.reconciledRoutePairID.current = null
    session.setState(INITIAL_CONTROLLED_PAIR_STATE)
  }, [session])
}
