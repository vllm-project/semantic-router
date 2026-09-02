import { useCallback, useEffect, useRef, useState } from 'react'

import type { EvaluationControlledPairExecution } from '../types/evaluationControlledPair'
import { getEvaluationControlledPair } from '../utils/evaluationPlaneApi'

interface ControlledPairResourceState {
  key: string
  execution: EvaluationControlledPairExecution | null
  loading: boolean
  error: string | null
}

function resourceError(error: unknown): string {
  return error instanceof Error ? error.message : 'Failed to load controlled comparison status.'
}

export function useEvaluationControlledPairResource(pairID: string | null) {
  const key = pairID || ''
  const [state, setState] = useState<ControlledPairResourceState>({
    key,
    execution: null,
    loading: Boolean(pairID),
    error: null,
  })
  const requestVersion = useRef(0)
  const controller = useRef<AbortController | null>(null)

  const load = useCallback(async () => {
    if (!pairID) return null
    const version = ++requestVersion.current
    controller.current?.abort()
    const nextController = new AbortController()
    controller.current = nextController
    setState((current) => ({
      key,
      execution: current.key === key ? current.execution : null,
      loading: true,
      error: null,
    }))
    try {
      const execution = await getEvaluationControlledPair(pairID, nextController.signal)
      if (nextController.signal.aborted || version !== requestVersion.current) return null
      setState({ key, execution, loading: false, error: null })
      return execution
    } catch (error) {
      if (nextController.signal.aborted || version !== requestVersion.current) return null
      setState((current) => ({
        key,
        execution: current.key === key ? current.execution : null,
        loading: false,
        error: resourceError(error),
      }))
      return null
    }
  }, [key, pairID])

  const adopt = useCallback(
    (execution: EvaluationControlledPairExecution) => {
      if (execution.id !== pairID) return false
      requestVersion.current += 1
      controller.current?.abort()
      setState({ key, execution, loading: false, error: null })
      return true
    },
    [key, pairID],
  )

  useEffect(() => {
    requestVersion.current += 1
    controller.current?.abort()
    setState({ key, execution: null, loading: Boolean(pairID), error: null })
    if (pairID) void load()
    return () => {
      requestVersion.current += 1
      controller.current?.abort()
    }
  }, [key, load, pairID])

  const current = state.key === key ? state : null
  const execution = current?.execution || null
  const executionState = execution?.state

  useEffect(() => {
    if (!pairID || !executionState || executionState === 'terminal') return
    const refreshWhenVisible = () => {
      if (!document.hidden) void load()
    }
    const interval = window.setInterval(() => {
      if (!document.hidden) void load()
    }, 5_000)
    document.addEventListener('visibilitychange', refreshWhenVisible)
    return () => {
      window.clearInterval(interval)
      document.removeEventListener('visibilitychange', refreshWhenVisible)
    }
  }, [executionState, load, pairID])

  return {
    execution,
    loading: Boolean(pairID) && !execution && (!current || current.loading),
    refreshing: Boolean(pairID && execution && current?.loading),
    error: current?.error || null,
    refresh: load,
    adopt,
  }
}
