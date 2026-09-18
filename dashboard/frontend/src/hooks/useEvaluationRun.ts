import { useCallback, useEffect, useRef, useState } from 'react'

import type { EvaluationRun } from '../types/evaluationPlane'
import { getEvaluationRun } from '../utils/evaluationPlaneApi'

interface RunState {
  key: string
  run: EvaluationRun | null
  loading: boolean
  error: string | null
}

function runErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : 'Failed to load the evaluation run.'
}

export function useEvaluationRun(runID: string | null, loadedRun: EvaluationRun | null) {
  const key = runID || ''
  const [state, setState] = useState<RunState>({ key, run: loadedRun, loading: false, error: null })
  const requestVersion = useRef(0)
  const controller = useRef<AbortController | null>(null)
  const loadedRunRef = useRef(loadedRun)
  loadedRunRef.current = loadedRun

  const load = useCallback(async () => {
    if (!runID) return
    const version = ++requestVersion.current
    controller.current?.abort()
    const nextController = new AbortController()
    controller.current = nextController
    setState((current) => ({
      key,
      run: current.key === key ? current.run || loadedRunRef.current : loadedRunRef.current,
      loading: true,
      error: null,
    }))
    try {
      const run = await getEvaluationRun(runID, nextController.signal)
      if (nextController.signal.aborted || version !== requestVersion.current) return
      setState({ key, run, loading: false, error: null })
    } catch (reason) {
      if (nextController.signal.aborted || version !== requestVersion.current) return
      setState((current) => ({
        key,
        run: current.key === key ? current.run : loadedRunRef.current,
        loading: false,
        error: runErrorMessage(reason),
      }))
    }
  }, [key, runID])

  useEffect(() => {
    requestVersion.current += 1
    controller.current?.abort()
    const initialRun = loadedRunRef.current?.id === runID ? loadedRunRef.current : null
    setState({ key, run: initialRun, loading: Boolean(runID && !initialRun), error: null })
    if (runID && !initialRun) void load()
    return () => {
      requestVersion.current += 1
      controller.current?.abort()
    }
  }, [key, load, runID])

  useEffect(() => {
    if (!loadedRun || loadedRun.id !== runID) return
    requestVersion.current += 1
    controller.current?.abort()
    setState((current) =>
      current.key === key ? { key, run: loadedRun, loading: false, error: null } : current,
    )
  }, [key, loadedRun, runID])

  const currentState = state.key === key ? state : null
  return {
    run: currentState?.run || loadedRun,
    loading: Boolean(runID) && (!currentState || currentState.loading),
    error: currentState?.error || null,
    refresh: load,
  }
}
