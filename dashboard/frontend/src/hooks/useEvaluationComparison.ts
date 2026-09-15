import { useCallback, useEffect, useRef, useState } from 'react'

import type { EvaluationComparison } from '../types/evaluationComparison'
import { compareEvaluationRuns } from '../utils/evaluationPlaneApi'

function comparisonErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : 'Failed to compare evaluation runs.'
}

type ComparisonState =
  | { key: string; status: 'idle'; comparison: null; error: null }
  | { key: string; status: 'loading'; comparison: null; error: null }
  | { key: string; status: 'ready'; comparison: EvaluationComparison; error: null }
  | { key: string; status: 'error'; comparison: null; error: string }

export function useEvaluationComparison(
  baselineID: string,
  candidateID: string,
  runLedgerComplete = true,
) {
  const key = `${runLedgerComplete ? 'complete' : 'blocked'}\u0000${baselineID}\u0000${candidateID}`
  const [state, setState] = useState<ComparisonState>({
    key,
    status: 'idle',
    comparison: null,
    error: null,
  })
  const requestVersion = useRef(0)
  const controller = useRef<AbortController | null>(null)

  const compare = useCallback(async () => {
    if (!runLedgerComplete || !baselineID || !candidateID || baselineID === candidateID) return
    const version = ++requestVersion.current
    controller.current?.abort()
    const nextController = new AbortController()
    controller.current = nextController
    setState({ key, status: 'loading', comparison: null, error: null })
    try {
      const nextComparison = await compareEvaluationRuns(
        baselineID,
        candidateID,
        nextController.signal,
      )
      if (nextController.signal.aborted || version !== requestVersion.current) return
      setState({ key, status: 'ready', comparison: nextComparison, error: null })
    } catch (comparisonError) {
      if (nextController.signal.aborted || version !== requestVersion.current) return
      setState({
        key,
        status: 'error',
        comparison: null,
        error: comparisonErrorMessage(comparisonError),
      })
    }
  }, [baselineID, candidateID, key, runLedgerComplete])

  useEffect(() => {
    requestVersion.current += 1
    controller.current?.abort()
    setState({ key, status: 'idle', comparison: null, error: null })
  }, [key])

  useEffect(
    () => () => {
      requestVersion.current += 1
      controller.current?.abort()
    },
    [],
  )

  const currentState = state.key === key ? state : null

  return {
    comparison: currentState?.comparison || null,
    loading: currentState?.status === 'loading',
    error: currentState?.error || null,
    compare,
  }
}
