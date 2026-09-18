import { useCallback, useEffect, useRef, useState } from 'react'

import type { EvaluationReport } from '../types/evaluationReport'
import { getEvaluationReport } from '../utils/evaluationPlaneApi'

function reportErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : 'Failed to load the evaluation report.'
}

export function useEvaluationReport(runID: string | null) {
  const [report, setReport] = useState<EvaluationReport | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const requestVersion = useRef(0)

  const clearError = useCallback(() => {
    setError(null)
  }, [])
  const recordError = useCallback((reportError: unknown) => {
    setError(reportErrorMessage(reportError))
  }, [])

  const refresh = useCallback(async () => {
    if (!runID) return
    const version = ++requestVersion.current
    setLoading(true)
    try {
      const nextReport = await getEvaluationReport(runID)
      if (version !== requestVersion.current) return
      setReport(nextReport)
      clearError()
    } catch (reportError) {
      if (version !== requestVersion.current) return
      setReport(null)
      recordError(reportError)
    } finally {
      if (version === requestVersion.current) setLoading(false)
    }
  }, [clearError, recordError, runID])

  useEffect(() => {
    const version = ++requestVersion.current
    setReport(null)
    clearError()
    setLoading(false)
    if (!runID) return
    const controller = new AbortController()
    setLoading(true)
    void getEvaluationReport(runID, controller.signal)
      .then((nextReport) => {
        if (version !== requestVersion.current) return
        setReport(nextReport)
        clearError()
      })
      .catch((reportError: unknown) => {
        if (controller.signal.aborted || version !== requestVersion.current) return
        recordError(reportError)
      })
      .finally(() => {
        if (!controller.signal.aborted && version === requestVersion.current) setLoading(false)
      })
    return () => {
      requestVersion.current += 1
      controller.abort()
    }
  }, [clearError, recordError, runID])

  return {
    report: report?.run.id === runID ? report : null,
    loading,
    error,
    refresh,
  }
}
