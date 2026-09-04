import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import type {
  CreateEvaluationRunRequest,
  EvaluationCatalog,
  EvaluationComparison,
  EvaluationReport,
  EvaluationRun,
  EvaluationRunEvent,
} from '../types/evaluationPlane'
import {
  cancelEvaluationRun,
  compareEvaluationRuns,
  createEvaluationRun,
  deleteEvaluationRun,
  getEvaluationCatalog,
  getEvaluationReport,
  listEvaluationRuns,
  startEvaluationRun,
  subscribeToEvaluationRun,
} from '../utils/evaluationPlaneApi'
import { appendEvaluationEvent } from './evaluationEventSupport'

function messageFrom(error: unknown, fallback: string): string {
  return error instanceof Error ? error.message : fallback
}

function sortRuns(runs: EvaluationRun[]): EvaluationRun[] {
  return [...runs].sort((left, right) => Date.parse(right.created_at) - Date.parse(left.created_at))
}

export function useEvaluationPlane() {
  const [catalog, setCatalog] = useState<EvaluationCatalog | null>(null)
  const [runs, setRuns] = useState<EvaluationRun[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [mutationPending, setMutationPending] = useState(false)
  const [mutationError, setMutationError] = useState<string | null>(null)
  const requestVersion = useRef(0)

  const refresh = useCallback(async (showLoading = false) => {
    const version = ++requestVersion.current
    if (showLoading) setLoading(true)
    try {
      const [nextCatalog, nextRuns] = await Promise.all([
        getEvaluationCatalog(),
        listEvaluationRuns(),
      ])
      if (version !== requestVersion.current) return
      setCatalog(nextCatalog)
      setRuns(sortRuns(nextRuns))
      setError(null)
    } catch (refreshError) {
      if (version !== requestVersion.current) return
      setError(messageFrom(refreshError, 'Failed to load the evaluation plane.'))
    } finally {
      if (version === requestVersion.current) setLoading(false)
    }
  }, [])

  const refreshRuns = useCallback(async () => {
    try {
      setRuns(sortRuns(await listEvaluationRuns()))
      setError(null)
    } catch (refreshError) {
      setError(messageFrom(refreshError, 'Failed to refresh evaluation runs.'))
    }
  }, [])

  useEffect(() => {
    void refresh(true)
    const interval = window.setInterval(() => {
      if (!document.hidden) void refreshRuns()
    }, 5_000)
    const handleVisibility = () => {
      if (!document.hidden) void refreshRuns()
    }
    document.addEventListener('visibilitychange', handleVisibility)
    return () => {
      requestVersion.current += 1
      window.clearInterval(interval)
      document.removeEventListener('visibilitychange', handleVisibility)
    }
  }, [refresh, refreshRuns])

  const replaceRun = useCallback((nextRun: EvaluationRun) => {
    setRuns((current) =>
      sortRuns([nextRun, ...current.filter((candidate) => candidate.id !== nextRun.id)]),
    )
  }, [])

  const mutateRun = useCallback(
    async (operation: () => Promise<EvaluationRun>, fallback: string) => {
      setMutationPending(true)
      setMutationError(null)
      try {
        const run = await operation()
        replaceRun(run)
        return run
      } catch (mutationFailure) {
        setMutationError(messageFrom(mutationFailure, fallback))
        return null
      } finally {
        setMutationPending(false)
      }
    },
    [replaceRun],
  )

  const createRun = useCallback(
    async (request: CreateEvaluationRunRequest) => {
      if (!catalog) {
        setMutationError('The evaluation catalog is not available yet.')
        return null
      }
      return mutateRun(
        () => createEvaluationRun(request, catalog),
        'Failed to create the evaluation run.',
      )
    },
    [catalog, mutateRun],
  )

  const startRun = useCallback(
    (id: string) => mutateRun(() => startEvaluationRun(id), 'Failed to start the evaluation run.'),
    [mutateRun],
  )

  const cancelRun = useCallback(
    (id: string) =>
      mutateRun(() => cancelEvaluationRun(id), 'Failed to cancel the evaluation run.'),
    [mutateRun],
  )

  const deleteRun = useCallback(async (id: string) => {
    setMutationPending(true)
    setMutationError(null)
    try {
      await deleteEvaluationRun(id)
      setRuns((current) => current.filter((run) => run.id !== id))
      return true
    } catch (mutationFailure) {
      setMutationError(messageFrom(mutationFailure, 'Failed to delete the evaluation run.'))
      return false
    } finally {
      setMutationPending(false)
    }
  }, [])

  return {
    catalog,
    runs,
    loading,
    error,
    mutationPending,
    mutationError,
    clearMutationError: () => setMutationError(null),
    refresh: () => refresh(true),
    refreshRuns,
    createRun,
    startRun,
    cancelRun,
    deleteRun,
  }
}

export function useEvaluationReport(runID: string | null) {
  const [report, setReport] = useState<EvaluationReport | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    if (!runID) return
    setLoading(true)
    try {
      setReport(await getEvaluationReport(runID))
      setError(null)
    } catch (reportError) {
      setReport(null)
      setError(messageFrom(reportError, 'Failed to load the evaluation report.'))
    } finally {
      setLoading(false)
    }
  }, [runID])

  useEffect(() => {
    setReport(null)
    setError(null)
    if (!runID) return
    const controller = new AbortController()
    setLoading(true)
    void getEvaluationReport(runID, controller.signal)
      .then((nextReport) => {
        setReport(nextReport)
        setError(null)
      })
      .catch((reportError: unknown) => {
        if (controller.signal.aborted) return
        setError(messageFrom(reportError, 'Failed to load the evaluation report.'))
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false)
      })
    return () => controller.abort()
  }, [runID])

  return { report: report?.run.id === runID ? report : null, loading, error, refresh }
}

export function useEvaluationComparison(baselineID: string, candidateID: string) {
  const [comparison, setComparison] = useState<EvaluationComparison | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const compare = useCallback(async () => {
    if (!baselineID || !candidateID || baselineID === candidateID) return
    setLoading(true)
    setComparison(null)
    try {
      setComparison(await compareEvaluationRuns(baselineID, candidateID))
      setError(null)
    } catch (comparisonError) {
      setError(messageFrom(comparisonError, 'Failed to compare evaluation runs.'))
    } finally {
      setLoading(false)
    }
  }, [baselineID, candidateID])

  useEffect(() => {
    setComparison(null)
    setError(null)
  }, [baselineID, candidateID])

  return { comparison, loading, error, compare }
}

export function useEvaluationRunEvents(run: EvaluationRun | null, onTerminal: () => void) {
  const [events, setEvents] = useState<EvaluationRunEvent[]>([])
  const [connected, setConnected] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const terminalHandler = useRef(onTerminal)
  terminalHandler.current = onTerminal
  const runID = run?.id || null
  const runStatus = run?.status || null

  useEffect(() => {
    setEvents([])
    setError(null)
    setConnected(false)
    if (!runID || runStatus !== 'running') return

    let disconnect: (() => void) | null = null
    const connect = () => {
      if (document.hidden || disconnect) return
      disconnect = subscribeToEvaluationRun(
        runID,
        (event) => {
          setConnected(true)
          setError(null)
          setEvents((current) => appendEvaluationEvent(current, event))
        },
        () => {
          disconnect = null
          setConnected(false)
          terminalHandler.current()
        },
        (streamError) => {
          disconnect = null
          setConnected(false)
          setError(streamError.message)
        },
      )
    }
    const handleVisibility = () => {
      if (document.hidden) {
        disconnect?.()
        disconnect = null
        setConnected(false)
      } else {
        connect()
      }
    }
    connect()
    document.addEventListener('visibilitychange', handleVisibility)
    return () => {
      document.removeEventListener('visibilitychange', handleVisibility)
      disconnect?.()
    }
  }, [runID, runStatus])

  return useMemo(() => ({ events, connected, error }), [connected, error, events])
}
