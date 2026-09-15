import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import type { EvaluationRun, EvaluationRunEvent } from '../types/evaluationPlane'
import { subscribeToEvaluationRun } from '../utils/evaluationPlaneApi'
import { appendEvaluationEvent } from './evaluationEventSupport'

export function useEvaluationRunEvents(run: EvaluationRun | null, onTerminal: () => void) {
  const [events, setEvents] = useState<EvaluationRunEvent[]>([])
  const [connected, setConnected] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [retryVersion, setRetryVersion] = useState(0)
  const terminalHandler = useRef(onTerminal)
  terminalHandler.current = onTerminal
  const runSnapshot = useRef(run)
  runSnapshot.current = run
  const runID = run?.id || null
  const runStatus = run?.status || null

  useEffect(() => {
    setEvents([])
    setError(null)
    setConnected(false)
    const subscribedRun = runSnapshot.current
    if (!subscribedRun) return

    let disconnect: (() => void) | null = null
    const connect = () => {
      if (document.hidden || disconnect) return
      disconnect = subscribeToEvaluationRun(
        subscribedRun,
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
          disconnect?.()
          disconnect = null
          setConnected(false)
          setError(streamError.message)
          terminalHandler.current()
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
  }, [retryVersion, runID, runStatus])

  const retry = useCallback(() => {
    terminalHandler.current()
    setRetryVersion((version) => version + 1)
  }, [])

  return useMemo(() => ({ events, connected, error, retry }), [connected, error, events, retry])
}
