import { useEffect, useRef, useState } from 'react'
import { SrBenchRequestError } from './api'
import { experimentApi, type Experiment } from './experimentApi'
import {
  clearExperiment,
  readExperiment,
  saveExperiment,
  type PendingExperiment,
} from './experimentSubmission'

/**
 * Creates experiment metadata only. A request whose response was lost stays in
 * session storage and is confirmed with the same idempotency key.
 */
export function useExperimentCreation(actorID: string, onCreated: (id: string) => void) {
  const [saved] = useState(() => readExperiment(actorID))
  const [pendingRequest, setPendingRequest] = useState<PendingExperiment | null>(saved.saved)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const mounted = useRef(false)
  const sequence = useRef(0)
  const busy = useRef(false)
  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
      sequence.current += 1
    }
  }, [])

  async function create(name: string) {
    if (busy.current || saved.error) return false
    const requestSequence = ++sequence.current
    const current = () => mounted.current && sequence.current === requestSequence
    busy.current = true
    setSubmitting(true)
    setError('')
    setNotice('')
    try {
      const body: PendingExperiment = pendingRequest ?? {
        version: 1,
        actorID,
        name: name.trim(),
        key: crypto.randomUUID(),
      }
      saveExperiment(body)
      setPendingRequest(body)
      let result: Experiment
      try {
        result = await experimentApi.create(body.name, body.key)
      } catch (cause) {
        if (!(cause instanceof SrBenchRequestError) || cause.code !== 'experiment_deleted')
          throw cause
        if (!current()) return false
        if (!clearExperiment(body))
          throw new Error('The saved submission changed. Reload to reconcile its identity.')
        setPendingRequest(null)
        setNotice('That experiment was deleted. Enter a name to create a new experiment.')
        return false
      }
      if (!current()) return false
      if (!clearExperiment(body))
        throw new Error(
          'The saved experiment changed while this response was pending. Reload to reconcile its identity.',
        )
      setPendingRequest(null)
      onCreated(result.id)
      return true
    } catch (cause) {
      if (current())
        setError(cause instanceof Error ? cause.message : 'Could not save the experiment.')
      return false
    } finally {
      busy.current = false
      if (current()) setSubmitting(false)
    }
  }

  return { pendingRequest, storageError: saved.error, submitting, error, notice, create }
}
