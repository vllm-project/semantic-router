import { useEffect, useRef, useState } from 'react'
import { Trash2 } from 'lucide-react'
import { Button } from '../../ui'
import ConfirmDialog from '../ConfirmDialog'
import { SrBenchRequestError } from './api'
import { experimentApi, type Experiment } from './experimentApi'

export default function ExperimentDelete({
  experiment,
  disabled,
  onDeleted,
  onRefresh,
}: {
  experiment: Experiment
  disabled: boolean
  onDeleted: () => void
  onRefresh: () => void
}) {
  const [open, setOpen] = useState(false)
  const [pending, setPending] = useState(false)
  const [error, setError] = useState('')
  const [needsRefresh, setNeedsRefresh] = useState(false)
  const mounted = useRef(false)
  const sequence = useRef(0)
  const submitting = useRef(false)
  const trigger = useRef<HTMLButtonElement>(null)
  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
      sequence.current += 1
    }
  }, [])
  const hasActiveRuns = (experiment.active_run_count ?? 0) > 0
  async function remove() {
    if (submitting.current || disabled || experiment.active_run_count !== 0) return
    if (needsRefresh) {
      setOpen(false)
      onRefresh()
      return
    }
    const requestSequence = ++sequence.current
    const current = () => mounted.current && sequence.current === requestSequence
    submitting.current = true
    setPending(true)
    setError('')
    try {
      const result = await experimentApi.delete(experiment.id)
      if (!current()) return
      if (result.id !== experiment.id || result.deleted !== true || result.runs_deleted !== 0)
        throw new Error(
          'The deletion receipt could not be confirmed. Check the same experiment again.',
        )
      onDeleted()
    } catch (cause) {
      if (current()) {
        setError(cause instanceof Error ? cause.message : 'Could not confirm experiment deletion.')
        setNeedsRefresh(
          cause instanceof SrBenchRequestError && cause.code === 'experiment_active_runs',
        )
      }
    } finally {
      submitting.current = false
      if (current()) setPending(false)
    }
  }
  return (
    <>
      <Button
        ref={trigger}
        variant="destructive-ghost"
        size="sm"
        disabled={disabled || experiment.active_run_count !== 0}
        onClick={() => setOpen(true)}
      >
        <Trash2 aria-hidden="true" /> Delete experiment
      </Button>
      {hasActiveRuns && (
        <span className="basis-full text-right text-xs text-muted-foreground">
          Wait for active runs to finish before deleting.
        </span>
      )}
      <ConfirmDialog
        isOpen={open}
        title="Delete experiment?"
        description={
          <>
            Delete <strong>{experiment.name}</strong> and its run links? Saved runs, results, costs
            and artifacts will be kept.
          </>
        }
        confirmLabel={
          needsRefresh ? 'Refresh experiment' : error ? 'Retry deletion' : 'Delete experiment'
        }
        errorMessage={error}
        errorDetails={
          error && !needsRefresh ? (
            <p>Retry checks the same experiment. No evaluation runs will be deleted.</p>
          ) : undefined
        }
        pending={pending}
        pendingLabel="Deleting…"
        returnFocusRef={trigger}
        onCancel={() => setOpen(false)}
        onConfirm={remove}
      />
    </>
  )
}
