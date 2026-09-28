import { useEffect, useRef, useState } from 'react'
import { benchApi, SrBenchRequestError } from './api'
import BenchSelect from './BenchSelect'
import RunOptionsStatus from './RunOptionsStatus'
import useRunOptions from './useRunOptions'
import { profileTitle } from './datasetPresentation'
import { clearReplay, readReplay, saveReplay, type PendingReplay } from './replaySubmission'
import type { Run } from './types'
import styles from './SrBench.module.css'

type Props = { actorID: string; canRun: boolean; onCreated: (run: Run) => void }

export default function ReplayComposer(props: Props) {
  return <ScopedReplayComposer key={props.actorID} {...props} />
}
function ScopedReplayComposer({ actorID, canRun, onCreated }: Props) {
  const [saved] = useState(() => readReplay(actorID))
  const [submission, setSubmission] = useState<PendingReplay | null>(saved.saved)
  const [baseline, setBaseline] = useState('')
  const [preview, setPreview] = useState('')
  const [pending, setPending] = useState(false)
  const [error, setError] = useState('')
  const mounted = useRef(false)
  const submitSequence = useRef(0)
  const submitting = useRef(false)
  const baselines = useRunOptions('replay', undefined, !submission && !saved.error)
  const previews = useRunOptions(
    'replay',
    baseline || undefined,
    !!baseline && !submission && !saved.error,
  )
  useEffect(() => {
    mounted.current = true
    const pendingSequence = submitSequence
    return () => {
      mounted.current = false
      pendingSequence.current++
    }
  }, [])
  const selected = previews.items.find((item) => item.run_id === preview)
  const ready =
    !!previews.baseline &&
    !!selected &&
    previews.loaded &&
    !previews.loading &&
    !previews.error &&
    !baselines.error
  const empty =
    baselines.loaded &&
    !baselines.loading &&
    !baselines.error &&
    !baselines.next &&
    !baselines.items.length
  async function replay() {
    if (!canRun || saved.error || submitting.current || (!submission && !ready)) return
    submitting.current = true
    const sequence = ++submitSequence.current
    const current = () => mounted.current && submitSequence.current === sequence
    const body: PendingReplay = submission ?? {
      version: 1,
      actorID,
      request: {
        baseline_run_id: baseline,
        preview_run_id: preview,
        idempotency_key: crypto.randomUUID(),
      },
    }
    setPending(true)
    setError('')
    try {
      saveReplay(body)
      setSubmission(body)
      const run = await benchApi.replay(body.request)
      if (!current()) return
      if (!clearReplay(body))
        throw new Error(
          'The saved estimate changed while this response was pending. Reload to reconcile the current submission.',
        )
      setSubmission(null)
      onCreated(run)
    } catch (cause) {
      if (!current()) return
      if (
        cause instanceof SrBenchRequestError &&
        cause.code === 'replay_ineligible' &&
        cause.dispatchStarted === false
      ) {
        try {
          if (!clearReplay(body)) throw new Error('Saved submission changed')
          setSubmission(null)
          setBaseline('')
          setPreview('')
          baselines.refresh()
        } catch {
          setError(
            'The service rejected this pair, but its saved identity could not be cleared safely. Reload to reconcile it.',
          )
          return
        }
      }
      setError(
        cause instanceof Error
          ? cause.message
          : 'The estimate response is unavailable. Reconcile this same submission.',
      )
    } finally {
      submitting.current = false
      if (current()) setPending(false)
    }
  }

  return (
    <section className={styles.replayComposer} aria-label="Estimate a routing change">
      {saved.error ? (
        <p role="alert" className={styles.error}>
          {saved.error}
        </p>
      ) : submission ? (
        <div className={styles.notice}>
          <h3>Estimate submission needs reconciliation</h3>
          <p>
            The exact pair and submission ID are saved for this account in this tab. Checking again
            retrieves or completes the same estimate.
          </p>
          <button disabled={!canRun || pending} onClick={() => void replay()}>
            {pending ? 'Reconciling…' : 'Check or submit same estimate'}
          </button>
        </div>
      ) : empty ? (
        <p className={styles.optionEmpty} role="status">
          {baselines.scanLimited
            ? 'No verified replay pairs available. Some saved evidence exceeded the verification limit.'
            : 'No compatible saved pairs. Replay needs a stateless routing preview and complete single-model answers on the same frozen protocol.'}
        </p>
      ) : (
        <>
          <p className={styles.muted}>
            Reuse saved answers without new model calls. Live evaluation is still needed to validate
            a routing change.
          </p>
          <RunOptionsStatus
            {...baselines}
            label="saved baselines"
            onRetry={baselines.reload}
            onMore={baselines.loadMore}
          />
          {!!baselines.items.length && (
            <div className={styles.optionSelectRow}>
              <BenchSelect
                label="Saved single-model baseline"
                value={baseline}
                disabled={pending || baselines.loading || !!baselines.error}
                searchable
                placeholder="Choose compatible saved answers"
                options={baselines.items.map((item) => ({
                  value: item.run_id,
                  label: item.name,
                  description: `${profileTitle(item.profile)} · ${item.case_count} cases`,
                }))}
                onChange={(id) => {
                  setBaseline(id)
                  setPreview('')
                  setError('')
                }}
              />
              {baseline && (
                <BenchSelect
                  label="Routing preview"
                  value={preview}
                  disabled={pending || previews.loading || !!previews.error}
                  searchable
                  placeholder="Choose a compatible preview"
                  onChange={setPreview}
                  options={previews.items.map((item) => ({
                    value: item.run_id,
                    label: item.name,
                    description: `${profileTitle(item.profile)} · ${item.case_count} cases`,
                  }))}
                />
              )}
            </div>
          )}
          {baseline && (
            <RunOptionsStatus
              {...previews}
              label="previews"
              onRetry={previews.reload}
              onMore={previews.loadMore}
            />
          )}
          {baseline &&
            previews.loaded &&
            !previews.loading &&
            !previews.error &&
            !previews.items.length &&
            !previews.next && (
              <p className={styles.optionEmpty} role="status">
                {previews.scanLimited
                  ? 'No verified previews for this baseline. Some evidence exceeded the verification limit.'
                  : 'This saved baseline no longer has a compatible preview. Choose another baseline.'}
              </p>
            )}
          {selected && (
            <div className={styles.actions}>
              <button
                className={styles.primary}
                disabled={!canRun || pending || !ready}
                onClick={() => void replay()}
              >
                {pending ? 'Creating estimate…' : 'Create offline estimate'}
              </button>
            </div>
          )}
        </>
      )}
      {error && (
        <p role="alert" className={styles.error}>
          {error}
        </p>
      )}
    </section>
  )
}
