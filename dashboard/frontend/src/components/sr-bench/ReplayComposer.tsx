import { useEffect, useRef, useState } from 'react'
import ProductLoadingState from '../ProductLoadingState'
import { benchApi, SrBenchRequestError } from './api'
import BenchSelect from './BenchSelect'
import BenchPagination from './BenchPagination'
import { profileTitle } from './datasetPresentation'
import { clearReplay, readReplay, saveReplay, type PendingReplay } from './replaySubmission'
import type { ReplayOption, Run } from './types'
import styles from './SrBench.module.css'

type Props = { runs: Run[]; actorID: string; canRun: boolean; onCreated: (run: Run) => void }

export default function ReplayComposer(props: Props) {
  return <ScopedReplayComposer key={props.actorID} {...props} />
}

function ScopedReplayComposer({ runs, actorID, canRun, onCreated }: Props) {
  const [saved] = useState(() => readReplay(actorID))
  const [submission, setSubmission] = useState<PendingReplay | null>(saved.saved)
  const [baseline, setBaseline] = useState('')
  const [preview, setPreview] = useState('')
  const [options, setOptions] = useState<ReplayOption[]>([])
  const [nextCursor, setNextCursor] = useState<string | null>(null)
  const [loaded, setLoaded] = useState(false)
  const [loading, setLoading] = useState(false)
  const [readError, setReadError] = useState('')
  const [failedCursor, setFailedCursor] = useState<string | undefined>()
  const [unavailablePage, setUnavailablePage] = useState(0)
  const [pending, setPending] = useState(false)
  const [error, setError] = useState('')
  const mounted = useRef(false)
  const readSequence = useRef(0)
  const submitSequence = useRef(0)
  const submitting = useRef(false)
  const reading = useRef<AbortController | null>(null)
  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
      readSequence.current += 1
      submitSequence.current += 1
      reading.current?.abort()
    }
  }, [])

  async function loadOptions(id: string, after?: string) {
    reading.current?.abort()
    const controller = new AbortController()
    reading.current = controller
    const sequence = ++readSequence.current
    const current = () =>
      mounted.current && !controller.signal.aborted && readSequence.current === sequence
    setLoading(true)
    setReadError('')
    setFailedCursor(after)
    try {
      const value = await benchApi.replayOptions(id, after, controller.signal)
      if (!current()) return
      if (
        value.baseline.run_id !== id ||
        value.options.length > 10 ||
        (value.has_more && (!value.next_cursor || value.next_cursor === after))
      )
        throw new Error(
          'The service returned an inconsistent preview page. Review compatibility again.',
        )
      setOptions((previous) => {
        const merged = new Map(
          (after ? previous : []).map((option) => [option.preview_run_id, option]),
        )
        value.options.forEach((option) => merged.set(option.preview_run_id, option))
        return [...merged.values()]
      })
      setNextCursor(value.has_more ? value.next_cursor : null)
      setLoaded(true)
    } catch (cause) {
      if (current())
        setReadError(
          cause instanceof Error ? cause.message : 'Preview compatibility could not be checked.',
        )
    } finally {
      if (current()) setLoading(false)
    }
  }

  const eligible = options.filter((option) => option.eligible)
  const unavailable = options.filter((option) => !option.eligible)
  const selected = eligible.find((option) => option.preview_run_id === preview)
  const ready = !!baseline && !!selected && loaded && !loading && !readError
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
          setBaseline(body.request.baseline_run_id)
          setPreview('')
          setOptions([])
          setLoaded(false)
          setNextCursor(null)
          setReadError('Compatibility changed. Check previews again before choosing a pair.')
          setFailedCursor(undefined)
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
    <section className={styles.replayComposer} aria-labelledby="replay-title">
      <h2 id="replay-title">Estimate a routing change</h2>
      <p>
        Reuse saved single-model answers with a compatible routing preview. No model generation
        occurs; live evaluation is still needed to validate quality, cost and latency.
      </p>
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
      ) : (
        <>
          <section className={styles.comparisonStep} aria-labelledby="replay-baseline-step">
            <div className={styles.stepHeading}>
              <span>1</span>
              <div>
                <h3 id="replay-baseline-step">Choose saved answers</h3>
                <p>Use a completed live run containing single-model results.</p>
              </div>
            </div>
            <BenchSelect
              label="Saved single-model baseline"
              value={baseline}
              disabled={pending}
              searchable
              placeholder="Select saved answers"
              onChange={(id) => {
                setBaseline(id)
                setPreview('')
                setOptions([])
                setLoaded(false)
                setNextCursor(null)
                setUnavailablePage(0)
                setError('')
                void loadOptions(id)
              }}
              options={runs
                .filter(
                  (run) =>
                    run.status === 'completed' &&
                    run.manifest.mode === 'live' &&
                    run.manifest.targets.some((target) => target.kind === 'single'),
                )
                .map((run) => ({
                  value: run.id,
                  label: run.manifest.name,
                  description: profileTitle(run.manifest.profile),
                }))}
            />
          </section>
          <section className={styles.comparisonStep} aria-labelledby="replay-preview-step">
            <div className={styles.stepHeading}>
              <span>2</span>
              <div>
                <h3 id="replay-preview-step">Choose a compatible preview</h3>
                <p>
                  The service checks frozen cases, request settings, saved answers and routing
                  behavior.
                </p>
              </div>
            </div>
            {!baseline ? (
              <p className={styles.emptyState}>Choose saved answers first.</p>
            ) : (
              <>
                {loading && <ProductLoadingState compact label="Checking preview compatibility…" />}
                {readError && (
                  <div role="alert" className={styles.error}>
                    <p>{readError}</p>
                    <button
                      disabled={loading}
                      onClick={() => void loadOptions(baseline, failedCursor)}
                    >
                      Retry compatibility check
                    </button>
                  </div>
                )}
                <BenchSelect
                  label="Routing preview"
                  value={preview}
                  disabled={pending || loading || !!readError || !loaded}
                  searchable
                  placeholder="Select a compatible preview"
                  onChange={setPreview}
                  options={eligible.map((option) => ({
                    value: option.preview_run_id,
                    label: option.name,
                    description: `${profileTitle(option.profile)} · ${option.case_count} cases`,
                  }))}
                />
                {loaded && !loading && !readError && !eligible.length && (
                  <p className={styles.emptyState}>
                    {nextCursor
                      ? 'No compatible previews among the loaded runs. Load more to continue checking.'
                      : 'No compatible previews. Create a stateless route preview on the same frozen cases and request settings.'}
                  </p>
                )}
                {selected && !readError && (
                  <p className={styles.muted}>
                    {selected.case_count} cases · Eligible for an offline estimate. Compatibility is
                    checked again on submission.
                  </p>
                )}
                {!!unavailable.length && (
                  <details className={styles.details}>
                    <summary>Unavailable previews ({unavailable.length} loaded)</summary>
                    <ul className={styles.replayUnavailable}>
                      {unavailable
                        .slice(unavailablePage * 5, unavailablePage * 5 + 5)
                        .map((option) => (
                          <li key={option.preview_run_id}>
                            <strong>{option.name}</strong>
                            <span>
                              {option.reasons.map((reason) => reason.message).join(' ') ||
                                'This preview is not compatible.'}
                            </span>
                          </li>
                        ))}
                    </ul>
                    <BenchPagination
                      label="Unavailable previews"
                      total={unavailable.length}
                      page={unavailablePage}
                      pageSize={5}
                      onChange={setUnavailablePage}
                    />
                  </details>
                )}
                {nextCursor && !readError && (
                  <button
                    disabled={loading || pending}
                    onClick={() => void loadOptions(baseline, nextCursor)}
                  >
                    Load more previews
                  </button>
                )}
              </>
            )}
          </section>
          <button
            className={styles.primary}
            disabled={!canRun || pending || !ready}
            onClick={() => void replay()}
          >
            {pending ? 'Creating estimate…' : 'Create offline estimate'}
          </button>
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
