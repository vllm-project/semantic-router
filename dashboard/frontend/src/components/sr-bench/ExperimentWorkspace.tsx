import { useEffect, useRef, useState } from 'react'
import ProductIcon from '../ProductIcon'
import ProductLoadingState from '../ProductLoadingState'
import BenchSelect from './BenchSelect'
import { experimentApi, type Experiment, type ExperimentPage } from './experimentApi'
import {
  clearExperiment,
  readExperiment,
  saveExperiment,
  type PendingExperiment,
} from './experimentSubmission'
import type { ExperimentRunContext, Run } from './types'
import styles from './SrBench.module.css'
import layout from './ExperimentWorkspace.module.css'
import controls from './BenchControls.module.css'

const labels: Record<ExperimentRunContext['role'], string> = {
  baseline: 'Single-model baseline',
  initial: 'Starting recipe',
  candidate: 'Candidate recipe',
  preview: 'Routing check',
  smoke: 'Pipeline check',
  validation: 'Holdout validation',
  estimate: 'Offline estimate',
  recovery: 'Recovery attempt',
}
interface Props {
  actorID: string
  id?: string
  runs: Run[]
  canWrite: boolean
  canRun: boolean
  onSelect: (id?: string) => void
  onOpenRun: (id: string) => void
  onCompare: (baseline: string) => void
  onCandidate: (
    baseline: string,
    experiment: string,
    mode: 'live' | 'preview',
    role: ExperimentRunContext['role'],
  ) => void
}
export default function ExperimentWorkspace({
  id,
  actorID,
  runs,
  canWrite,
  canRun,
  onSelect,
  onOpenRun,
  onCompare,
  onCandidate,
}: Props) {
  const [experiments, setExperiments] = useState<Experiment[]>([])
  const [detail, setDetail] = useState<ExperimentPage | null>(null)
  const [cursors, setCursors] = useState([0])
  const [next, setNext] = useState<number | null>(null)
  const [loading, setLoading] = useState(true)
  const [pending, setPending] = useState(false)
  const [error, setError] = useState('')
  const [name, setName] = useState('')
  const [runID, setRunID] = useState('')
  const [role, setRole] = useState<ExperimentRunContext['role']>('candidate')
  const [hypothesis, setHypothesis] = useState('')
  const [revision, setRevision] = useState(0)
  const [saved] = useState(() => readExperiment(actorID))
  const [createRequest, setCreateRequest] = useState<PendingExperiment | null>(saved.saved)
  const mounted = useRef(false)
  const sequence = useRef(0)
  const saving = useRef(false)
  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
      sequence.current += 1
    }
  }, [])
  const cursor = cursors[cursors.length - 1]
  const selected = runs.find((run) => run.id === runID)
  const roles = selected?.experiment_roles ?? []
  useEffect(() => {
    const controller = new AbortController()
    setLoading(true)
    setError('')
    const loadingPage = id
      ? experimentApi.runs(id, cursor, controller.signal).then((value) => {
          if (!controller.signal.aborted) {
            setDetail(value)
            setNext(value.next_cursor)
          }
        })
      : experimentApi.list(cursor, controller.signal).then((value) => {
          if (!controller.signal.aborted) {
            setExperiments(value.experiments)
            setNext(value.next_cursor)
          }
        })
    void loadingPage
      .catch((cause) => {
        if (!controller.signal.aborted)
          setError(cause instanceof Error ? cause.message : 'Could not read experiments.')
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false)
      })
    return () => controller.abort()
  }, [id, cursor, revision])

  async function save() {
    if (saving.current || !canWrite || (!id && saved.error)) return
    const requestSequence = ++sequence.current
    const current = () => mounted.current && sequence.current === requestSequence
    saving.current = true
    setPending(true)
    setError('')
    try {
      if (id) {
        if (!roles.includes(role)) throw new Error('Choose an eligible role for this saved run.')
        await experimentApi.attach(id, runID, role, hypothesis)
        if (!current()) return
        setRunID('')
        setHypothesis('')
        setRevision((value) => value + 1)
      } else {
        const body: PendingExperiment = createRequest ?? {
          version: 1,
          actorID,
          name: name.trim(),
          key: crypto.randomUUID(),
        }
        saveExperiment(body)
        setCreateRequest(body)
        const result = await experimentApi.create(body.name, body.key)
        if (!current()) return
        if (!clearExperiment(body))
          throw new Error(
            'The saved experiment changed while this response was pending. Reload to reconcile its identity.',
          )
        setCreateRequest(null)
        onSelect(result.id)
      }
    } catch (cause) {
      if (current())
        setError(cause instanceof Error ? cause.message : 'Could not save the experiment.')
    } finally {
      saving.current = false
      if (current()) setPending(false)
    }
  }
  return (
    <section className={styles.panel} aria-label="Evaluation experiments">
      <div className={styles.sectionHeading}>
        <div>
          {id && (
            <button className={styles.backLink} onClick={() => onSelect()}>
              <ProductIcon name="arrow-left" /> All experiments
            </button>
          )}
          <h2>{id ? (detail?.experiment.name ?? 'Experiment') : 'Experiments'}</h2>
          <p className={styles.muted}>
            Keep your baseline, recipe hypotheses and validation evidence together. Each run remains
            independent.
          </p>
        </div>
        <button onClick={() => setRevision((value) => value + 1)} disabled={loading}>
          <ProductIcon name="refresh" /> Refresh
        </button>
      </div>
      {error && (
        <p className={styles.error} role="alert">
          {error}
        </p>
      )}
      {loading ? (
        <ProductLoadingState compact label="Loading experiment evidence…" />
      ) : (
        !error &&
        (id ? (
          <>
            {canRun && (
              <button
                className={controls.compactButton}
                onClick={() => onCandidate('', id, 'live', 'baseline')}
              >
                <ProductIcon name="plus" /> Create baseline
              </button>
            )}
            <ol className={layout.workflow} aria-label="Evaluation workflow">
              <li>
                <strong>1. Establish a baseline</strong>
                <span>Smoke checks, then a fixed Quick comparison.</span>
              </li>
              <li>
                <strong>2. Iterate the recipe</strong>
                <span>Preview routing, test candidates and compare saved results.</span>
              </li>
              <li>
                <strong>3. Validate on holdout</strong>
                <span>Freeze the recipe before Standard evaluation.</span>
              </li>
            </ol>
            <div className={layout.timeline}>
              {detail?.members.map((member) => {
                const run = runs.find((value) => value.id === member.run_id)
                const baseline = member.role === 'baseline' && run?.status === 'completed'
                return (
                  <article key={member.run_id} className={layout.entry}>
                    <div className={layout.marker}>
                      <ProductIcon name={member.role === 'preview' ? 'decision' : 'evaluation'} />
                    </div>
                    <div className={layout.entryBody}>
                      <span className={styles.eyebrow}>{labels[member.role]}</span>
                      <button className={layout.runLink} onClick={() => onOpenRun(member.run_id)}>
                        {run?.manifest.name ?? 'Saved evaluation'}{' '}
                        <ProductIcon name="arrow-right" />
                      </button>
                      <p className={styles.muted}>
                        {run
                          ? `${run.manifest.profile} · ${run.status} · ${run.progress.completed}/${run.progress.total} completed`
                          : 'Open the saved run to inspect its evidence.'}
                      </p>
                      {member.hypothesis && <p>{member.hypothesis}</p>}
                      {baseline && (
                        <div className={styles.actions}>
                          <button onClick={() => onCompare(member.run_id)}>
                            <ProductIcon name="chart" /> Compare iterations
                          </button>
                          {canRun && (
                            <>
                              <button
                                onClick={() => onCandidate(member.run_id, id, 'preview', 'preview')}
                              >
                                <ProductIcon name="decision" /> Preview candidate
                              </button>
                              <button
                                onClick={() =>
                                  onCandidate(
                                    member.run_id,
                                    id,
                                    'live',
                                    run.manifest.profile === 'standard'
                                      ? 'validation'
                                      : 'candidate',
                                  )
                                }
                              >
                                <ProductIcon name="play" /> Evaluate candidate
                              </button>
                            </>
                          )}
                        </div>
                      )}
                    </div>
                  </article>
                )
              })}
              {!detail?.members.length && (
                <p className={styles.muted}>
                  Link your starting runs below. Reuse a completed baseline to evaluate a candidate
                  on the same questions.
                </p>
              )}
            </div>
          </>
        ) : (
          <div className={layout.cards}>
            {experiments.map((item) => (
              <button
                className={layout.experimentCard}
                key={item.id}
                onClick={() => onSelect(item.id)}
              >
                <ProductIcon name="evaluation" />
                <strong>{item.name}</strong>
                <span>Updated {new Date(item.updated_at).toLocaleDateString()}</span>
                <ProductIcon name="arrow-right" />
              </button>
            ))}
            {!experiments.length && (
              <p className={styles.muted}>
                Create an experiment to organize a recipe improvement loop.
              </p>
            )}
          </div>
        ))
      )}
      {(cursors.length > 1 || next !== null) && (
        <nav className={styles.pagination} aria-label="Experiment pages">
          <button
            className={controls.compactButton}
            disabled={loading || cursors.length === 1}
            onClick={() => setCursors((values) => values.slice(0, -1))}
          >
            <ProductIcon name="chevron-left" /> Previous
          </button>
          <span>Page {cursors.length}</span>
          <button
            className={controls.compactButton}
            disabled={loading || !!error || next === null}
            onClick={() => next !== null && setCursors((values) => [...values, next])}
          >
            Next <ProductIcon name="chevron-right" />
          </button>
        </nav>
      )}
      {canWrite && (
        <div className={layout.editor}>
          <h3>{id ? 'Link saved evidence' : 'New experiment'}</h3>
          {id ? (
            <>
              <BenchSelect
                label="Saved run"
                value={runID}
                disabled={pending}
                searchable
                options={runs
                  .filter(
                    (run) =>
                      !detail?.members.some((member) => member.run_id === run.id) &&
                      run.experiment_roles?.length,
                  )
                  .map((run) => ({
                    value: run.id,
                    label: run.manifest.name,
                    description: `${run.manifest.profile} · ${run.status}`,
                  }))}
                onChange={(value) => {
                  setRunID(value)
                  setRole(runs.find((run) => run.id === value)!.experiment_roles![0])
                }}
              />
              {roles.length > 0 && (
                <BenchSelect
                  label="Role in this experiment"
                  value={role}
                  disabled={pending}
                  options={roles.map((value) => ({ value, label: labels[value] }))}
                  onChange={(value) => setRole(value as ExperimentRunContext['role'])}
                />
              )}
              <label>
                Hypothesis or note
                <textarea
                  disabled={pending}
                  value={hypothesis}
                  maxLength={2000}
                  onChange={(event) => setHypothesis(event.target.value)}
                  placeholder="What changed, and what do you expect?"
                />
              </label>
            </>
          ) : saved.error ? (
            <p className={styles.error} role="alert">
              {saved.error}
            </p>
          ) : createRequest ? (
            <div className={styles.notice}>
              <strong>Experiment submission needs reconciliation</strong>
              <p>{createRequest.name}</p>
              <p>
                Check the same saved submission to retrieve or finish this experiment. No evaluation
                is started.
              </p>
            </div>
          ) : (
            <label>
              Experiment name
              <input
                value={name}
                maxLength={160}
                onChange={(event) => setName(event.target.value)}
                placeholder="Balance quality and cost"
              />
            </label>
          )}
          <button
            className={`${styles.primary} ${controls.compactButton}`}
            disabled={
              pending ||
              (id
                ? !runID || !roles.includes(role)
                : !!saved.error || (!createRequest && !name.trim()))
            }
            onClick={() => void save()}
          >
            <ProductIcon name={id ? 'link' : 'plus'} />
            {pending
              ? 'Saving…'
              : id
                ? 'Link run'
                : createRequest
                  ? 'Check or create same experiment'
                  : 'Create experiment'}
          </button>
        </div>
      )}
    </section>
  )
}
