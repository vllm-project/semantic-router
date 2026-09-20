import { useEffect, useRef, useState } from 'react'
import ProductIcon from '../ProductIcon'
import ProductLoadingState from '../ProductLoadingState'
import BenchSelect from './BenchSelect'
import ExperimentDelete from './ExperimentDelete'
import ExperimentEvidence from './ExperimentEvidence'
import { experimentRoleLabels } from './experimentEvidencePresentation'
import { canReuseBaseline } from './baselineReuse'
import { active } from './model'
import { RunStatus } from './RunList'
import useExperimentMembers from './useExperimentMembers'
import { SrBenchRequestError } from './api'
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
  const [readError, setReadError] = useState('')
  const [notice, setNotice] = useState('')
  const [name, setName] = useState('')
  const [runID, setRunID] = useState('')
  const [role, setRole] = useState<ExperimentRunContext['role']>('candidate')
  const [hypothesis, setHypothesis] = useState('')
  const [revision, setRevision] = useState(0)
  const [showCreate, setShowCreate] = useState(false)
  const [showSaved, setShowSaved] = useState(false)
  const [referenceID, setReferenceID] = useState('')
  const [activeRunID, setActiveRunID] = useState('')
  const membership = useExperimentMembers(id, revision)
  const knownRoles = Object.fromEntries(
    membership.members.map((member) => [member.run_id, member.role]),
  )
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
    setReadError('')
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
          setReadError(cause instanceof Error ? cause.message : 'Could not read experiments.')
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false)
      })
    return () => controller.abort()
  }, [id, cursor, revision])

  async function save() {
    if (saving.current || !canWrite || (id && !membership.complete) || (!id && saved.error)) return
    const requestSequence = ++sequence.current
    const current = () => mounted.current && sequence.current === requestSequence
    saving.current = true
    setPending(true)
    setError('')
    setNotice('')
    try {
      if (id) {
        if (!roles.includes(role)) throw new Error('Choose an eligible role for this saved run.')
        await experimentApi.attach(id, runID, role, hypothesis)
        if (!current()) return
        setNotice('Saved run added to this experiment.')
        setRunID('')
        setHypothesis('')
        setShowSaved(false)
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
        let result: Experiment
        try {
          result = await experimentApi.create(body.name, body.key)
        } catch (cause) {
          if (!(cause instanceof SrBenchRequestError) || cause.code !== 'experiment_deleted')
            throw cause
          if (!current()) return
          if (!clearExperiment(body))
            throw new Error('The saved submission changed. Reload to reconcile its identity.')
          setCreateRequest(null)
          setName('')
          setNotice('That experiment was deleted. Enter a name to create a new experiment.')
          return
        }
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
  const references = runs.filter(
    (run) => knownRoles[run.id] === 'baseline' && canReuseBaseline(run),
  )
  const reference = references.find((run) => run.id === referenceID) ?? references[0]
  const activeRuns = runs.filter((run) => knownRoles[run.id] && active(run.status))
  const activeRun =
    activeRuns.find((run) => run.id === activeRunID) ??
    activeRuns.find((run) => knownRoles[run.id] === 'baseline') ??
    activeRuns[0]
  const loadedRunIDs = new Set(runs.map((run) => run.id))
  const activeRunCount =
    membership.complete && membership.members.every((member) => loadedRunIDs.has(member.run_id))
      ? activeRuns.length
      : undefined
  const hasStartingRecipe = Object.values(knownRoles).includes('initial')
  const hasRecipeResults = runs.some(
    (run) =>
      ['initial', 'candidate', 'validation'].includes(knownRoles[run.id]) &&
      ['completed', 'failed'].includes(run.status) &&
      run.progress.total > 0 &&
      run.progress.completed + run.progress.failed === run.progress.total,
  )
  const compareAction = reference && (
    <button
      className={`${hasRecipeResults && !activeRun ? styles.primary : ''} ${controls.compactButton}`}
      onClick={() => onCompare(reference.id)}
    >
      <ProductIcon name="chart" /> Compare results
    </button>
  )
  const runCount = detail?.experiment.run_count
  const empty = runCount === 0
  const showCreateForm =
    !id && (showCreate || !experiments.length || !!createRequest || !!saved.error)
  const openSaved = () => setShowSaved(true)

  const editor = canWrite && (
    <div className={layout.editor}>
      <div>
        <h3>{id ? 'Add saved results' : 'New experiment'}</h3>
        <p className={styles.muted}>
          {id
            ? 'Reuse a run you already have. Adding it here does not rerun the evaluation.'
            : 'Give this comparison a name. You will choose models and questions in the next step.'}
        </p>
      </div>
      {id ? (
        <>
          <BenchSelect
            label="Saved run"
            value={runID}
            disabled={pending}
            searchable
            options={runs
              .filter((run) => !knownRoles[run.id] && run.experiment_roles?.length)
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
          {roles.length > 1 && (
            <BenchSelect
              label="Use this result as"
              value={role}
              disabled={pending}
              options={roles.map((value) => ({ value, label: experimentRoleLabels[value] }))}
              onChange={(value) => setRole(value as ExperimentRunContext['role'])}
            />
          )}
          {roles.length === 1 && <p className={styles.muted}>{experimentRoleLabels[roles[0]]}</p>}
          <label>
            What changed? (optional)
            <textarea
              disabled={pending}
              value={hypothesis}
              maxLength={2000}
              onChange={(event) => setHypothesis(event.target.value)}
              placeholder="For example: prefer the less expensive model for simple questions."
            />
          </label>
        </>
      ) : saved.error ? (
        <p className={styles.error} role="alert">
          {saved.error}
        </p>
      ) : createRequest ? (
        <div className={styles.notice}>
          <strong>Check your previous creation request</strong>
          <p>{createRequest.name}</p>
          <p>
            The response was not confirmed. Check the same request before creating another
            experiment.
          </p>
        </div>
      ) : (
        <label>
          Experiment name
          <input
            value={name}
            maxLength={160}
            onChange={(event) => setName(event.target.value)}
            placeholder="Balance · reasoning and cost"
          />
        </label>
      )}
      <div className={styles.actions}>
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
              ? 'Add run'
              : createRequest
                ? 'Check creation request'
                : 'Create experiment'}
        </button>
        {!createRequest && (id || experiments.length > 0) && (
          <button
            className={controls.compactButton}
            disabled={pending}
            onClick={() => (id ? setShowSaved(false) : setShowCreate(false))}
          >
            {id ? 'Close' : 'Cancel'}
          </button>
        )}
      </div>
    </div>
  )

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
            {id
              ? 'Compare your recipe versions against the same single-model reference.'
              : 'One place to compare single models, test recipe changes and see whether they improve quality or cost.'}
          </p>
        </div>
        <div className={styles.actions}>
          <button
            className={controls.compactButton}
            onClick={() => setRevision((value) => value + 1)}
            disabled={loading || pending}
          >
            <ProductIcon name="refresh" /> Refresh
          </button>
          {!id && canWrite && !showCreateForm && (
            <button
              className={`${styles.primary} ${controls.compactButton}`}
              onClick={() => setShowCreate(true)}
            >
              <ProductIcon name="plus" /> New experiment
            </button>
          )}
          {id && detail && canWrite && !loading && !readError && (
            <ExperimentDelete
              key={`${actorID}:${id}`}
              experiment={detail.experiment}
              disabled={pending}
              onDeleted={() => onSelect()}
              onRefresh={() => setRevision((value) => value + 1)}
            />
          )}
        </div>
      </div>
      {readError && (
        <p className={styles.error} role="alert">
          {readError}
        </p>
      )}
      {error && (
        <p className={styles.error} role="alert">
          {error}
        </p>
      )}
      {notice && (
        <p className={styles.notice} role="status">
          {notice}
        </p>
      )}
      {loading ? (
        <ProductLoadingState compact label="Loading experiment…" />
      ) : (
        !readError &&
        (id ? (
          <>
            <div className={layout.overview}>
              <div className={layout.journey} aria-label="How an experiment works">
                <span>
                  <ProductIcon name="model" /> Single models
                </span>
                <ProductIcon name="arrow-right" />
                <span>
                  <ProductIcon name="mixture" /> Starting recipe
                </span>
                <ProductIcon name="arrow-right" />
                <span>
                  <ProductIcon name="chart" /> Recipe versions
                </span>
              </div>
              <span className={layout.count}>
                {runCount !== undefined && `${runCount} saved ${runCount === 1 ? 'run' : 'runs'}`}
                {!!activeRunCount && ` · ${activeRunCount} active`}
              </span>
            </div>
            {membership.loading ? (
              <ProductLoadingState compact label="Finding the next step…" />
            ) : membership.error ? (
              <div className={styles.notice} role="alert">
                <p>Could not load all runs in this experiment. {membership.error}</p>
                <button className={controls.compactButton} onClick={membership.reload}>
                  Reload experiment runs
                </button>
              </div>
            ) : (
              membership.complete && (
                <div className={layout.nextStep}>
                  <div className={layout.nextHeading}>
                    <ProductIcon name={activeRun ? 'logs' : reference ? 'mixture' : 'model'} />
                    <div>
                      <h3>
                        {activeRun
                          ? knownRoles[activeRun.id] === 'baseline'
                            ? 'Reference evaluation in progress'
                            : 'Evaluation in progress'
                          : empty
                            ? 'Start with your single models'
                            : reference
                              ? hasRecipeResults
                                ? 'Review your recipe versions'
                                : 'Test a recipe on the same questions'
                              : 'Choose your reference results'}
                      </h3>
                      <p>
                        {activeRun
                          ? 'Open the run to follow response activity and results as attempts finish.'
                          : empty
                            ? 'Run the connected models on shared questions, or use results you already have.'
                            : reference
                              ? hasRecipeResults
                                ? 'Compare saved results to see how quality and cost changed. Test another version when you have a change to evaluate.'
                                : 'A reference fixes the questions and settings. Measure your starting recipe, then compare changes against your best single model.'
                              : 'Use a finished single-model run as the reference for recipe versions. Running evaluations remain visible below.'}
                      </p>
                    </div>
                  </div>
                  {activeRun && (
                    <section className={layout.activeRun} aria-label="Active evaluation">
                      {activeRuns.length > 1 ? (
                        <BenchSelect
                          label="Active run"
                          value={activeRun.id}
                          options={activeRuns.map((run) => ({
                            value: run.id,
                            label: run.manifest.name,
                            description: `${experimentRoleLabels[knownRoles[run.id]]} · ${run.status}`,
                          }))}
                          onChange={setActiveRunID}
                        />
                      ) : (
                        <strong>{activeRun.manifest.name}</strong>
                      )}
                      <span className={styles.muted}>
                        {experimentRoleLabels[knownRoles[activeRun.id]]}
                        {activeRuns.length > 1 && ` · ${activeRuns.length} active runs`}
                      </span>
                      <div className={layout.activeProgress}>
                        <RunStatus status={activeRun.status} />
                        <span>
                          {activeRun.progress.completed.toLocaleString()} /{' '}
                          {activeRun.progress.total.toLocaleString()} attempts completed
                        </span>
                        <span>{activeRun.progress.failed.toLocaleString()} failed</span>
                      </div>
                      <div className={styles.actions}>
                        <button
                          className={`${styles.primary} ${controls.compactButton}`}
                          onClick={() => onOpenRun(activeRun.id)}
                        >
                          <ProductIcon name="arrow-right" /> Open run
                        </button>
                      </div>
                    </section>
                  )}
                  {reference ? (
                    <>
                      {references.length > 1 ? (
                        <BenchSelect
                          label="Single-model reference"
                          value={reference.id}
                          options={references.map((run) => ({
                            value: run.id,
                            label: run.manifest.name,
                            description: `${run.manifest.profile} · ${run.status}`,
                          }))}
                          onChange={setReferenceID}
                        />
                      ) : (
                        <div className={layout.reference}>
                          <span>Single-model reference</span>
                          <button onClick={() => onOpenRun(reference.id)}>
                            {reference.manifest.name} <ProductIcon name="arrow-right" />
                          </button>
                        </div>
                      )}
                      <div className={styles.actions}>
                        {hasRecipeResults && compareAction}
                        {canRun && (
                          <>
                            <button
                              className={`${hasRecipeResults || activeRun ? '' : styles.primary} ${controls.compactButton}`}
                              onClick={() =>
                                onCandidate(
                                  reference.id,
                                  id,
                                  'live',
                                  reference.manifest.profile === 'standard'
                                    ? 'validation'
                                    : hasStartingRecipe
                                      ? 'candidate'
                                      : 'initial',
                                )
                              }
                            >
                              <ProductIcon name="play" />{' '}
                              {reference.manifest.profile === 'standard'
                                ? 'Validate final recipe'
                                : hasStartingRecipe
                                  ? 'Test another recipe'
                                  : 'Evaluate starting recipe'}
                            </button>
                            <button
                              className={controls.compactButton}
                              onClick={() => onCandidate(reference.id, id, 'preview', 'preview')}
                            >
                              <ProductIcon name="decision" /> Check routing first
                            </button>
                          </>
                        )}
                        {!hasRecipeResults && compareAction}
                      </div>
                    </>
                  ) : (
                    <div className={styles.actions}>
                      {canRun && !activeRun && (
                        <button
                          className={`${styles.primary} ${controls.compactButton}`}
                          onClick={() => onCandidate('', id, 'live', 'baseline')}
                        >
                          <ProductIcon name="play" /> Run single models
                        </button>
                      )}
                      {canWrite && (
                        <button className={controls.compactButton} onClick={openSaved}>
                          <ProductIcon name="link" /> Use saved results
                        </button>
                      )}
                    </div>
                  )}
                </div>
              )
            )}
            {membership.complete && showSaved && editor}
            <div className={layout.resultsHeading}>
              <div>
                <h3>Saved results</h3>
                <p className={styles.muted}>
                  Open a run for its scores, costs and questions. Compare results for the difference
                  between versions.
                </p>
              </div>
              {canWrite && membership.complete && !empty && !showSaved && (
                <button className={controls.compactButton} onClick={openSaved}>
                  <ProductIcon name="link" /> Add saved results
                </button>
              )}
            </div>
            <ExperimentEvidence
              members={detail?.members ?? []}
              runs={runs}
              onOpenRun={onOpenRun}
              hasMore={next !== null}
            />
          </>
        ) : (
          <>
            {showCreateForm && editor}
            {!!experiments.length && (
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
              </div>
            )}
            {!experiments.length && (
              <div className={layout.emptyGuide}>
                <ProductIcon name="mixture" />
                <p>
                  An experiment groups related evaluations. Create one, add a single-model
                  reference, then test your recipe on the same questions.
                </p>
                <span>Creating the experiment does not start any evaluation.</span>
              </div>
            )}
          </>
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
            disabled={loading || !!readError || next === null}
            onClick={() => next !== null && setCursors((values) => [...values, next])}
          >
            Next <ProductIcon name="chevron-right" />
          </button>
        </nav>
      )}
    </section>
  )
}
