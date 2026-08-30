import { type FormEvent, useEffect, useMemo, useState } from 'react'

import type {
  EvaluationChangeProfileId,
  CreateEvaluationRunRequest,
  EvaluationCatalog,
  EvaluationMode,
  EvaluationRun,
  EvaluationTrackId,
} from '../../types/evaluationPlane'
import { TRACK_PRESENTATION } from '../../types/evaluationPlane'
import {
  gateApplicabilityForProfile,
  SUPPORTED_GATE_CONTRACT_VERSION,
} from './evaluationGateContract'
import { evidenceRank } from './evaluationPresentation'
import styles from './EvaluationForm.module.css'

interface EvaluationExperimentFormProps {
  catalog: EvaluationCatalog
  runs: EvaluationRun[]
  canCreate: boolean
  canAutoStart: boolean
  pending: boolean
  onSubmit: (request: CreateEvaluationRunRequest) => Promise<boolean>
}

function supportedTracks(
  catalog: EvaluationCatalog,
  targetID: string,
  mode: EvaluationMode,
): EvaluationTrackId[] {
  const target = catalog.targets.find((candidate) => candidate.id === targetID)
  if (!target || !target.modes.includes(mode)) return []
  return catalog.tracks
    .filter((track) => track.modes.includes(mode) && target.track_ids.includes(track.id))
    .map((track) => track.id)
}

export default function EvaluationExperimentForm({
  catalog,
  runs,
  canCreate,
  canAutoStart,
  pending,
  onSubmit,
}: EvaluationExperimentFormProps) {
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [mode, setMode] = useState<EvaluationMode>('replay')
  const [changeProfile, setChangeProfile] = useState<EvaluationChangeProfileId | ''>(
    catalog.change_profiles[0]?.id || '',
  )
  const [targetID, setTargetID] = useState(catalog.targets[0]?.id || '')
  const [suiteIDs, setSuiteIDs] = useState<string[]>(
    catalog.suites[0] ? [catalog.suites[0].id] : [],
  )
  const [trackIDs, setTrackIDs] = useState<EvaluationTrackId[]>(catalog.suites[0]?.track_ids || [])
  const [sampleLimit, setSampleLimit] = useState(100)
  const [concurrency, setConcurrency] = useState(4)
  const [seed, setSeed] = useState(42)
  const [baselineRunID, setBaselineRunID] = useState('')
  const [autoStart, setAutoStart] = useState(canAutoStart)
  const [validationError, setValidationError] = useState('')

  const availableTrackIDs = useMemo(
    () => supportedTracks(catalog, targetID, mode),
    [catalog, mode, targetID],
  )
  const compatibleSuites = useMemo(
    () =>
      catalog.suites.filter(
        (suite) =>
          suite.modes.includes(mode) &&
          suite.track_ids.every((trackID) => availableTrackIDs.includes(trackID)),
      ),
    [availableTrackIDs, catalog.suites, mode],
  )
  const completedRuns = useMemo(
    () => runs.filter((run) => run.status === 'completed' && run.change_profile === changeProfile),
    [changeProfile, runs],
  )
  const selectedChangeProfile = catalog.change_profiles.find(
    (profile) => profile.id === changeProfile,
  )
  const gateApplicability =
    changeProfile && catalog.gate_contract_version === SUPPORTED_GATE_CONTRACT_VERSION
      ? gateApplicabilityForProfile(changeProfile)
      : []
  const evidenceLevel = compatibleSuites
    .filter((suite) => suiteIDs.includes(suite.id))
    .map((suite) => suite.evidence_level)
    .sort((left, right) => evidenceRank(right) - evidenceRank(left))[0]

  useEffect(() => {
    const compatibleTarget = catalog.targets.find(
      (target) => target.id === targetID && target.modes.includes(mode) && target.healthy !== false,
    )
    if (!compatibleTarget) {
      setTargetID(
        catalog.targets.find((target) => target.modes.includes(mode) && target.healthy !== false)
          ?.id || '',
      )
    }
  }, [catalog.targets, mode, targetID])

  useEffect(() => {
    setTrackIDs((current) => current.filter((trackID) => availableTrackIDs.includes(trackID)))
    setSuiteIDs((current) =>
      current.filter((suiteID) => compatibleSuites.some((suite) => suite.id === suiteID)),
    )
  }, [availableTrackIDs, compatibleSuites])

  useEffect(() => {
    if (!catalog.change_profiles.some((profile) => profile.id === changeProfile)) {
      setChangeProfile(catalog.change_profiles[0]?.id || '')
    }
  }, [catalog.change_profiles, changeProfile])

  useEffect(() => {
    if (baselineRunID && !completedRuns.some((run) => run.id === baselineRunID)) {
      setBaselineRunID('')
    }
  }, [baselineRunID, completedRuns])

  useEffect(() => {
    if (!canAutoStart) setAutoStart(false)
  }, [canAutoStart])

  const toggleSuite = (suiteID: string) => {
    const suite = catalog.suites.find((candidate) => candidate.id === suiteID)
    if (!suite) return
    setSuiteIDs((current) => {
      if (current.includes(suiteID)) return current.filter((id) => id !== suiteID)
      return [...current, suiteID]
    })
    setTrackIDs((current) => [
      ...new Set([...current, ...suite.track_ids.filter((id) => availableTrackIDs.includes(id))]),
    ])
  }

  const toggleTrack = (trackID: EvaluationTrackId) => {
    setTrackIDs((current) =>
      current.includes(trackID) ? current.filter((id) => id !== trackID) : [...current, trackID],
    )
  }

  const submit = async (event: FormEvent) => {
    event.preventDefault()
    if (!canCreate) return
    if (!name.trim()) return setValidationError('Experiment name is required.')
    if (!changeProfile || !selectedChangeProfile) {
      return setValidationError('Select a change profile from the server catalog.')
    }
    if (!targetID) return setValidationError('Select an available catalog target.')
    if (suiteIDs.length === 0) return setValidationError('Select at least one benchmark suite.')
    if (trackIDs.length === 0) return setValidationError('Select at least one evaluation track.')
    if (sampleLimit < 1 || concurrency < 1) {
      return setValidationError('Sample limit and concurrency must be positive.')
    }
    setValidationError('')
    const created = await onSubmit({
      name,
      description,
      suite_ids: suiteIDs,
      track_ids: trackIDs,
      mode,
      target_id: targetID,
      change_profile: changeProfile,
      sample_limit: sampleLimit,
      concurrency,
      seed,
      ...(baselineRunID ? { baseline_run_id: baselineRunID } : {}),
      auto_start: autoStart,
    })
    if (created) {
      setName('')
      setDescription('')
    }
  }

  if (!canCreate) {
    return (
      <section className={styles.permissionState}>
        <span>Read-only evaluation access</span>
        <h2>Experiment creation is not available for this session.</h2>
        <p>You can still inspect completed evidence, reports, provenance, and comparisons.</p>
      </section>
    )
  }

  return (
    <form className={styles.form} onSubmit={submit}>
      <div className={styles.intro}>
        <div>
          <span className={styles.eyebrow}>Immutable run snapshot</span>
          <h2>New evaluation experiment</h2>
          <p>
            Suites and execution targets come from the server catalog. The browser cannot supply its
            own execution address.
          </p>
        </div>
        <div className={styles.introBadges}>
          <span className={styles.evidence}>{evidenceLevel || 'Evidence pending'}</span>
          <span className={styles.evidence}>{catalog.gate_contract_version}</span>
        </div>
      </div>

      {validationError ? (
        <div className={styles.error} role="alert">
          {validationError}
        </div>
      ) : null}

      <section className={styles.formSection}>
        <div className={styles.sectionHeading}>
          <span>01</span>
          <div>
            <h3>Identity and execution</h3>
            <p>Name the hypothesis and choose replay or live evidence.</p>
          </div>
        </div>
        <div className={styles.fieldGrid}>
          <label className={styles.fieldWide}>
            Experiment name
            <input
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="Recipe v3 vs production baseline"
              required
            />
          </label>
          <label className={styles.fieldWide}>
            Description
            <textarea
              value={description}
              onChange={(event) => setDescription(event.target.value)}
              placeholder="Hypothesis, expected trade-offs, and promotion decision."
              rows={3}
            />
          </label>
          <fieldset className={styles.choiceGroup}>
            <legend>Mode</legend>
            {(['replay', 'live'] as const).map((option) => (
              <label key={option} className={styles.choiceCard}>
                <input
                  type="radio"
                  name="evaluation-mode"
                  value={option}
                  checked={mode === option}
                  onChange={() => setMode(option)}
                />
                <span>
                  <strong>{option === 'replay' ? 'Replay' : 'Live'}</strong>
                  <small>
                    {option === 'replay'
                      ? 'Deterministic, reproducible evidence.'
                      : 'Execute against an approved runtime target.'}
                  </small>
                </span>
              </label>
            ))}
          </fieldset>
          <label>
            Catalog target
            <select value={targetID} onChange={(event) => setTargetID(event.target.value)} required>
              <option value="">Select target</option>
              {catalog.targets.map((target) => (
                <option
                  key={target.id}
                  value={target.id}
                  disabled={!target.modes.includes(mode) || target.healthy === false}
                >
                  {target.name}
                  {target.healthy === false ? ' · unavailable' : ''}
                </option>
              ))}
            </select>
            <small>
              {catalog.targets.find((target) => target.id === targetID)?.description ||
                'Only server-approved targets are selectable.'}
            </small>
          </label>
          <label>
            Baseline run
            <select
              value={baselineRunID}
              onChange={(event) => setBaselineRunID(event.target.value)}
            >
              <option value="">No baseline</option>
              {completedRuns.map((run) => (
                <option key={run.id} value={run.id}>
                  {run.name}
                </option>
              ))}
            </select>
          </label>
        </div>
      </section>

      <section className={styles.formSection}>
        <div className={styles.sectionHeading}>
          <span>02</span>
          <div>
            <h3>Change profile and G0–G9 contract</h3>
            <p>
              The profile defines which release gates are required, advisory, or not applicable.
            </p>
          </div>
        </div>
        <div className={styles.profileHeader}>
          <label>
            Change profile
            <select
              value={changeProfile}
              onChange={(event) =>
                setChangeProfile(event.target.value as EvaluationChangeProfileId)
              }
              required
            >
              <option value="">Select profile</option>
              {catalog.change_profiles.map((profile) => (
                <option key={profile.id} value={profile.id}>
                  {profile.name}
                </option>
              ))}
            </select>
            <small>
              {selectedChangeProfile?.description ||
                'Only server-declared change profiles are selectable.'}
            </small>
          </label>
          <div>
            <span>Gate contract</span>
            <code>{catalog.gate_contract_version}</code>
          </div>
        </div>
        {gateApplicability.length ? (
          <div className={styles.gateMatrix} aria-label="G0–G9 gate applicability">
            {gateApplicability.map((gate) => (
              <article key={gate.id} data-disposition={gate.disposition}>
                <div>
                  <code>{gate.id}</code>
                  <strong>{gate.name}</strong>
                </div>
                <span>{gate.disposition.replace('_', ' ')}</span>
                <small>{gate.description}</small>
              </article>
            ))}
          </div>
        ) : (
          <div className={styles.contractWarning} role="status">
            This dashboard cannot explain applicability for gate contract{' '}
            <code>{catalog.gate_contract_version}</code>. The server report remains authoritative.
          </div>
        )}
      </section>

      <section className={styles.formSection}>
        <div className={styles.sectionHeading}>
          <span>03</span>
          <div>
            <h3>Benchmark suites</h3>
            <p>Select versioned workloads, then refine the tracks executed by this run.</p>
          </div>
        </div>
        <div className={styles.catalogGrid}>
          {compatibleSuites.map((suite) => (
            <label
              key={suite.id}
              className={`${styles.catalogCard} ${suiteIDs.includes(suite.id) ? styles.selected : ''}`}
            >
              <input
                type="checkbox"
                checked={suiteIDs.includes(suite.id)}
                onChange={() => toggleSuite(suite.id)}
              />
              <span>
                <strong>{suite.name}</strong>
                <small>{suite.description}</small>
                <em>
                  {suite.evidence_level}
                  {suite.case_count ? ` · ${suite.case_count} cases` : ''}
                  {suite.revision ? ` · ${suite.revision}` : ''}
                </em>
              </span>
            </label>
          ))}
        </div>
      </section>

      <section className={styles.formSection}>
        <div className={styles.sectionHeading}>
          <span>04</span>
          <div>
            <h3>Evaluation tracks</h3>
            <p>Each track reports its own status, metrics, evidence, and gates.</p>
          </div>
        </div>
        <div className={styles.trackGrid}>
          {catalog.tracks.map((track) => {
            const available = availableTrackIDs.includes(track.id)
            return (
              <label
                key={track.id}
                className={`${styles.trackCard} ${trackIDs.includes(track.id) ? styles.selected : ''} ${!available ? styles.disabled : ''}`}
              >
                <input
                  type="checkbox"
                  checked={trackIDs.includes(track.id)}
                  disabled={!available}
                  onChange={() => toggleTrack(track.id)}
                />
                <span>
                  <strong>{TRACK_PRESENTATION[track.id].label}</strong>
                  <small>{track.description}</small>
                  <em>{track.metrics.length} metrics</em>
                </span>
              </label>
            )
          })}
        </div>
      </section>

      <section className={styles.formSection}>
        <div className={styles.sectionHeading}>
          <span>05</span>
          <div>
            <h3>Budget and reproducibility</h3>
            <p>Bound execution and pin the deterministic seed.</p>
          </div>
        </div>
        <div className={styles.numericGrid}>
          <label>
            Sample limit
            <input
              type="number"
              min={1}
              max={100000}
              value={sampleLimit}
              onChange={(event) => setSampleLimit(Number(event.target.value))}
            />
          </label>
          <label>
            Concurrency
            <input
              type="number"
              min={1}
              max={256}
              value={concurrency}
              onChange={(event) => setConcurrency(Number(event.target.value))}
            />
          </label>
          <label>
            Seed
            <input
              type="number"
              value={seed}
              onChange={(event) => setSeed(Number(event.target.value))}
            />
          </label>
        </div>
        <label className={styles.autoStart}>
          <input
            type="checkbox"
            checked={autoStart}
            disabled={!canAutoStart}
            onChange={(event) => setAutoStart(event.target.checked)}
          />
          <span>
            <strong>Start immediately</strong>
            <small>
              {canAutoStart
                ? 'Create the snapshot and enqueue execution.'
                : 'Requires evaluation.run permission.'}
            </small>
          </span>
        </label>
      </section>

      <div className={styles.actions}>
        <span>
          {suiteIDs.length} suites · {trackIDs.length} tracks · profile{' '}
          {changeProfile || 'not selected'} · target {targetID || 'not selected'}
        </span>
        <button type="submit" disabled={pending}>
          {pending ? 'Creating…' : autoStart ? 'Create and start' : 'Create draft'}
        </button>
      </div>
    </form>
  )
}
