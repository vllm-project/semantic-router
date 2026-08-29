import type { EvaluationComparison, EvaluationRun } from '../../types/evaluationPlane'
import EvaluationGateList from './EvaluationGateList'
import { effectiveGateVerdict, formatDelta, formatMetric } from './evaluationPresentation'
import { GateVerdictBadge } from './EvaluationPrimitives'
import styles from './EvaluationReport.module.css'

interface EvaluationCompareProps {
  runs: EvaluationRun[]
  baselineID: string
  candidateID: string
  comparison: EvaluationComparison | null
  loading: boolean
  error: string | null
  onBaselineChange: (id: string) => void
  onCandidateChange: (id: string) => void
  onCompare: () => void
}

export default function EvaluationCompare({
  runs,
  baselineID,
  candidateID,
  comparison,
  loading,
  error,
  onBaselineChange,
  onCandidateChange,
  onCompare,
}: EvaluationCompareProps) {
  const completedRuns = runs.filter((run) => run.status === 'completed')
  const baselineRun = completedRuns.find((run) => run.id === baselineID)
  const candidateRun = completedRuns.find((run) => run.id === candidateID)
  const profileMismatch = Boolean(
    baselineRun && candidateRun && baselineRun.change_profile !== candidateRun.change_profile,
  )
  const invalidPair = !baselineID || !candidateID || baselineID === candidateID || profileMismatch
  const comparisonVerdict = comparison
    ? effectiveGateVerdict(comparison.verdict, comparison.gates)
    : null
  return (
    <div className={styles.report}>
      <section className={styles.compareHero}>
        <div>
          <span className={styles.eyebrow}>Paired evidence</span>
          <h2>Compare candidate against baseline</h2>
          <p>Review metric deltas and promotion gates on two completed, reproducible runs.</p>
        </div>
        <div className={styles.compareControls}>
          <label>
            Baseline
            <select value={baselineID} onChange={(event) => onBaselineChange(event.target.value)}>
              <option value="">Select baseline</option>
              {completedRuns.map((run) => (
                <option
                  key={run.id}
                  value={run.id}
                  disabled={
                    run.id === candidateID ||
                    Boolean(candidateRun && run.change_profile !== candidateRun.change_profile)
                  }
                >
                  {run.name} · {run.change_profile}
                </option>
              ))}
            </select>
          </label>
          <label>
            Candidate
            <select value={candidateID} onChange={(event) => onCandidateChange(event.target.value)}>
              <option value="">Select candidate</option>
              {completedRuns.map((run) => (
                <option
                  key={run.id}
                  value={run.id}
                  disabled={
                    run.id === baselineID ||
                    Boolean(baselineRun && run.change_profile !== baselineRun.change_profile)
                  }
                >
                  {run.name} · {run.change_profile}
                </option>
              ))}
            </select>
          </label>
          <button type="button" disabled={invalidPair || loading} onClick={onCompare}>
            {loading ? 'Comparing…' : 'Compare runs'}
          </button>
        </div>
      </section>

      {error ? (
        <div className={styles.error} role="alert">
          {error}
        </div>
      ) : null}
      {profileMismatch ? (
        <div className={styles.error} role="alert">
          Baseline and candidate must use the same change profile.
        </div>
      ) : null}
      {!comparison && !error ? (
        <div className={styles.empty}>
          Choose two different completed runs, then calculate the comparison.
        </div>
      ) : null}
      {comparison ? (
        <>
          <section className={styles.section}>
            <div className={styles.sectionHeader}>
              <div>
                <span className={styles.eyebrow}>Comparison verdict</span>
                <h3>{comparison.summary}</h3>
              </div>
              {comparisonVerdict ? <GateVerdictBadge verdict={comparisonVerdict} /> : null}
            </div>
            <div className={styles.deltaGrid}>
              {comparison.metrics.map((metric) => (
                <article key={metric.id}>
                  <span>{metric.name}</span>
                  <strong>{formatMetric(metric)}</strong>
                  <small
                    className={
                      (metric.delta || 0) < 0 ? styles.negativeDelta : styles.positiveDelta
                    }
                  >
                    {formatDelta(metric) || 'No delta'} vs baseline
                  </small>
                </article>
              ))}
            </div>
          </section>
          <section className={styles.section}>
            <div className={styles.sectionHeader}>
              <div>
                <span className={styles.eyebrow}>Regression boundary</span>
                <h3>Comparison gates</h3>
              </div>
            </div>
            <EvaluationGateList gates={comparison.gates} />
          </section>
          <section className={styles.section}>
            <div className={styles.sectionHeader}>
              <div>
                <span className={styles.eyebrow}>Next actions</span>
                <h3>Recommendations</h3>
              </div>
            </div>
            {comparison.recommendations.length ? (
              <ol className={styles.recommendations}>
                {comparison.recommendations.map((item, index) => (
                  <li key={`${index}-${item}`}>{item}</li>
                ))}
              </ol>
            ) : (
              <p className={styles.empty}>No comparison recommendations were generated.</p>
            )}
          </section>
        </>
      ) : null}
    </div>
  )
}
