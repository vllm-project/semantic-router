import ProductLoadingState from '../ProductLoadingState'
import type { EvaluationReport, EvaluationRun } from '../../types/evaluationPlane'
import EvaluationReportView from './EvaluationReportView'
import styles from './EvaluationPlane.module.css'

interface EvaluationReportsProps {
  runs: EvaluationRun[]
  selectedRunID: string
  report: EvaluationReport | null
  loading: boolean
  error: string | null
  onSelect: (runID: string) => void
  onRetry: () => void
}

export default function EvaluationReports({
  runs,
  selectedRunID,
  report,
  loading,
  error,
  onSelect,
  onRetry,
}: EvaluationReportsProps) {
  const reportableRuns = runs.filter((run) =>
    ['completed', 'failed', 'cancelled'].includes(run.status),
  )
  return (
    <div className={styles.sectionStack}>
      <section className={styles.panel}>
        <div className={styles.panelHeader}>
          <div>
            <span className={styles.eyebrow}>Evidence browser</span>
            <h2>Reports</h2>
            <p>
              Inspect gate verdicts, track coverage, costs, recommendations, and immutable
              provenance.
            </p>
          </div>
          <label className={styles.reportSelector}>
            <span>Run</span>
            <select value={selectedRunID} onChange={(event) => onSelect(event.target.value)}>
              <option value="">Select a terminal run</option>
              {reportableRuns.map((run) => (
                <option key={run.id} value={run.id}>
                  {run.name} · {run.status}
                </option>
              ))}
            </select>
          </label>
        </div>
      </section>
      {loading ? (
        <div className={styles.panel}>
          <ProductLoadingState label="Loading evaluation report" compact />
        </div>
      ) : null}
      {error ? (
        <div className={styles.errorState} role="alert">
          <h2>Report unavailable</h2>
          <p>{error}</p>
          <button type="button" onClick={onRetry}>
            Retry
          </button>
        </div>
      ) : null}
      {!loading && !error && report ? <EvaluationReportView report={report} /> : null}
      {!loading && !error && !report ? (
        <div className={styles.emptyState}>
          <p>Select a terminal run to load its report.</p>
        </div>
      ) : null}
    </div>
  )
}
