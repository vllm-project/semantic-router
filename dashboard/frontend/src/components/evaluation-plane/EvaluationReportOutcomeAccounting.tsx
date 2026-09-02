import type { EvaluationFailureSummary } from '../../types/evaluationReport'
import type { EvaluationDiagnosticArtifactIssue } from '../../types/evaluationReportDiagnostics'
import EvaluationReportDiagnosticIssue from './EvaluationReportDiagnosticIssue'
import styles from './EvaluationReportDiagnostics.module.css'
import reportStyles from './EvaluationReportLayout.module.css'
import { TRACK_PRESENTATION } from './evaluationTrackPresentation'
import tableStyles from './EvaluationReportTable.module.css'

interface EvaluationReportOutcomeAccountingProps {
  failureSummary: EvaluationFailureSummary | null
  issue: EvaluationDiagnosticArtifactIssue | null
}

function OutcomeSummary({ summary }: { summary: EvaluationFailureSummary }) {
  const succeeded = summary.total_records - summary.failed - summary.unavailable
  return (
    <div className={styles.diagnosticSummary}>
      <div>
        <span>Total records</span>
        <strong>{summary.total_records}</strong>
      </div>
      <div>
        <span>Succeeded</span>
        <strong>{succeeded}</strong>
      </div>
      <div>
        <span>Failed</span>
        <strong>{summary.failed}</strong>
      </div>
      <div>
        <span>Not measured</span>
        <strong>{summary.unavailable}</strong>
      </div>
    </div>
  )
}

function OutcomeTable({ summary }: { summary: EvaluationFailureSummary }) {
  return (
    <div
      className={tableStyles.tableScroll}
      role="region"
      tabIndex={0}
      aria-label="Scrollable outcome accounting by evaluation area"
    >
      <table className={tableStyles.table}>
        <caption>Outcome accounting by evaluation area</caption>
        <thead>
          <tr>
            <th scope="col">Evaluation area</th>
            <th scope="col">Succeeded</th>
            <th scope="col">Failed</th>
            <th scope="col">Not measured</th>
          </tr>
        </thead>
        <tbody>
          {summary.by_track.map((row) => (
            <tr key={row.track_id}>
              <th scope="row">{TRACK_PRESENTATION[row.track_id].label}</th>
              <td>{row.succeeded}</td>
              <td>{row.failed}</td>
              <td>{row.unavailable}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function EvaluationReportOutcomeAccounting({
  failureSummary,
  issue,
}: EvaluationReportOutcomeAccountingProps) {
  if (!failureSummary && !issue) return null
  return (
    <section className={styles.diagnosticArtifact} aria-labelledby="diagnostic-outcome-title">
      <div className={reportStyles.subsectionHeader}>
        <div>
          <h4 id="diagnostic-outcome-title">Outcome accounting</h4>
          <p>Verified completion totals retained without case-level content.</p>
        </div>
      </div>
      {issue ? (
        <EvaluationReportDiagnosticIssue label="Outcome accounting" issue={issue} />
      ) : failureSummary ? (
        <>
          <OutcomeSummary summary={failureSummary} />
          <OutcomeTable summary={failureSummary} />
        </>
      ) : null}
    </section>
  )
}
