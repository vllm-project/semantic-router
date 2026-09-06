import type { EvaluationReport, EvaluationTrackReport } from '../../types/evaluationReport'
import { TRACK_PRESENTATION } from './evaluationTrackPresentation'
import { evaluationResultScopeLabel } from './evaluationPresentation'
import EvaluationIssueDetails from './EvaluationIssueDetails'
import { RunStatusBadge } from './EvaluationPrimitives'
import styles from './EvaluationReportLayout.module.css'
import tableStyles from './EvaluationReportTable.module.css'

function trackProductSummary(track: EvaluationTrackReport): string {
  switch (track.status) {
    case 'completed':
      return track.coverage.unavailable
        ? 'Verified results are available, with some cases not measured.'
        : 'Verified results are available for this area.'
    case 'failed':
      return 'This area stopped before a final result was published.'
    case 'cancelled':
      return 'This area stopped before completing.'
    case 'skipped':
      return 'This area was not executed for this run.'
    case 'unavailable':
      return 'This area did not produce verified results.'
    case 'pending':
    case 'running':
    case 'sealing':
      return 'This area did not publish a final result.'
  }
}

export default function EvaluationReportTracks({ report }: { report: EvaluationReport }) {
  return (
    <section className={styles.section} aria-labelledby="report-tracks-title">
      <div className={styles.sectionHeader}>
        <div>
          <span className={styles.eyebrow}>Evaluation coverage</span>
          <h3 id="report-tracks-title">Results by evaluation area</h3>
          <p>Status, coverage, and validation depth for every selected area.</p>
        </div>
        <span>{report.tracks.length} selected areas</span>
      </div>
      <div
        className={tableStyles.tableScroll}
        role="region"
        tabIndex={0}
        aria-label="Scrollable results by evaluation area"
      >
        <table className={tableStyles.table}>
          <caption>Result status and coverage by selected evaluation area</caption>
          <thead>
            <tr>
              <th scope="col">Evaluation area</th>
              <th scope="col">Status</th>
              <th scope="col">Coverage</th>
              <th scope="col">Validation depth</th>
              <th scope="col">Summary</th>
            </tr>
          </thead>
          <tbody>
            {report.tracks.map((track) => (
              <tr key={track.track_id}>
                <th scope="row">{TRACK_PRESENTATION[track.track_id].label}</th>
                <td>
                  <RunStatusBadge status={track.status} />
                </td>
                <td>
                  {track.coverage.evaluated}/{track.coverage.total} ·{' '}
                  {Math.round(track.coverage.fraction * 100)}%
                  {track.coverage.unavailable
                    ? ` · ${track.coverage.unavailable} not measured`
                    : ''}
                </td>
                <td>{evaluationResultScopeLabel(track.evidence_level)}</td>
                <td>
                  <div className={styles.trackSummary}>
                    <span>{trackProductSummary(track)}</span>
                    <EvaluationIssueDetails
                      issues={[
                        ...(track.summary
                          ? [{ label: 'Recorded summary', message: track.summary }]
                          : []),
                        ...(track.error ? [{ label: 'Recorded error', message: track.error }] : []),
                      ]}
                    />
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}
