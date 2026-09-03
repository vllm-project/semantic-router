import type { EvaluationCatalog } from '../../types/evaluationPlane'
import type { EvaluationReport } from '../../types/evaluationReport'
import { EVALUATION_TRACK_IDS } from '../../types/evaluationPlane'
import { evaluationResultScopeLabel } from './evaluationPresentation'
import EvaluationMethodReadiness from './EvaluationMethodReadiness'
import { RunStatusBadge } from './EvaluationPrimitives'
import { TRACK_PRESENTATION } from './evaluationTrackPresentation'
import styles from './EvaluationPlane.module.css'
import tableStyles from './EvaluationReportTable.module.css'

interface EvaluationTrackReadinessProps {
  catalog: EvaluationCatalog
  latestReport: EvaluationReport | null
}

export default function EvaluationTrackReadiness({
  catalog,
  latestReport,
}: EvaluationTrackReadinessProps) {
  return (
    <>
      <section className={styles.surface} aria-labelledby="track-readiness-title">
        <header className={styles.surfaceHeader}>
          <div>
            <span className={styles.eyebrow}>Evaluation coverage</span>
            <h2 id="track-readiness-title">What each evaluation area can measure</h2>
            <p>
              Available evaluation depth and the latest measured result are shown separately. The
              final scope depends on how the experiment runs and which data completes.
            </p>
          </div>
          <div className={styles.catalogFacts} aria-label="Evaluation catalog summary">
            <span>{catalog.tracks.length} evaluation areas</span>
            <span>{catalog.suites.length} benchmarks</span>
          </div>
        </header>
        <div
          className={`${tableStyles.tableScroll} ${styles.catalogTableFrame}`}
          role="region"
          tabIndex={0}
          aria-label="Scrollable evaluation area readiness"
        >
          <table className={`${tableStyles.table} ${tableStyles.tableReadiness}`}>
            <caption>Available measurements and latest results by evaluation area</caption>
            <thead>
              <tr>
                <th scope="col">Evaluation area</th>
                <th scope="col">Metrics</th>
                <th scope="col">Latest result</th>
                <th scope="col">Available validation depth</th>
              </tr>
            </thead>
            <tbody>
              {EVALUATION_TRACK_IDS.map((trackID) => {
                const contract = catalog.tracks.find((track) => track.id === trackID)!
                const observation = latestReport?.tracks.find((track) => track.track_id === trackID)
                return (
                  <tr key={trackID}>
                    <th scope="row">
                      <strong>{TRACK_PRESENTATION[trackID].label}</strong>
                      <span>{contract.description}</span>
                    </th>
                    <td>{contract.metrics.length} metrics</td>
                    <td>
                      {observation ? (
                        <span className={tableStyles.inlineStatus}>
                          <RunStatusBadge status={observation.status} />
                          {evaluationResultScopeLabel(observation.evidence_level)} ·{' '}
                          {observation.coverage.evaluated}/{observation.coverage.total} cases ·{' '}
                          {Math.round(observation.coverage.fraction * 100)}%
                          {observation.coverage.unavailable
                            ? ` · ${observation.coverage.unavailable} not measured`
                            : ''}
                        </span>
                      ) : (
                        'Not included in the latest run · select this area in a new experiment'
                      )}
                    </td>
                    <td>{contract.evidence_levels.map(evaluationResultScopeLabel).join(' · ')}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </section>

      <EvaluationMethodReadiness catalog={catalog} />
    </>
  )
}
