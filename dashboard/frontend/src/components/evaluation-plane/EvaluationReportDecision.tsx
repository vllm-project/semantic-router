import type { EvaluationReport } from '../../types/evaluationReport'
import EvaluationGateList from './EvaluationGateList'
import { evaluationMetricLabel } from './evaluationMetricPresentation'
import { TRACK_PRESENTATION } from './evaluationTrackPresentation'
import {
  evaluationPromotionVerdict,
  evaluationResultScopeLabel,
  formatMetric,
  selectHeadlineMetrics,
} from './evaluationPresentation'
import { GateVerdictBadge } from './EvaluationPrimitives'
import styles from './EvaluationReportDecision.module.css'
import layoutStyles from './EvaluationReportLayout.module.css'

export default function EvaluationReportDecision({ report }: { report: EvaluationReport }) {
  const isDiagnostic = report.run.evidence_level === 'E0'
  const requiredGates = report.gates.filter((gate) => gate.disposition === 'required')
  const requiredPassed = requiredGates.filter((gate) => gate.verdict === 'pass').length
  const requiredFailed = requiredGates.filter((gate) => gate.verdict === 'fail').length
  const requiredUnavailable = requiredGates.filter((gate) => gate.verdict === 'unavailable').length
  const requiredBlockers = requiredGates.filter(
    (gate) => gate.verdict === 'fail' || gate.verdict === 'unavailable',
  )
  const promotionVerdict = evaluationPromotionVerdict(report)
  const headlines = selectHeadlineMetrics(report)

  return (
    <>
      <section className={layoutStyles.section} aria-labelledby="report-decision-title">
        <div className={layoutStyles.sectionHeader}>
          <div>
            <span className={layoutStyles.eyebrow}>Release readiness</span>
            <h3 id="report-decision-title">
              {isDiagnostic
                ? 'Diagnostic result only'
                : requiredBlockers.length
                  ? 'Not ready to release'
                  : 'Ready for release'}
            </h3>
            <p>
              {`${requiredPassed}/${requiredGates.length} required checks passed · ${requiredFailed} blocked · ${requiredUnavailable} incomplete`}
              {isDiagnostic ? ' · diagnostic runs do not support a release decision' : ''}
            </p>
          </div>
          <GateVerdictBadge verdict={promotionVerdict} disposition="required" />
        </div>
        {headlines.length ? (
          <dl className={styles.headlineStrip}>
            {headlines.map((metric) => (
              <div key={`${metric.track_id || 'system'}-${metric.id}`}>
                <dt>{evaluationMetricLabel(metric)}</dt>
                <dd>{formatMetric(metric)}</dd>
                <span>
                  {evaluationResultScopeLabel(report.run.evidence_level)} ·{' '}
                  {metric.track_id ? TRACK_PRESENTATION[metric.track_id].label : 'System'}
                </span>
              </div>
            ))}
          </dl>
        ) : (
          <p className={layoutStyles.empty}>
            {isDiagnostic
              ? 'No verified headline applies to this diagnostic run. Open the metric explorer for all supporting observations.'
              : 'No measured headline aggregate applies to this run scope.'}
          </p>
        )}
      </section>

      {requiredBlockers.length ? (
        <section className={layoutStyles.section} aria-labelledby="report-blockers-title">
          <div className={layoutStyles.sectionHeader}>
            <div>
              <span className={layoutStyles.eyebrow}>What needs attention</span>
              <h3 id="report-blockers-title">Incomplete release checks</h3>
              <p>Each check explains the result and what is needed before it can pass.</p>
            </div>
            <span>{requiredBlockers.length} checks</span>
          </div>
          <EvaluationGateList gates={requiredBlockers} />
        </section>
      ) : null}
    </>
  )
}
