import type { EvaluationReport } from '../../types/evaluationReport'
import type { EvaluationReportDiagnosticsState } from '../../types/evaluationReportDiagnostics'
import EvaluationMetricTable from './EvaluationMetricTable'
import EvaluationMethodResults from './EvaluationMethodResults'
import EvaluationMixtureReport from './EvaluationMixtureReport'
import EvaluationReportDecision from './EvaluationReportDecision'
import EvaluationReportDiagnostics from './EvaluationReportDiagnostics'
import EvaluationReportDisclosures from './EvaluationReportDisclosures'
import EvaluationReportTracks from './EvaluationReportTracks'
import EvaluationRoutingRecipeReport from './EvaluationRoutingRecipeReport'
import { evaluationPromotionVerdict, evaluationResultScopeLabel } from './evaluationPresentation'
import { changeProfileLabel } from './evaluationRunPresentation'
import {
  CoverageBar,
  EvaluationTag,
  GateVerdictBadge,
  RunStatusBadge,
} from './EvaluationPrimitives'
import heroStyles from './EvaluationReportHero.module.css'
import styles from './EvaluationReportLayout.module.css'

export default function EvaluationReportView({
  report,
  diagnostics,
}: {
  report: EvaluationReport
  diagnostics: EvaluationReportDiagnosticsState
}) {
  const isDiagnostic = report.run.evidence_level === 'E0'

  return (
    <article className={styles.report} aria-labelledby="evaluation-report-title">
      <section className={heroStyles.reportHero}>
        <div className={heroStyles.reportHeroCopy}>
          <span className={styles.eyebrow}>Evaluation report</span>
          <h2 id="evaluation-report-title">{report.run.name}</h2>
          <p>{report.run.description || 'No experiment description was recorded.'}</p>
          <div className={heroStyles.heroBadges}>
            <RunStatusBadge status={report.run.status} />
            <GateVerdictBadge verdict={evaluationPromotionVerdict(report)} disposition="required" />
            <EvaluationTag tone="info">
              {evaluationResultScopeLabel(report.run.evidence_level)}
            </EvaluationTag>
            <EvaluationTag>{report.run.mode === 'live' ? 'Live run' : 'Replay'}</EvaluationTag>
            <EvaluationTag>{changeProfileLabel(report.run.change_profile)}</EvaluationTag>
          </div>
        </div>
        <div>
          <CoverageBar coverage={report.summary.coverage} />
          <p className={styles.scopeCopy}>Measured coverage for this run.</p>
        </div>
      </section>

      {isDiagnostic ? (
        <div className={heroStyles.claimNotice} role="status">
          <strong>Diagnostic run — no release recommendation</strong>
          <span>
            The results below validate the evaluation setup and help diagnose behavior. They do not
            establish the controlled or live outcomes needed for a release decision.
          </span>
        </div>
      ) : null}

      <EvaluationMixtureReport report={report} />
      <EvaluationReportDecision report={report} />
      {report.run.mixture && report.routing_recipe_report ? (
        <EvaluationRoutingRecipeReport
          plan={report.run.mixture.routing_recipe_plan}
          report={report.routing_recipe_report}
        />
      ) : null}
      <EvaluationMethodResults report={report} />

      <section className={styles.section} aria-labelledby="report-metrics-title">
        <div className={styles.sectionHeader}>
          <div>
            <span className={styles.eyebrow}>Measured outcomes</span>
            <h3 id="report-metrics-title">Metric explorer</h3>
            <p>
              Verified headline results are identified explicitly. Supporting diagnostics remain
              available for investigation without being treated as release evidence.
            </p>
          </div>
          <span>{report.metrics.length} aggregates</span>
        </div>
        <EvaluationMetricTable metrics={report.metrics} evidenceLevel={report.run.evidence_level} />
      </section>

      <section className={styles.section} aria-labelledby="report-diagnostics-title">
        <div className={styles.sectionHeader}>
          <div>
            <span className={styles.eyebrow}>Verified artifacts</span>
            <h3 id="report-diagnostics-title">Execution diagnostics</h3>
            <p>Outcome accounting and capacity observations load independently.</p>
          </div>
        </div>
        <EvaluationReportDiagnostics metrics={report.metrics} {...diagnostics} />
      </section>

      <EvaluationReportTracks report={report} />
      <EvaluationReportDisclosures report={report} />
    </article>
  )
}
