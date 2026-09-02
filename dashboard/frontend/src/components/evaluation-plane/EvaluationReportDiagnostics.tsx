import type { EvaluationCapacityProfile } from '../../types/evaluationCapacityReport'
import type { EvaluationFailureSummary, EvaluationMetric } from '../../types/evaluationReport'
import type { EvaluationDiagnosticArtifactIssue } from '../../types/evaluationReportDiagnostics'
import EvaluationReportAnalysisDetails from './EvaluationReportAnalysisDetails'
import EvaluationReportCapacity from './EvaluationReportCapacity'
import EvaluationReportOutcomeAccounting from './EvaluationReportOutcomeAccounting'
import { uniqueAnalysisPlans } from './evaluationReportAnalysis'
import styles from './EvaluationReportDiagnostics.module.css'
import reportStyles from './EvaluationReportLayout.module.css'

interface EvaluationReportDiagnosticsProps {
  metrics: EvaluationMetric[]
  failureSummary: EvaluationFailureSummary | null
  capacityProfile: EvaluationCapacityProfile | null
  failureSummaryIssue: EvaluationDiagnosticArtifactIssue | null
  capacityProfileIssue: EvaluationDiagnosticArtifactIssue | null
  loading: boolean
}

const VERIFICATION_COPY = 'Diagnostic artifacts verified by the evaluation service.'

function DiagnosticsStatus({ loading }: { loading: boolean }) {
  return (
    <div className={styles.diagnosticsStack}>
      <p className={reportStyles.scopeCopy}>{VERIFICATION_COPY}</p>
      <p className={reportStyles.empty}>
        {loading
          ? 'Loading diagnostic artifacts…'
          : 'This run did not publish aggregate diagnostics.'}
      </p>
    </div>
  )
}

export default function EvaluationReportDiagnostics({
  metrics,
  failureSummary,
  capacityProfile,
  failureSummaryIssue,
  capacityProfileIssue,
  loading,
}: EvaluationReportDiagnosticsProps) {
  const analysisPlans = uniqueAnalysisPlans(metrics)
  if (loading) return <DiagnosticsStatus loading />
  const hasDiagnostics = Boolean(
    failureSummary ||
      capacityProfile ||
      failureSummaryIssue ||
      capacityProfileIssue ||
      analysisPlans.length,
  )
  if (!hasDiagnostics) return <DiagnosticsStatus loading={false} />
  return (
    <div className={styles.diagnosticsStack}>
      <p className={reportStyles.scopeCopy}>{VERIFICATION_COPY}</p>
      <EvaluationReportAnalysisDetails metricCount={metrics.length} plans={analysisPlans} />
      <EvaluationReportOutcomeAccounting
        failureSummary={failureSummary}
        issue={failureSummaryIssue}
      />
      <EvaluationReportCapacity capacityProfile={capacityProfile} issue={capacityProfileIssue} />
    </div>
  )
}
