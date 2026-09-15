import type { EvaluationDiagnosticArtifactIssue } from '../../types/evaluationReportDiagnostics'
import EvaluationIssueDetails from './EvaluationIssueDetails'
import heroStyles from './EvaluationReportHero.module.css'

interface EvaluationReportDiagnosticIssueProps {
  label: string
  issue: EvaluationDiagnosticArtifactIssue
}

export default function EvaluationReportDiagnosticIssue({
  label,
  issue,
}: EvaluationReportDiagnosticIssueProps) {
  const invalid = issue.kind === 'invalid'
  return (
    <div className={heroStyles.inlineNotice} role="alert" aria-label={`${label} diagnostic error`}>
      <strong>
        {invalid ? 'Diagnostic could not be verified' : 'Diagnostic is not available'}
      </strong>
      <span>
        {invalid
          ? 'This diagnostic is excluded because its saved evidence could not be verified. Other verified results remain available.'
          : 'This diagnostic was not published or could not be retrieved. Other verified results remain available.'}
      </span>
      <EvaluationIssueDetails
        issues={[
          { label: 'Artifact', message: issue.artifactName },
          { label: 'Service response', message: issue.message },
        ]}
      />
    </div>
  )
}
