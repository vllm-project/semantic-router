import { EvaluationTechnicalDisclosure } from './EvaluationDisclosure'
import styles from './EvaluationIssueDetails.module.css'

export interface EvaluationIssueDetail {
  label: string
  message: string
}

interface EvaluationIssueDetailsProps {
  issues: EvaluationIssueDetail[]
  className?: string
}

export default function EvaluationIssueDetails({ issues, className }: EvaluationIssueDetailsProps) {
  if (issues.length === 0) return null

  return (
    <EvaluationTechnicalDisclosure
      className={[styles.details, className].filter(Boolean).join(' ')}
      summary="Technical details"
      summaryClassName={styles.summary}
    >
      <dl>
        {issues.map((issue, index) => (
          <div key={`${issue.label}-${index}`}>
            <dt>{issue.label}</dt>
            <dd>{issue.message}</dd>
          </div>
        ))}
      </dl>
    </EvaluationTechnicalDisclosure>
  )
}
