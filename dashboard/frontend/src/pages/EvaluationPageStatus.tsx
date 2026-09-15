import type { EvaluationRunLedgerWarning } from '../types/evaluationPlane'
import { EvaluationTechnicalDisclosure } from '../components/evaluation-plane/EvaluationDisclosure'
import EvaluationIssueDetails, {
  type EvaluationIssueDetail,
} from '../components/evaluation-plane/EvaluationIssueDetails'
import { EvaluationActionButton } from '../components/evaluation-plane/EvaluationPrimitives'
import styles from './EvaluationPage.module.css'

interface EvaluationPageStatusProps {
  readonlyLoading: boolean
  serverReadonly: boolean
  hasCatalog: boolean
  catalogError: string | null
  runsError: string | null
  runsLoaded: boolean
  refreshing: boolean
  runLedgerComplete: boolean
  runLedgerWarningCount: number
  runLedgerWarnings: EvaluationRunLedgerWarning[]
  mutationError: string | null
  onRefresh: () => void
  onClearMutationError: () => void
}

function EvaluationRefreshStatus(props: EvaluationPageStatusProps) {
  if (!props.hasCatalog || (!props.catalogError && !props.runsError)) return null
  const refreshMessages = [
    props.catalogError
      ? 'Evaluation setup could not refresh. Showing the last loaded benchmark catalog.'
      : null,
    props.runsError
      ? props.runsLoaded
        ? 'Run history could not refresh. Showing the last loaded run state.'
        : 'Run history could not be loaded.'
      : null,
  ]
    .filter(Boolean)
    .join(' ')
  const refreshDetails: EvaluationIssueDetail[] = [
    ...(props.catalogError ? [{ label: 'Benchmark catalog', message: props.catalogError }] : []),
    ...(props.runsError ? [{ label: 'Run history', message: props.runsError }] : []),
  ]
  return (
    <div className={styles.staleBanner} role="status">
      <div className={styles.issueCopy}>
        <span>{refreshMessages}</span>
        <EvaluationIssueDetails issues={refreshDetails} />
      </div>
      <EvaluationActionButton
        type="button"
        compact
        disabled={props.refreshing}
        onClick={props.onRefresh}
      >
        {props.refreshing ? 'Retrying…' : 'Retry refresh'}
      </EvaluationActionButton>
    </div>
  )
}

function EvaluationLedgerStatus(props: EvaluationPageStatusProps) {
  if (!props.runsLoaded || props.runLedgerComplete || props.runLedgerWarningCount === 0) return null
  return (
    <div className={styles.ledgerBanner} role="alert">
      <div>
        <strong>Some saved runs could not be read</strong>
        <span>
          {props.runLedgerWarningCount} saved run
          {props.runLedgerWarningCount === 1 ? ' is' : 's are'} excluded. Available results remain
          safe to inspect, but baseline selection and comparison are paused until repaired.
        </span>
        {props.runLedgerWarnings.length < props.runLedgerWarningCount ? (
          <small>
            Showing {props.runLedgerWarnings.length} of {props.runLedgerWarningCount} warning
            details returned by run history.
          </small>
        ) : null}
      </div>
      {props.runLedgerWarnings.length ? (
        <EvaluationTechnicalDisclosure
          className={styles.warningDetails}
          summary={`Technical details · ${props.runLedgerWarnings.length}`}
          summaryClassName={styles.warningDetailsSummary}
        >
          <ul aria-label="Unreadable saved run details">
            {props.runLedgerWarnings.map((warning) => (
              <li key={`${warning.code}-${warning.evidence_id}-${warning.evidence_file}`}>
                <span className={styles.evidenceIdentity}>
                  <small>Run record</small>
                  <code>{warning.evidence_id}</code>
                </span>
                <span>
                  {warning.evidence_file}: {warning.message}
                </span>
              </li>
            ))}
          </ul>
        </EvaluationTechnicalDisclosure>
      ) : null}
    </div>
  )
}

function EvaluationMutationStatus(props: EvaluationPageStatusProps) {
  if (!props.mutationError) return null
  return (
    <div className={styles.errorBanner} role="alert">
      <div className={styles.issueCopy}>
        <span>
          The last evaluation action could not be completed. Review the technical details, then
          retry the action.
        </span>
        <EvaluationIssueDetails
          issues={[{ label: 'Evaluation action', message: props.mutationError }]}
        />
      </div>
      <EvaluationActionButton type="button" compact onClick={props.onClearMutationError}>
        Dismiss
      </EvaluationActionButton>
    </div>
  )
}

export default function EvaluationPageStatus(props: EvaluationPageStatusProps) {
  return (
    <>
      {!props.readonlyLoading && props.serverReadonly ? (
        <div className={styles.readonlyBanner} role="status">
          Results remain readable. The server is in read-only mode, so runs cannot be created,
          started, cancelled, or deleted.
        </div>
      ) : null}
      <EvaluationRefreshStatus {...props} />
      <EvaluationLedgerStatus {...props} />
      <EvaluationMutationStatus {...props} />
    </>
  )
}
