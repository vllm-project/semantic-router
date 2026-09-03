import ProductLoadingState from '../ProductLoadingState'
import type { EvaluationRun } from '../../types/evaluationPlane'
import type { EvaluationReport } from '../../types/evaluationReport'
import EvaluationIssueDetails from './EvaluationIssueDetails'
import { EvaluationActionButton } from './EvaluationPrimitives'
import EvaluationReportContainer from './EvaluationReportContainer'
import { runOptionLabels } from './evaluationRunPresentation'
import styles from './EvaluationPlane.module.css'
import reportStyles from './EvaluationReports.module.css'

interface EvaluationReportsProps {
  runs: EvaluationRun[]
  selectedRunID: string
  report: EvaluationReport | null
  loading: boolean
  runLedgerAvailable: boolean
  totalRuns: number
  hasMoreRuns: boolean
  loadingMoreRuns: boolean
  error: string | null
  onSelect: (runID: string) => void
  onRetry: () => void
  onLoadMoreRuns: () => void
}

interface ReportSelectorProps {
  reportableRuns: EvaluationRun[]
  selectedReportRun: EvaluationRun | null
  selectedRunID: string
  reportLabels: Map<string, string>
  runLedgerAvailable: boolean
  onSelect: (runID: string) => void
}

interface ReportLibraryProps extends ReportSelectorProps {
  totalRuns: number
  hasMoreRuns: boolean
  loadingMoreRuns: boolean
  onLoadMoreRuns: () => void
}

function ReportSelector({
  reportableRuns,
  selectedReportRun,
  selectedRunID,
  reportLabels,
  runLedgerAvailable,
  onSelect,
}: ReportSelectorProps) {
  return (
    <label className={reportStyles.reportSelector}>
      <span>Run</span>
      <select
        value={selectedRunID}
        disabled={!runLedgerAvailable}
        onChange={(event) => onSelect(event.target.value)}
      >
        <option value="">
          {runLedgerAvailable ? 'Select a completed run' : 'Run history unavailable'}
        </option>
        {selectedReportRun ? (
          <option value={selectedReportRun.id}>{reportLabels.get(selectedReportRun.id)}</option>
        ) : null}
        {reportableRuns.map((run) => (
          <option key={run.id} value={run.id}>
            {reportLabels.get(run.id)}
          </option>
        ))}
      </select>
    </label>
  )
}

function ReportLibrary({
  reportableRuns,
  selectedReportRun,
  selectedRunID,
  reportLabels,
  runLedgerAvailable,
  loadedRuns,
  totalRuns,
  hasMoreRuns,
  loadingMoreRuns,
  onSelect,
  onLoadMoreRuns,
}: ReportLibraryProps & { loadedRuns: number }) {
  return (
    <section className={styles.surface}>
      <div className={styles.surfaceHeader}>
        <div>
          <span className={styles.eyebrow}>Report library</span>
          <h2>Reports</h2>
          <p>
            Review measured outcomes, release blockers, cost, diagnostics, and the exact
            configuration behind each result.
          </p>
        </div>
        <ReportSelector
          reportableRuns={reportableRuns}
          selectedReportRun={selectedReportRun}
          selectedRunID={selectedRunID}
          reportLabels={reportLabels}
          runLedgerAvailable={runLedgerAvailable}
          onSelect={onSelect}
        />
      </div>
      {runLedgerAvailable && hasMoreRuns ? (
        <div className={styles.scopeNotice} role="status">
          <span>
            Report selection covers {loadedRuns} of {totalRuns} loaded runs.
          </span>
          <EvaluationActionButton
            type="button"
            compact
            disabled={loadingMoreRuns}
            onClick={onLoadMoreRuns}
          >
            {loadingMoreRuns ? 'Loading older runs…' : 'Load older reports'}
          </EvaluationActionButton>
        </div>
      ) : null}
    </section>
  )
}

function ReportResult({
  loading,
  error,
  report,
  runLedgerAvailable,
  reportableRuns,
  hasMoreRuns,
  onRetry,
}: Pick<
  EvaluationReportsProps,
  'loading' | 'error' | 'report' | 'runLedgerAvailable' | 'hasMoreRuns' | 'onRetry'
> & { reportableRuns: EvaluationRun[] }) {
  return (
    <>
      {loading ? (
        <div className={reportStyles.reportLoading}>
          <ProductLoadingState label="Loading evaluation report" compact />
        </div>
      ) : null}
      {error ? (
        <div className={styles.errorState} role="alert">
          <div>
            <h2>Report could not be loaded</h2>
            <p>Retry to load the selected completed run and its saved evidence.</p>
            <EvaluationIssueDetails issues={[{ label: 'Report request', message: error }]} />
          </div>
          <EvaluationActionButton type="button" onClick={onRetry}>
            Retry
          </EvaluationActionButton>
        </div>
      ) : null}
      {!loading && !error && report ? <EvaluationReportContainer report={report} /> : null}
      {!loading && !error && !report ? (
        <div className={styles.emptyState}>
          <p>
            {!runLedgerAvailable
              ? 'Retry run history to discover completed reports.'
              : reportableRuns.length
                ? 'Select a completed run to load its full report.'
                : hasMoreRuns
                  ? 'No completed report is present in the loaded runs. Load older runs to continue searching.'
                  : 'No completed run has published a report yet. Failed and cancelled runs remain in the run inspector.'}
          </p>
        </div>
      ) : null}
    </>
  )
}

export default function EvaluationReports(props: EvaluationReportsProps) {
  const reportableRuns = props.runs.filter((run) => run.status === 'completed')
  const selectedReportRun =
    props.report && !reportableRuns.some((run) => run.id === props.report?.run.id)
      ? props.report.run
      : null
  const reportLabels = runOptionLabels([
    ...(selectedReportRun ? [selectedReportRun] : []),
    ...reportableRuns,
  ])
  return (
    <div className={styles.sectionStack} aria-busy={props.loading}>
      <ReportLibrary
        reportableRuns={reportableRuns}
        selectedReportRun={selectedReportRun}
        selectedRunID={props.selectedRunID}
        reportLabels={reportLabels}
        runLedgerAvailable={props.runLedgerAvailable}
        loadedRuns={props.runs.length}
        totalRuns={props.totalRuns}
        hasMoreRuns={props.hasMoreRuns}
        loadingMoreRuns={props.loadingMoreRuns}
        onSelect={props.onSelect}
        onLoadMoreRuns={props.onLoadMoreRuns}
      />
      <ReportResult
        loading={props.loading}
        error={props.error}
        report={props.report}
        runLedgerAvailable={props.runLedgerAvailable}
        reportableRuns={reportableRuns}
        hasMoreRuns={props.hasMoreRuns}
        onRetry={props.onRetry}
      />
    </div>
  )
}
