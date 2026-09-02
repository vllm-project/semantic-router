import type { EvaluationView } from '../../pages/evaluationRoute'
import type { EvaluationCatalog, EvaluationRun } from '../../types/evaluationPlane'
import type { EvaluationReport } from '../../types/evaluationReport'
import EvaluationLatestEvidence from './EvaluationLatestEvidence'
import { buildEvaluationOverviewModel } from './evaluationOverview'
import EvaluationOverviewReadiness from './EvaluationOverviewReadiness'
import EvaluationTrackReadiness from './EvaluationTrackReadiness'
import styles from './EvaluationPlane.module.css'

interface EvaluationOverviewProps {
  catalog: EvaluationCatalog
  runs: EvaluationRun[]
  totalRuns: number
  hasMoreRuns: boolean
  loadingMoreRuns: boolean
  runLedgerAvailable: boolean
  runLedgerComplete: boolean
  latestReport: EvaluationReport | null
  requestedReportRunID: string | null
  reportLoading: boolean
  reportError: string | null
  onRetryReport: () => void
  onLoadMoreRuns: () => void
  onNavigate: (view: EvaluationView) => void
  onOpenReport: (runID: string) => void
}

export default function EvaluationOverview({
  catalog,
  runs,
  totalRuns,
  hasMoreRuns,
  loadingMoreRuns,
  runLedgerAvailable,
  runLedgerComplete,
  latestReport,
  requestedReportRunID,
  reportLoading,
  reportError,
  onRetryReport,
  onLoadMoreRuns,
  onNavigate,
  onOpenReport,
}: EvaluationOverviewProps) {
  const model = buildEvaluationOverviewModel({ runs, latestReport, requestedReportRunID })

  return (
    <div className={styles.sectionStack}>
      <EvaluationOverviewReadiness
        model={model}
        runs={runs}
        totalRuns={totalRuns}
        hasMoreRuns={hasMoreRuns}
        runLedgerAvailable={runLedgerAvailable}
        runLedgerComplete={runLedgerComplete}
        reportLoading={reportLoading}
        onNavigate={onNavigate}
      />
      <EvaluationLatestEvidence
        model={model}
        latestReport={latestReport}
        reportLoading={reportLoading}
        reportError={reportError}
        runLedgerAvailable={runLedgerAvailable}
        runLedgerComplete={runLedgerComplete}
        hasMoreRuns={hasMoreRuns}
        loadingMoreRuns={loadingMoreRuns}
        onRetryReport={onRetryReport}
        onLoadMoreRuns={onLoadMoreRuns}
        onOpenReport={onOpenReport}
      />
      <EvaluationTrackReadiness catalog={catalog} latestReport={latestReport} />
    </div>
  )
}
