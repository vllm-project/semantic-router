import type { EvaluationRun, EvaluationRunEvent } from '../../types/evaluationPlane'
import type { EvaluationControlledPairExecution } from '../../types/evaluationControlledPair'
import { formatDateTime } from '../../utils/dateTime'
import { EvaluationActionButton } from './EvaluationPrimitives'
import EvaluationRunInspector from './EvaluationRunInspector'
import EvaluationRunLedger, { EvaluationRunLedgerFilters } from './EvaluationRunLedger'
import useEvaluationRunLedger from './useEvaluationRunLedger'
import styles from './EvaluationPlane.module.css'
import runStyles from './EvaluationRuns.module.css'

interface EvaluationRunsProps {
  runs: EvaluationRun[]
  selectedRunID: string | null
  selectedRun: EvaluationRun | null
  selectedRunLoading: boolean
  selectedRunError: string | null
  onRetrySelectedRun: () => void
  selectedPair: EvaluationControlledPairExecution | null
  selectedPairLoading: boolean
  selectedPairRefreshing: boolean
  selectedPairError: string | null
  onRetrySelectedPair: () => void
  events: EvaluationRunEvent[]
  eventsConnected: boolean
  eventsError: string | null
  onReconnectEvents: () => void
  canRun: boolean
  canDelete: boolean
  refreshing: boolean
  loadingMore: boolean
  runLedgerAvailable: boolean
  autoRefreshPaused: boolean
  totalRuns: number
  hasMoreRuns: boolean
  lastUpdatedAt: Date | null
  mutationKey: string | null
  onSelect: (run: EvaluationRun) => void
  onStart: (run: EvaluationRun) => void
  onCancel: (run: EvaluationRun) => void
  onDelete: (run: EvaluationRun) => void
  onOpenReport: (run: EvaluationRun) => void
  onRefresh: () => void
  onLoadMore: () => void
}

function EvaluationRunsHeader({ props }: { props: EvaluationRunsProps }) {
  const refreshStatus = props.autoRefreshPaused
    ? 'Multiple pages loaded · refresh manually'
    : props.lastUpdatedAt
      ? `Updated ${formatDateTime(props.lastUpdatedAt.toISOString())}`
      : 'Not refreshed yet'
  return (
    <header className={styles.surfaceHeader}>
      <div>
        <span className={styles.eyebrow}>Run history</span>
        <h2 id="evaluation-runs-title">Evaluation runs</h2>
        <p>Find an evaluation, review its setup and outcome, then open its report when ready.</p>
      </div>
      <div className={runStyles.refreshCluster}>
        <span>{refreshStatus}</span>
        <EvaluationActionButton
          type="button"
          compact
          variant="quiet"
          disabled={props.refreshing || props.loadingMore}
          aria-busy={props.refreshing}
          onClick={props.onRefresh}
          aria-label="Refresh evaluation runs"
        >
          {props.refreshing ? 'Refreshing…' : 'Refresh'}
        </EvaluationActionButton>
      </div>
    </header>
  )
}

export default function EvaluationRuns(props: EvaluationRunsProps) {
  const ledger = useEvaluationRunLedger(props.runs)
  return (
    <div className={styles.sectionStack}>
      <section className={styles.surface} aria-labelledby="evaluation-runs-title">
        <EvaluationRunsHeader props={props} />

        <EvaluationRunLedgerFilters
          model={ledger}
          runLedgerAvailable={props.runLedgerAvailable}
          loadedRuns={props.runs.length}
          totalRuns={props.totalRuns}
          hasMoreRuns={props.hasMoreRuns}
        />

        <div className={runStyles.runWorkspace}>
          <EvaluationRunLedger
            runs={props.runs}
            selectedRunID={props.selectedRunID}
            runLedgerAvailable={props.runLedgerAvailable}
            totalRuns={props.totalRuns}
            hasMoreRuns={props.hasMoreRuns}
            loadingMore={props.loadingMore}
            refreshing={props.refreshing}
            model={ledger}
            onSelect={props.onSelect}
            onLoadMore={props.onLoadMore}
          />
          <EvaluationRunInspector
            selectedRunID={props.selectedRunID}
            run={props.selectedRun}
            loading={props.selectedRunLoading}
            error={props.selectedRunError}
            controlledPairExecution={props.selectedPair}
            controlledPairLoading={props.selectedPairLoading}
            controlledPairRefreshing={props.selectedPairRefreshing}
            controlledPairError={props.selectedPairError}
            events={props.events}
            eventsConnected={props.eventsConnected}
            eventsError={props.eventsError}
            canRun={props.canRun}
            canDelete={props.canDelete}
            mutationKey={props.mutationKey}
            onRetry={props.onRetrySelectedRun}
            onRetryControlledPair={props.onRetrySelectedPair}
            onReconnectEvents={props.onReconnectEvents}
            onStart={props.onStart}
            onCancel={props.onCancel}
            onDelete={props.onDelete}
            onOpenReport={props.onOpenReport}
          />
        </div>
      </section>
    </div>
  )
}
