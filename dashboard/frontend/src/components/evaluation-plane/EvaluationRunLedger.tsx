import type {
  EvaluationRun,
  EvaluationRunStatus,
  EvaluationTrackId,
} from '../../types/evaluationPlane'
import { EVALUATION_TRACK_IDS } from '../../types/evaluationPlane'
import { formatDateTime } from '../../utils/dateTime'
import EvaluationIssueDetails from './EvaluationIssueDetails'
import { EvaluationActionButton, RunStatusBadge } from './EvaluationPrimitives'
import { RUN_STATUS_LABELS } from './evaluationPresentation'
import { changeProfileLabel } from './evaluationRunPresentation'
import { TRACK_PRESENTATION } from './evaluationTrackPresentation'
import planeStyles from './EvaluationPlane.module.css'
import styles from './EvaluationRunLedger.module.css'
import type { EvaluationRunLedgerModel } from './useEvaluationRunLedger'

interface EvaluationRunLedgerFiltersProps {
  model: EvaluationRunLedgerModel
  runLedgerAvailable: boolean
  loadedRuns: number
  totalRuns: number
  hasMoreRuns: boolean
}

export function EvaluationRunLedgerFilters({
  model,
  runLedgerAvailable,
  loadedRuns,
  totalRuns,
  hasMoreRuns,
}: EvaluationRunLedgerFiltersProps) {
  const { search, status, track, filteredRuns, setSearch, setStatus, setTrack } = model
  return (
    <>
      <div className={styles.filters} role="search" aria-label="Filter evaluation runs">
        <label>
          <span>Search</span>
          <input
            type="search"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Run, target, or change type"
          />
        </label>
        <label>
          <span>Status</span>
          <select
            value={status}
            onChange={(event) => setStatus(event.target.value as EvaluationRunStatus | 'all')}
          >
            <option value="all">All statuses</option>
            {Object.entries(RUN_STATUS_LABELS).map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </select>
        </label>
        <label>
          <span>Evaluation area</span>
          <select
            value={track}
            onChange={(event) => setTrack(event.target.value as EvaluationTrackId | 'all')}
          >
            <option value="all">All areas</option>
            {EVALUATION_TRACK_IDS.map((id) => (
              <option key={id} value={id}>
                {TRACK_PRESENTATION[id].label}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className={styles.filterSummary} aria-live="polite">
        {runLedgerAvailable ? (
          <>
            <strong>{filteredRuns.length}</strong> matching run
            {filteredRuns.length === 1 ? '' : 's'}
            <span aria-hidden="true">·</span>
            <span>
              {loadedRuns} of {totalRuns} loaded
            </span>
          </>
        ) : (
          'Run history unavailable'
        )}
      </div>
      {hasMoreRuns ? (
        <p className={planeStyles.scopeNotice} role="status">
          Search and filters cover only the {loadedRuns} loaded runs. Load older records to search
          and filter the full history.
        </p>
      ) : null}
    </>
  )
}

interface EvaluationRunLedgerProps {
  runs: EvaluationRun[]
  selectedRunID: string | null
  runLedgerAvailable: boolean
  totalRuns: number
  hasMoreRuns: boolean
  loadingMore: boolean
  refreshing: boolean
  model: EvaluationRunLedgerModel
  onSelect: (run: EvaluationRun) => void
  onLoadMore: () => void
}

function runProgressSummary(run: EvaluationRun): string {
  switch (run.status) {
    case 'pending':
      return 'Awaiting start'
    case 'running':
      return run.progress.total > 0
        ? `${run.progress.completed} of ${run.progress.total} steps complete`
        : 'Evaluation in progress'
    case 'sealing':
      return 'Finalizing verified results'
    case 'completed':
      return 'Report ready'
    case 'failed':
      return 'Stopped before completion'
    case 'cancelled':
      return 'Cancelled before completion'
  }
}

function RunLedgerEmpty({
  runs,
  available,
  filtersActive,
  onReset,
}: {
  runs: EvaluationRun[]
  available: boolean
  filtersActive: boolean
  onReset: () => void
}) {
  return (
    <div className={planeStyles.emptyState}>
      <div>
        <strong>
          {!available
            ? 'Run history is unavailable.'
            : runs.length
              ? 'No runs match these filters.'
              : 'No evaluation runs yet.'}
        </strong>
        <p>
          {!available
            ? 'Retry before interpreting run history.'
            : runs.length
              ? 'Reset filters to return to all runs.'
              : 'Create an experiment to establish the first evidence baseline.'}
        </p>
      </div>
      {filtersActive ? (
        <EvaluationActionButton type="button" compact onClick={onReset}>
          Reset filters
        </EvaluationActionButton>
      ) : null}
    </div>
  )
}

function RunLedgerList({
  runs,
  selectedRunID,
  onSelect,
}: {
  runs: EvaluationRun[]
  selectedRunID: string | null
  onSelect: (run: EvaluationRun) => void
}) {
  return (
    <ol className={styles.runList} aria-label="Evaluation run history">
      {runs.map((run) => (
        <li
          key={run.id}
          className={`${styles.runRow} ${selectedRunID === run.id ? styles.runSelected : ''}`}
        >
          <button
            type="button"
            data-evaluation-ledger-row="true"
            className={styles.runSummary}
            aria-label={`Open ${run.name} details`}
            aria-current={selectedRunID === run.id ? 'true' : undefined}
            onClick={() => onSelect(run)}
          >
            <span className={styles.runRowTop}>
              <strong>{run.name}</strong>
              <RunStatusBadge status={run.status} />
            </span>
            <span className={styles.runRowMeta}>
              {run.mixture?.entrypoint_model || (run.mode === 'live' ? 'Live run' : 'Replay')} ·{' '}
              {run.mixture?.recipe_name || changeProfileLabel(run.change_profile)} ·{' '}
              {formatDateTime(run.created_at)}
            </span>
            <span className={styles.runRowProgress}>
              {Math.round(run.progress.percent)}% · {runProgressSummary(run)}
            </span>
          </button>
        </li>
      ))}
    </ol>
  )
}

function RunLedgerPagination({
  page,
  pages,
  setPage,
}: Pick<EvaluationRunLedgerModel, 'page' | 'pages' | 'setPage'>) {
  if (pages <= 1) return null
  return (
    <nav className={styles.pagination} aria-label="Run history pages">
      <EvaluationActionButton
        type="button"
        compact
        disabled={page === 1}
        onClick={() => setPage((value) => value - 1)}
      >
        Previous
      </EvaluationActionButton>
      <span>
        Page {page} of {pages}
      </span>
      <EvaluationActionButton
        type="button"
        compact
        disabled={page === pages}
        onClick={() => setPage((value) => value + 1)}
      >
        Next
      </EvaluationActionButton>
    </nav>
  )
}

function RunLedgerLoadMore(props: EvaluationRunLedgerProps) {
  if (!props.hasMoreRuns) return null
  return (
    <div className={styles.pagination} aria-label="Load more evaluation runs">
      <span>
        {props.runs.length} of {props.totalRuns} runs loaded
      </span>
      <EvaluationActionButton
        type="button"
        compact
        disabled={props.loadingMore || props.refreshing}
        onClick={props.onLoadMore}
      >
        {props.loadingMore ? 'Loading more…' : 'Load more'}
      </EvaluationActionButton>
    </div>
  )
}

export default function EvaluationRunLedger(props: EvaluationRunLedgerProps) {
  const { visibleRuns, filtersActive, resetFilters } = props.model
  const progressDetails = visibleRuns.flatMap((run) =>
    run.progress.message ? [{ label: `${run.name} progress`, message: run.progress.message }] : [],
  )
  return (
    <div className={styles.runLedger}>
      {visibleRuns.length === 0 ? (
        <RunLedgerEmpty
          runs={props.runs}
          available={props.runLedgerAvailable}
          filtersActive={filtersActive}
          onReset={resetFilters}
        />
      ) : (
        <RunLedgerList
          runs={visibleRuns}
          selectedRunID={props.selectedRunID}
          onSelect={props.onSelect}
        />
      )}
      <EvaluationIssueDetails className={styles.ledgerTechnicalDetails} issues={progressDetails} />
      <RunLedgerPagination
        page={props.model.page}
        pages={props.model.pages}
        setPage={props.model.setPage}
      />
      <RunLedgerLoadMore {...props} />
    </div>
  )
}
