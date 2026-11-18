import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'

import type { Column } from '../components/DataTable'
import { DataTable } from '../components/DataTable'
import TableHeader from '../components/TableHeader'
import { formatDateTime } from '../utils/dateTime'

import configStyles from './ConfigPage.module.css'
import styles from './InsightsPage.module.css'
import runStyles from './InsightsRun.module.css'
import { isInsightsReplayUnavailableError } from './insightsPageApi'
import { fetchAbortableInsightsJSON, isAbortError } from './insightsPageRequestSupport'
import { ModelName } from './insightsModelLabelSupport'
import {
  countText,
  formatRunCost,
  formatRunDuration,
  getRunPath,
  HARNESS_LABELS,
  modelHue,
  RUN_MARKER_GLYPHS,
  RUN_MARKER_LABELS,
  RUN_OUTCOME_LABELS,
  runAgentTreeText,
  runTitle,
} from './insightsRunSupport'
import type { RunListResponse, RunSummary } from './insightsRunTypes'

const runPageSize = 25

export function RunOutcomeBadge({ run }: { run: RunSummary }) {
  const status = run.outcome.status
  return (
    <span className={runStyles.outcome} data-status={status}>
      {RUN_OUTCOME_LABELS[status] ?? status}
    </span>
  )
}

export function RunEvents({ events }: { events?: Record<string, number> }) {
  const entries = Object.entries(events ?? {})
  if (entries.length === 0) return null
  return (
    <span className={runStyles.events}>
      {entries.map(([marker, count]) => (
        <span
          key={marker}
          className={runStyles.marker}
          data-marker={marker}
          title={`${RUN_MARKER_LABELS[marker] ?? marker}: ${count}`}
        >
          {RUN_MARKER_GLYPHS[marker] ?? '•'} {count}
        </span>
      ))}
    </span>
  )
}

function createRunColumns(): Column<RunSummary>[] {
  return [
    {
      key: 'task',
      header: 'Run',
      width: '300px',
      render: (run) => (
        <Link className={runStyles.runTask} to={getRunPath(run.session_id)}>
          <strong title={run.session_name ? run.prompt : undefined}>{runTitle(run)}</strong>
          <span>
            {formatDateTime(run.started_at)} · {formatRunDuration(run.duration_ms)} ·{' '}
            {countText(run.calls, 'call')}
          </span>
        </Link>
      ),
    },
    {
      key: 'harness',
      header: 'Harness',
      width: '110px',
      render: (run) => HARNESS_LABELS[run.harness ?? ''] ?? 'Client',
    },
    { key: 'agents', header: 'Agents', width: '150px', render: runAgentTreeText },
    {
      key: 'models',
      header: 'Models',
      width: '180px',
      render: (run) => (
        <span className={runStyles.models}>
          {run.models.map((model) => (
            <span key={model.name} title={`${model.value} calls`}>
              <i style={{ background: modelHue(run.models, model.name) }} />
              <ModelName name={model.name} compact />
            </span>
          ))}
        </span>
      ),
    },
    {
      key: 'outcome',
      header: 'Outcome',
      width: '190px',
      render: (run) => (
        <span className={runStyles.events}>
          <RunOutcomeBadge run={run} />
          <RunEvents events={run.events} />
        </span>
      ),
    },
    {
      key: 'cost',
      header: 'Cost',
      width: '110px',
      align: 'right',
      render: (run) => formatRunCost(run.cost),
    },
  ]
}

function useRuns(search: string, page: number) {
  const [state, setState] = useState<{ runs: RunSummary[]; total: number; error: string | null }>({
    runs: [],
    total: 0,
    error: null,
  })
  const abortRef = useRef<AbortController | null>(null)
  const refresh = useCallback(async () => {
    abortRef.current?.abort()
    const abort = new AbortController()
    abortRef.current = abort
    const query = new URLSearchParams({
      limit: String(runPageSize),
      offset: String((page - 1) * runPageSize),
    })
    if (search.trim()) query.set('search', search.trim())
    try {
      const list = await fetchAbortableInsightsJSON<RunListResponse>(
        `/api/router/api/v1/observability/replays/runs?${query.toString()}`,
        'agent runs',
        abort.signal,
      )
      setState({ runs: list.data || [], total: list.total, error: null })
    } catch (err) {
      if (isAbortError(err)) return
      const message = err instanceof Error ? err.message : 'Unknown error'
      setState({
        runs: [],
        total: 0,
        error: isInsightsReplayUnavailableError(err) ? null : message,
      })
    }
  }, [page, search])
  useEffect(() => {
    void refresh()
    return () => abortRef.current?.abort()
  }, [refresh])
  return { ...state, refresh }
}

// One row per agent run: the task, the harness, its agents, the models that
// answered, how it ended, what it cost and took, and what to look at.
export default function InsightsRunsView() {
  const [search, setSearch] = useState('')
  const [page, setPage] = useState(1)
  const { runs, total, error, refresh } = useRuns(search, page)
  const columns = useMemo(() => createRunColumns(), [])
  const totalPages = Math.max(1, Math.ceil(total / runPageSize))
  return (
    <>
      {error ? <div className={styles.error}>{error}</div> : null}
      <div className={configStyles.sectionPanel}>
        <section className={configStyles.sectionTableBlock}>
          <div className={styles.toolbar}>
            <div>
              <h2 className={styles.sectionTitle}>Runs</h2>
              <p className={styles.sectionSubtitle}>
                One row per agent run, the session a harness names. Open a run for its timeline.
              </p>
            </div>
            <div className={styles.toolbarActions}>
              <button type="button" onClick={() => void refresh()} className={styles.refreshButton}>
                Refresh
              </button>
            </div>
          </div>
          <TableHeader
            title="Agent Runs"
            count={total}
            searchPlaceholder="Search by session ID..."
            searchValue={search}
            onSearchChange={(value) => {
              setSearch(value)
              setPage(1)
            }}
            variant="embedded"
          />
          <DataTable
            columns={columns}
            data={runs}
            keyExtractor={(run) => run.session_id}
            emptyMessage="No runs recorded yet"
            className={styles.insightsTable}
          />
        </section>
      </div>
      {total > runPageSize ? (
        <div className={styles.pagination}>
          <button
            type="button"
            className={styles.paginationButton}
            onClick={() => setPage((current) => Math.max(1, current - 1))}
            disabled={page === 1}
          >
            Previous
          </button>
          <span className={styles.paginationInfo}>
            Page {page} of {totalPages} ({total} runs)
          </span>
          <button
            type="button"
            className={styles.paginationButton}
            onClick={() => setPage((current) => Math.min(totalPages, current + 1))}
            disabled={page === totalPages}
          >
            Next
          </button>
        </div>
      ) : null}
    </>
  )
}
