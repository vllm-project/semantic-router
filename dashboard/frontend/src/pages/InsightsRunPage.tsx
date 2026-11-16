import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'

import type { Column } from '../components/DataTable'
import { DataTable } from '../components/DataTable'
import ProductIcon from '../components/ProductIcon'
import ProductLoadingState from '../components/ProductLoadingState'
import { formatDateTime } from '../utils/dateTime'
import { formatInsightsCost } from '../utils/insightsCost'

import styles from './InsightsPage.module.css'
import runStyles from './InsightsRun.module.css'
import { fetchInsightsJSON } from './insightsPageApi'
import { RunLegend } from './InsightsRunDiagram'
import { RunSequence } from './InsightsRunSequence'
import { ModelLabelsProvider, ModelName } from './insightsModelLabelSupport'
import { RunEvents, RunOutcomeBadge } from './InsightsRunsView'
import { compactID } from './insightsReviewSupport'
import {
  buildRunSequence,
  useNarrowRunLayout,
  type RunCallsRow,
  type RunSequence as RunSequenceLayout,
} from './insightsRunSequence'
import {
  countText,
  describeCallStatus,
  formatRunCost,
  formatRunDuration,
  HARNESS_LABELS,
  runAgentTreeText,
  RUN_MARKER_GLYPHS,
  runTitle,
} from './insightsRunSupport'
import type { RunCall, RunResponse } from './insightsRunTypes'

function createCallColumns(): Column<RunCall>[] {
  return [
    {
      key: 'time',
      header: 'Time',
      width: '150px',
      render: (call) => formatDateTime(call.timestamp),
    },
    { key: 'agent', header: 'Agent', width: '110px', render: (call) => call.agent },
    {
      key: 'model',
      header: 'Model',
      width: '140px',
      render: (call) => <ModelName name={call.model} compact />,
    },
    { key: 'decision', header: 'Decision', width: '130px', render: (call) => call.decision || '-' },
    { key: 'status', header: 'Status', width: '110px', render: describeCallStatus },
    {
      key: 'cost',
      header: 'Cost',
      width: '90px',
      align: 'right',
      render: (call) =>
        call.cost !== undefined ? formatInsightsCost(call.cost, call.currency) : '-',
    },
    {
      key: 'markers',
      header: 'Events',
      width: '90px',
      render: (call) => (call.markers ?? []).map((marker) => RUN_MARKER_GLYPHS[marker]).join(' '),
    },
  ]
}

function useRun(sessionId: string) {
  const [response, setResponse] = useState<RunResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const load = useCallback(async () => {
    setLoading(true)
    try {
      const query = new URLSearchParams({ session_id: sessionId })
      setResponse(
        await fetchInsightsJSON<RunResponse>(
          `/api/router/api/v1/observability/replays/run?${query.toString()}`,
          'agent run',
        ),
      )
      setError(null)
    } catch (err) {
      setResponse(null)
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }, [sessionId])
  useEffect(() => {
    void load()
  }, [load])
  return { response, error, loading, load }
}

function RunHero({ response }: { response: RunResponse }) {
  const run = response.run
  return (
    <header className={styles.recordHero}>
      <div className={styles.recordHeroCopy}>
        <span className={styles.recordEyebrow}>
          Agent run · {HARNESS_LABELS[run.harness ?? ''] ?? 'Client'} · session{' '}
          {compactID(run.session_id)}
        </span>
        <h1 className={runStyles.task} title={run.prompt}>
          {runTitle(run)}
        </h1>
        <div className={runStyles.heroStats}>
          <span>{runAgentTreeText(run)}</span>
          <span>{countText(run.calls, 'call')}</span>
          <span>{formatRunDuration(run.duration_ms)}</span>
          <span>{formatRunCost(run.cost)}</span>
          <span>
            tools {run.outcome.tool_results}, failed {run.outcome.tool_failures}
          </span>
          {run.outcome.reported?.length ? (
            <span>
              reported{' '}
              {run.outcome.reported
                .map(({ outcome }) => `${outcome.source}: ${outcome.verdict}`)
                .join(', ')}
            </span>
          ) : null}
          <RunEvents events={run.events} />
        </div>
      </div>
      <RunOutcomeBadge run={run} />
    </header>
  )
}

// One agent run, its models read by their labels.
export default function InsightsRunPage() {
  return (
    <ModelLabelsProvider>
      <AgentRun />
    </ModelLabelsProvider>
  )
}

// The row whose details show, and the call picked in it: at first the row of
// the step a link names.
function useOpenRow(sequence: RunSequenceLayout | undefined, step: string | null) {
  // undefined until a row is opened or closed: the row of the linked step opens.
  const [expanded, setExpanded] = useState<string | null>()
  const [callId, setCallId] = useState<string | undefined>(step ?? undefined)
  const rows = useMemo(
    () =>
      (sequence?.rows ?? []).filter(
        (row): row is RunCallsRow => row.kind === 'turns' || row.kind === 'side',
      ),
    [sequence],
  )
  const open =
    expanded === undefined
      ? rows.find((row) => row.calls.some((entry) => entry.record_id === callId))
      : rows.find((row) => row.key === expanded)
  const call =
    open?.calls.find((entry) => entry.record_id === callId) ??
    (open?.calls.length === 1 ? open.calls[0] : undefined)
  const toggle = (row: RunCallsRow) => {
    setExpanded(open?.key === row.key ? null : row.key)
    setCallId(row.calls.length === 1 ? row.calls[0].record_id : undefined)
  }
  const selectCall = (next: RunCall) => {
    const row = rows.find((entry) => entry.calls.includes(next))
    setExpanded(row?.key ?? null)
    setCallId(next.record_id)
    if (row) document.getElementById(`run-row-${row.key}`)?.scrollIntoView({ block: 'nearest' })
  }
  return { open, call, toggle, selectCall }
}

// One agent run: its agents as lifelines with their calls top to bottom, a
// row's details under it on click, and every call in a table one click further.
function AgentRun() {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const sessionId = searchParams.get('session') ?? ''
  const { response, error, loading, load } = useRun(sessionId)
  const narrow = useNarrowRunLayout()
  const sequence = useMemo(() => (response ? buildRunSequence(response) : undefined), [response])
  const [showCalls, setShowCalls] = useState(false)
  const columns = useMemo(() => createCallColumns(), [])
  const { open, call, toggle, selectCall } = useOpenRow(sequence, searchParams.get('step'))

  return (
    <main className={styles.recordPage}>
      <div className={styles.recordToolbar}>
        <button
          type="button"
          className={styles.recordBack}
          onClick={() => navigate('/insights?view=runs')}
        >
          <ProductIcon name="arrow-left" width={16} height={16} />
          Runs
        </button>
        <button type="button" onClick={() => void load()} className={styles.recordAction}>
          <ProductIcon name="refresh" width={15} height={15} />
          Refresh
        </button>
      </div>
      {error ? <div className={styles.error}>{error}</div> : null}
      {loading && !response ? <ProductLoadingState label="Loading run" compact /> : null}
      {response ? (
        <>
          <RunHero response={response} />
          <section className={runStyles.section}>
            <RunLegend run={response.run} />
            {sequence ? (
              <RunSequence
                run={response.run}
                sequence={sequence}
                expanded={open?.key}
                call={call}
                narrow={narrow}
                onToggle={toggle}
                onSelectCall={selectCall}
              />
            ) : null}
          </section>
          <section className={runStyles.section}>
            <button
              type="button"
              className={styles.recordAction}
              onClick={() => setShowCalls((shown) => !shown)}
            >
              {showCalls ? 'Hide' : 'Show'} all {countText(response.calls.length, 'call')}
            </button>
            {showCalls ? (
              <DataTable
                columns={columns}
                data={response.calls}
                keyExtractor={(entry) => entry.record_id}
                onView={selectCall}
                className={styles.insightsTable}
              />
            ) : null}
          </section>
        </>
      ) : null}
    </main>
  )
}
