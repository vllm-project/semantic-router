import { useState } from 'react'
import { Button, cn } from '../../ui'
import type { Manifest, PageState, RunEvent } from './types'
import BenchSelect from './BenchSelect'
import BenchPagination from './BenchPagination'
import { describeRunEvent } from './eventPresentation'
import { targetName } from './targetPresentation'
import { BenchLoading, EmptyNote, Notice } from './benchUi'
import { benchText } from './benchStyles'

export default function RunEvents({
  manifest,
  events,
  page: evidencePage,
  loadMore,
}: {
  manifest: Manifest
  events: RunEvent[]
  page: PageState
  loadMore: () => Promise<void>
}) {
  const { loading, error } = evidencePage
  const [filter, setFilter] = useState('all')
  const [page, setPage] = useState(0)
  const rows = events
    .map((event) => ({ event, ...describeRunEvent(event) }))
    .filter(
      (row) => filter === 'all' || (filter === 'attention' ? row.attention : row.group === filter),
    )
  const current = Math.min(page, Math.max(0, Math.ceil(rows.length / 25) - 1))
  return (
    <section className="mt-6 min-w-0" aria-labelledby="run-events">
      <div className="mb-2 flex flex-wrap items-start justify-between gap-x-6 gap-y-3">
        <div className="flex min-w-0 flex-1 flex-col gap-1">
          <h3 id="run-events" className={benchText.h3}>
            Run events ({events.length}
            {evidencePage.nextCursor !== null ? ' loaded' : ''})
          </h3>
          <p className={cn(benchText.muted, 'max-w-3xl')}>
            Saved activity, oldest first. Filtering applies to loaded events. Refresh evidence for a
            new snapshot; run progress continues updating independently. Scores and costs are in
            Results.
          </p>
        </div>
        <BenchSelect
          variant="toolbar"
          label="Event type"
          value={filter}
          onChange={(value) => {
            setFilter(value)
            setPage(0)
          }}
          options={[
            { value: 'all', label: 'All events' },
            { value: 'run', label: 'Run lifecycle' },
            { value: 'cases', label: 'Case results' },
            { value: 'requests', label: 'Model requests' },
            { value: 'attention', label: 'Needs attention' },
          ]}
        />
      </div>
      {loading && !events.length && <BenchLoading label="Loading saved run events…" lines={3} />}
      {error && (
        <Notice tone="danger" as="p" role="alert">
          Run events are unavailable: {error}
        </Notice>
      )}
      <ol
        className="my-2 min-w-0 divide-y divide-border rounded-lg border border-border bg-card empty:hidden"
        aria-label="Saved run events"
      >
        {rows.slice(current * 25, current * 25 + 25).map((row, index) => (
          <li
            key={row.event.seq ?? row.event.sequence ?? current * 25 + index}
            className="break-words px-4 py-3"
          >
            <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
              <strong
                className={cn(
                  'text-[13px] font-medium',
                  row.attention ? 'text-destructive-foreground' : 'text-foreground',
                )}
              >
                {row.title}
              </strong>
              {row.timestamp && (
                <time
                  dateTime={row.timestamp}
                  className="text-[11px] text-subtle-foreground tabular-nums"
                >
                  {new Date(row.timestamp).toLocaleString()}
                </time>
              )}
            </div>
            <p className={`${benchText.body} mt-1`}>{row.description}</p>
            {(row.caseID || row.targetID || row.role) && (
              <p className={`${benchText.muted} mt-1`}>
                {[
                  row.caseID && `Case: ${row.caseID}`,
                  row.targetID && `Model: ${targetName(manifest, row.targetID)}`,
                  row.role && `Role: ${row.role}`,
                ]
                  .filter(Boolean)
                  .join(' · ')}
              </p>
            )}
            {row.reason && <p className="mt-1 text-xs text-destructive-foreground">{row.reason}</p>}
            {row.callID && (
              <small className="mt-1 block font-mono text-[11px] text-subtle-foreground">
                Call: {row.callID}
              </small>
            )}
          </li>
        ))}
      </ol>
      {!rows.length && !loading && !error && (
        <EmptyNote>
          {events.length ? 'No events match this filter.' : 'No events have been saved yet.'}
        </EmptyNote>
      )}
      {evidencePage.nextCursor !== null && (
        <p className={benchText.muted}>
          More events may be available. Load the next saved page to continue.
        </p>
      )}
      <BenchPagination
        label="Events"
        total={rows.length}
        page={current}
        pageSize={25}
        onChange={setPage}
      />
      {evidencePage.nextCursor !== null && (
        <Button
          variant="outline"
          size="sm"
          className="mt-2"
          disabled={loading}
          onClick={() => void loadMore()}
        >
          {loading ? 'Loading events…' : 'Load more events'}
        </Button>
      )}
    </section>
  )
}
