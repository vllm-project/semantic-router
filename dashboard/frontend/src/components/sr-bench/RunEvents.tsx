import { useState } from 'react'
import ProductLoadingState from '../ProductLoadingState'
import type { Manifest, PageState, RunEvent } from './types'
import BenchSelect from './BenchSelect'
import BenchPagination from './BenchPagination'
import { describeRunEvent } from './eventPresentation'
import { targetName } from './targetPresentation'
import styles from './SrBench.module.css'

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
    <section aria-labelledby="run-events">
      <div className={styles.sectionHeading}>
        <div>
          <h3 id="run-events">
            Run events ({events.length}
            {evidencePage.nextCursor !== null ? ' loaded' : ''})
          </h3>
          <p className={styles.muted}>
            Saved activity, oldest first. Filtering applies to loaded events. Refresh evidence for a
            new snapshot; run progress continues updating independently. Scores and costs are in
            Results.
          </p>
        </div>
        <BenchSelect
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
      {loading && !events.length && (
        <ProductLoadingState compact label="Loading saved run events…" />
      )}
      {error && (
        <p role="alert" className={styles.error}>
          Run events are unavailable: {error}
        </p>
      )}
      <ol className={styles.eventList} aria-label="Saved run events">
        {rows.slice(current * 25, current * 25 + 25).map((row, index) => (
          <li key={row.event.seq ?? row.event.sequence ?? current * 25 + index}>
            <div className={styles.eventHeading}>
              <strong className={row.attention ? styles.eventAttention : undefined}>
                {row.title}
              </strong>
              {row.timestamp && (
                <time dateTime={row.timestamp}>{new Date(row.timestamp).toLocaleString()}</time>
              )}
            </div>
            <p>{row.description}</p>
            {(row.caseID || row.targetID || row.role) && (
              <p className={styles.muted}>
                {[
                  row.caseID && `Case: ${row.caseID}`,
                  row.targetID && `Model: ${targetName(manifest, row.targetID)}`,
                  row.role && `Role: ${row.role}`,
                ]
                  .filter(Boolean)
                  .join(' · ')}
              </p>
            )}
            {row.reason && <p className={styles.eventAttention}>{row.reason}</p>}
            {row.callID && <small className={styles.muted}>Call: {row.callID}</small>}
          </li>
        ))}
      </ol>
      {!rows.length && !loading && !error && (
        <p className={styles.emptyState}>
          {events.length ? 'No events match this filter.' : 'No events have been saved yet.'}
        </p>
      )}
      {evidencePage.nextCursor !== null && (
        <p className={styles.muted}>
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
        <button disabled={loading} onClick={() => void loadMore()}>
          {loading ? 'Loading events…' : 'Load more events'}
        </button>
      )}
    </section>
  )
}
