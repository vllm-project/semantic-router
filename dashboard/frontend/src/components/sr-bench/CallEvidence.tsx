import { useEffect, useState } from 'react'
import { benchApi } from './api'
import { money, number, seconds } from './model'
import type { CallRecord, PageState } from './types'
import styles from './SrBench.module.css'

export default function CallEvidence({
  id,
  calls,
  page,
  loadMore,
}: {
  id: string
  calls: CallRecord[]
  page: PageState
  loadMore: () => Promise<void>
}) {
  const [selected, setSelected] = useState<string | null>(null)
  const [detail, setDetail] = useState<CallRecord | null>(null)
  const [error, setError] = useState('')
  useEffect(() => {
    const controller = new AbortController()
    setDetail(null)
    setError('')
    if (selected)
      void benchApi
        .call(id, selected, controller.signal)
        .then((call) => {
          if (!controller.signal.aborted) setDetail(call)
        })
        .catch((cause) => {
          if (!controller.signal.aborted)
            setError(cause instanceof Error ? cause.message : 'Could not load this call.')
        })
    return () => controller.abort()
  }, [id, selected])
  return (
    <section>
      <h3>Call records</h3>
      <p className={styles.muted}>
        Showing {number(calls.length)} of {number(page.total)} persisted call summaries. Prompts and
        responses load only when you inspect a call. These pages are snapshots; refresh evidence to
        reload current statuses. Aggregate metrics above always come from the full report.
      </p>
      <div className={styles.tableScroll}>
        <table>
          <thead>
            <tr>
              <th>Call</th>
              <th>Case / target</th>
              <th>Role</th>
              <th>Status</th>
              <th>Model</th>
              <th>Cost / latency</th>
            </tr>
          </thead>
          <tbody>
            {calls.map((call) => (
              <tr key={call.id}>
                <td>
                  <button className={styles.linkButton} onClick={() => setSelected(call.id)}>
                    {call.id}
                  </button>
                </td>
                <td>
                  {call.case_id} / {call.target_id}
                </td>
                <td>{call.role}</td>
                <td>{call.status}</td>
                <td>{call.selected_model ?? call.model ?? '—'}</td>
                <td>
                  {money(call.cost_usd)} / {seconds(call.latency_s)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {page.error && (
        <p role="alert" className={styles.error}>
          {page.error}
        </p>
      )}
      {page.nextCursor !== null && (
        <button disabled={page.loading} onClick={() => void loadMore()}>
          {page.loading ? 'Loading call records…' : 'Load more call records'}
        </button>
      )}
      {selected && (
        <div className={styles.caseDetail}>
          <div className={styles.sectionHeading}>
            <h4>Call evidence: {selected}</h4>
            <button onClick={() => setSelected(null)}>Close call</button>
          </div>
          {error ? (
            <p role="alert" className={styles.error}>
              {error}
            </p>
          ) : detail ? (
            <pre>{JSON.stringify(detail, null, 2)}</pre>
          ) : (
            <p role="status">Loading call evidence…</p>
          )}
        </div>
      )}
    </section>
  )
}
