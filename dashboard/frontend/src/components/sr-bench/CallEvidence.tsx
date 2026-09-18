import { useEffect, useState } from 'react'
import { benchApi } from './api'
import { money, number, seconds } from './model'
import type { CallRecord, PageState } from './types'
import styles from './SrBench.module.css'
import ProductIcon from '../ProductIcon'
import ProductLoadingState from '../ProductLoadingState'

export default function CallEvidence({
  id,
  calls,
  page,
  loadMore,
  accountingReconciled = false,
}: {
  id: string
  calls: CallRecord[]
  page: PageState
  loadMore: () => Promise<void>
  accountingReconciled?: boolean
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
      <div hidden={!!selected}>
        <h3>Call records</h3>
        {page.total !== null && (
          <p className={styles.muted}>
            Showing {number(calls.length)} of {number(page.total)} persisted call summaries. Prompts
            and responses load only when you inspect a call. These pages are snapshots; refresh
            evidence to reload current statuses. Aggregate metrics above always come from the full
            report.
          </p>
        )}
        {page.loading && page.total === null && (
          <ProductLoadingState compact label="Loading persisted call records…" />
        )}
        {!page.loading && !page.error && page.total === 0 && (
          <p className={styles.muted}>No persisted call records yet.</p>
        )}
        {accountingReconciled && (
          <p className={styles.muted}>
            Original receipt accounting; see the report for reconciled totals. Saved usage and costs
            below remain unchanged.
          </p>
        )}
        <div
          className={`${styles.tableScroll} ${styles.callTableScroll}`}
          role="region"
          aria-label="Persisted call records"
          tabIndex={0}
        >
          <table>
            <thead>
              <tr>
                <th>Call</th>
                <th>Case / target</th>
                <th>Role</th>
                <th>Status</th>
                <th>Model</th>
                <th>{accountingReconciled ? 'Original cost / latency' : 'Cost / latency'}</th>
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
      </div>
      {selected && (
        <div className={styles.recordDetail}>
          <div className={styles.sectionHeading}>
            <h4>Call evidence: {selected}</h4>
            <button onClick={() => setSelected(null)}>
              <ProductIcon name="arrow-left" />
              Back to calls
            </button>
          </div>
          {accountingReconciled && (
            <p className={styles.muted}>
              Original receipt accounting; see the report for reconciled totals.
            </p>
          )}
          {error ? (
            <p role="alert" className={styles.error}>
              {error}
            </p>
          ) : detail ? (
            <>
              <div className={styles.metricGrid}>
                <div>
                  <span>Status</span>
                  <strong>{detail.status}</strong>
                </div>
                <div>
                  <span>Selected model</span>
                  <strong>{detail.selected_model ?? detail.model ?? '—'}</strong>
                </div>
                <div>
                  <span>Model cost</span>
                  <strong>{money(detail.cost_usd)}</strong>
                </div>
                <div>
                  <span>Latency</span>
                  <strong>{seconds(detail.latency_s)}</strong>
                </div>
              </div>
              <details className={styles.details}>
                <summary>Original call receipt</summary>
                <pre>{JSON.stringify(detail, null, 2)}</pre>
              </details>
            </>
          ) : (
            <ProductLoadingState compact label="Loading call evidence…" />
          )}
        </div>
      )}
    </section>
  )
}
