import { useEffect, useState } from 'react'
import { ArrowLeft, Search } from 'lucide-react'
import { Button, InputGroup } from '../../ui'
import { benchApi } from './api'
import BenchPagination from './BenchPagination'
import CallDetail from './CallDetail'
import { money, number, seconds } from './model'
import type { CallRecord, Manifest, PageState } from './types'
import { targetName } from './targetPresentation'
import { BenchLoading, Notice, TableFrame } from './benchUi'
import { benchTable, benchText, linkButtonClass } from './benchStyles'

export default function CallEvidence({
  id,
  calls,
  manifest,
  page,
  loadMore,
  accountingReconciled = false,
}: {
  id: string
  calls: CallRecord[]
  manifest: Manifest
  page: PageState
  loadMore: () => Promise<void>
  accountingReconciled?: boolean
}) {
  const [listPage, setListPage] = useState(0)
  const [filter, setFilter] = useState('')
  const visible = calls.filter((call) =>
    `${call.id} ${call.case_id} ${call.target_id} ${targetName(manifest, call.target_id)} ${call.role} ${call.status} ${call.selected_model ?? call.model ?? ''}`
      .toLowerCase()
      .includes(filter.toLowerCase()),
  )
  const currentPage = Math.min(listPage, Math.max(0, Math.ceil(visible.length / 25) - 1))
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
    <section className="min-w-0">
      <div hidden={!!selected}>
        <div className="mb-1 flex flex-wrap items-end justify-between gap-3">
          <h3 className={benchText.h3}>Call records</h3>
          <InputGroup
            type="search"
            aria-label="Filter loaded calls"
            value={filter}
            placeholder="Call, case, target or status"
            leading={<Search aria-hidden="true" />}
            containerClassName="w-full sm:w-72"
            className="h-8"
            onChange={(event) => {
              setFilter(event.target.value)
              setListPage(0)
            }}
          />
        </div>
        {page.total !== null && (
          <p className={benchText.muted}>
            Loaded {number(calls.length)} of {number(page.total)} persisted call summaries.
            Filtering applies to loaded calls. Prompts and responses load only when you inspect a
            call. These pages are snapshots; refresh evidence to reload current statuses. Aggregate
            metrics above always come from the full report.
          </p>
        )}
        {page.loading && page.total === null && (
          <BenchLoading label="Loading persisted call records…" lines={3} />
        )}
        {!page.loading && !page.error && page.total === 0 && (
          <p className={`${benchText.muted} mt-2`}>No persisted call records yet.</p>
        )}
        {accountingReconciled && (
          <p className={`${benchText.muted} mt-1`}>
            Original receipt accounting; see the report for reconciled totals. Saved usage and costs
            below remain unchanged.
          </p>
        )}
        <TableFrame
          className="max-h-[min(28rem,60vh)] overscroll-contain [scrollbar-gutter:stable] [&_thead_th]:sticky [&_thead_th]:top-0 [&_thead_th]:z-[1]"
          role="region"
          aria-label="Persisted call records"
          tabIndex={0}
        >
          <table className={benchTable}>
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
              {visible.slice(currentPage * 25, currentPage * 25 + 25).map((call) => (
                <tr key={call.id}>
                  <td>
                    <button
                      className={`${linkButtonClass} font-mono text-xs`}
                      onClick={() => setSelected(call.id)}
                    >
                      {call.id}
                    </button>
                  </td>
                  <td>
                    {call.case_id} / {targetName(manifest, call.target_id)}
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
        </TableFrame>
        {!visible.length && calls.length > 0 && (
          <p className={benchText.muted}>No loaded calls match this filter.</p>
        )}
        <BenchPagination
          label="Calls"
          total={visible.length}
          page={currentPage}
          pageSize={25}
          onChange={setListPage}
        />
        {page.error && (
          <Notice tone="danger" as="p" role="alert">
            {page.error}
          </Notice>
        )}
        {page.nextCursor !== null && (
          <Button
            variant="outline"
            size="sm"
            disabled={page.loading}
            onClick={() => void loadMore()}
          >
            {page.loading ? 'Loading call records…' : 'Load more call records'}
          </Button>
        )}
      </div>
      {selected && (
        <div className="min-w-0">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
            <h4 className={`${benchText.h3} break-all`}>
              Call evidence: <span className="font-mono text-[12.5px]">{selected}</span>
            </h4>
            <Button variant="outline" size="sm" onClick={() => setSelected(null)}>
              <ArrowLeft aria-hidden="true" />
              Back to calls
            </Button>
          </div>
          {accountingReconciled && (
            <p className={benchText.muted}>
              Original receipt accounting; see the report for reconciled totals.
            </p>
          )}
          {error ? (
            <Notice tone="danger" as="p" role="alert">
              {error}
            </Notice>
          ) : detail ? (
            <CallDetail key={detail.id} call={detail} />
          ) : (
            <BenchLoading label="Loading call evidence…" lines={4} />
          )}
        </div>
      )}
    </section>
  )
}
