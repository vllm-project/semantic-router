import { useState } from 'react'
import BenchPagination from './BenchPagination'
import { money, number } from './model'
import type { Report } from './types'
import { RunStatus } from './RunList'
import styles from './SrBench.module.css'

export default function RunLineage({ report }: { report: Report | null }) {
  const [page, setPage] = useState(0)
  const recovery = report?.recovery as
    | {
        parent_run_id?: string
        mode?: string
        parent_snapshot?: {
          progress?: { completed?: number; total?: number }
          known_spend_usd?: number
          spend_complete?: boolean
        }
      }
    | undefined
  const children = report?.child_attempts as
    | Array<{
        id: string
        status: string
        progress: { completed: number; total: number; failed: number }
      }>
    | undefined
  const current = Math.min(page, Math.max(0, Math.ceil((children?.length ?? 0) / 10) - 1))
  if (!recovery && !children?.length) return null
  return (
    <aside className={styles.notice}>
      <h3>Recovery lineage</h3>
      {recovery && (
        <>
          <p>
            This run covers only the explicitly selected recovery cells. Its denominator and spend
            exclude the original attempt.
          </p>
          <p>
            <a href={`?view=runs&run=${encodeURIComponent(recovery.parent_run_id ?? '')}`}>
              Open parent run
            </a>{' '}
            ·{' '}
            {recovery.mode === 'undispatched'
              ? 'Continued undispatched work'
              : 'New attempts for known failed cases'}
          </p>
          <p>
            Parent snapshot: {number(recovery.parent_snapshot?.progress?.completed)} /{' '}
            {number(recovery.parent_snapshot?.progress?.total)} completed · known spend{' '}
            {money(recovery.parent_snapshot?.known_spend_usd)}
            {recovery.parent_snapshot?.spend_complete ? '' : ' (incomplete accounting)'}. Parent
            spend is retained separately and is not a total for this run.
          </p>
        </>
      )}
      {!!children?.length && (
        <div className={styles.tableScroll}>
          <table>
            <thead>
              <tr>
                <th>Child attempt</th>
                <th>Status</th>
                <th>Completed / own scope</th>
                <th>Failures</th>
              </tr>
            </thead>
            <tbody>
              {children.slice(current * 10, current * 10 + 10).map((child) => (
                <tr key={child.id}>
                  <td>
                    <a href={`?view=runs&run=${encodeURIComponent(child.id)}`}>{child.id}</a>
                  </td>
                  <td>
                    <RunStatus status={child.status} />
                  </td>
                  <td>
                    {number(child.progress.completed)} / {number(child.progress.total)}
                  </td>
                  <td>{number(child.progress.failed)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <BenchPagination
            label="Recovery attempts"
            total={children.length}
            page={current}
            pageSize={10}
            onChange={setPage}
          />
        </div>
      )}
    </aside>
  )
}
