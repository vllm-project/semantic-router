import { useState } from 'react'
import { active, number, percent } from './model'
import type { Run } from './types'
import { profileTitle } from './datasetPresentation'
import styles from './SrBench.module.css'

export function RunStatus({ status }: { status: string }) {
  const tone = status === 'completed' ? 'success' : active(status) ? 'active' : 'attention'
  return <span className={`${styles.status} ${styles[tone]}`}>{status}</span>
}

export default function RunList({
  runs,
  selectedID,
  onSelect,
}: {
  runs: Run[]
  selectedID: string | null
  onSelect: (id: string) => void
}) {
  const [query, setQuery] = useState('')
  const [status, setStatus] = useState('all')
  const [mode, setMode] = useState('all')
  const [page, setPage] = useState(0)
  const filtered = runs.filter((run) => {
    const text = [
      run.id,
      run.manifest.name,
      run.manifest.profile,
      ...run.manifest.targets.map((target) => `${target.id} ${target.kind} ${target.model}`),
    ]
      .join(' ')
      .toLowerCase()
    return (
      text.includes(query.toLowerCase()) &&
      (status === 'all' || (status === 'active' ? active(run.status) : run.status === status)) &&
      (mode === 'all' || run.manifest.mode === mode)
    )
  })
  const currentPage = Math.min(page, Math.max(0, Math.ceil(filtered.length / 10) - 1))
  return (
    <section className={styles.panel} aria-labelledby="run-list-title">
      <div className={styles.sectionHeading}>
        <div>
          <h2 id="run-list-title">Evaluation runs</h2>
          <p>Runs and evidence stay with the service when you close this page.</p>
        </div>
        <span className={styles.badge}>
          {runs.filter((run) => active(run.status)).length} active
        </span>
      </div>
      <div className={styles.runFilters}>
        <label>
          Search runs
          <input
            type="search"
            value={query}
            placeholder="Name, model or run ID"
            onChange={(event) => {
              setQuery(event.target.value)
              setPage(0)
            }}
          />
        </label>
        <label>
          Run status
          <select
            value={status}
            onChange={(event) => {
              setStatus(event.target.value)
              setPage(0)
            }}
          >
            <option value="all">All statuses</option>
            <option value="active">Active</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
            <option value="interrupted">Interrupted</option>
            <option value="cancelled">Cancelled</option>
          </select>
        </label>
        <label>
          Run mode
          <select
            value={mode}
            onChange={(event) => {
              setMode(event.target.value)
              setPage(0)
            }}
          >
            <option value="all">All modes</option>
            <option value="live">Live evaluation</option>
            <option value="preview">Route preview</option>
            <option value="replay">Diagnostic replay</option>
          </select>
        </label>
      </div>
      {filtered.length ? (
        <div className={styles.tableScroll}>
          <table>
            <thead>
              <tr>
                <th>Run / targets</th>
                <th>Mode / profile</th>
                <th>Status</th>
                <th>Progress</th>
                <th>Last update</th>
              </tr>
            </thead>
            <tbody>
              {filtered.slice(currentPage * 10, currentPage * 10 + 10).map((run) => {
                const finished = run.progress.completed + run.progress.failed
                return (
                  <tr key={run.id} aria-selected={run.id === selectedID}>
                    <td>
                      <button className={styles.linkButton} onClick={() => onSelect(run.id)}>
                        {run.manifest.name}
                      </button>

                      <div className={styles.targetChips}>
                        {run.manifest.targets.map((target) => (
                          <span className={styles.badge} key={target.id}>
                            {target.id} · {target.kind === 'mom' ? 'MoM' : 'Single'}
                          </span>
                        ))}
                      </div>
                    </td>
                    <td>
                      {run.manifest.mode === 'live'
                        ? 'Live'
                        : run.manifest.mode === 'preview'
                          ? 'Preview'
                          : 'Replay'}{' '}
                      / {profileTitle(run.manifest.profile)}
                    </td>
                    <td>
                      <RunStatus status={run.status} />
                    </td>
                    <td>
                      <strong>
                        {number(run.progress.completed)} / {number(run.progress.total)}
                      </strong>{' '}
                      complete
                      <progress
                        className={styles.progress}
                        max={Math.max(1, run.progress.total)}
                        value={finished}
                        aria-label={`${run.manifest.name} progress`}
                      />
                      <small>
                        {number(run.progress.failed)} failed ·{' '}
                        {percent(run.progress.total ? finished / run.progress.total : 0)} processed
                      </small>
                    </td>
                    <td>
                      <time dateTime={run.updated_at}>
                        {new Date(run.updated_at).toLocaleString()}
                      </time>
                      <small>Created {new Date(run.created_at).toLocaleString()}</small>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <p className={styles.emptyState}>
          {runs.length
            ? 'No runs match these filters.'
            : 'No runs yet. Create an evaluation using a prepared dataset.'}
        </p>
      )}
      {filtered.length > 10 && (
        <div className={styles.actions}>
          <button disabled={currentPage === 0} onClick={() => setPage(currentPage - 1)}>
            Previous runs
          </button>
          <span>
            {number(filtered.length)} runs · page {currentPage + 1} of{' '}
            {Math.ceil(filtered.length / 10)}
          </span>
          <button
            disabled={(currentPage + 1) * 10 >= filtered.length}
            onClick={() => setPage(currentPage + 1)}
          >
            Next runs
          </button>
        </div>
      )}
    </section>
  )
}
