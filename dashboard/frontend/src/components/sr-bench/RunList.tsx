import BenchSelect from './BenchSelect'
import BenchPagination from './BenchPagination'
import { active, number, percent } from './model'
import type { Run } from './types'
import { profileTitle } from './datasetPresentation'
import { targetLabel } from './targetPresentation'
import styles from './SrBench.module.css'

export function RunStatus({ status }: { status: string }) {
  const tone = status === 'completed' ? 'success' : active(status) ? 'active' : 'attention'
  return <span className={`${styles.status} ${styles[tone]}`}>{status}</span>
}

export interface RunFilters {
  query: string
  status: string
  mode: string
  profile: string
  page: number
}

export default function RunList({
  runs,
  selectedID,
  onSelect,
  filters,
  onFilters,
}: {
  runs: Run[]
  selectedID: string | null
  onSelect: (id: string) => void
  filters: RunFilters
  onFilters: (filters: Partial<RunFilters>) => void
}) {
  const { query, status, mode, profile, page } = filters
  const changeFilter = (patch: Partial<RunFilters>) => onFilters({ ...patch, page: 0 })
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
      (mode === 'all' || run.manifest.mode === mode) &&
      (profile === 'all' || run.manifest.profile === profile)
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
              changeFilter({ query: event.target.value })
            }}
          />
        </label>
        <BenchSelect
          label="Run status"
          value={status}
          onChange={(value) => {
            changeFilter({ status: value })
          }}
          options={[
            { value: 'all', label: 'All statuses' },
            { value: 'active', label: 'Active' },
            { value: 'completed', label: 'Completed' },
            { value: 'failed', label: 'Failed' },
            { value: 'interrupted', label: 'Interrupted' },
            { value: 'cancelled', label: 'Cancelled' },
          ]}
        />
        <BenchSelect
          label="Run mode"
          value={mode}
          onChange={(value) => {
            changeFilter({ mode: value })
          }}
          options={[
            { value: 'all', label: 'All modes' },
            { value: 'live', label: 'Live evaluation' },
            { value: 'preview', label: 'Route preview' },
            { value: 'replay', label: 'Diagnostic replay' },
          ]}
        />
        <BenchSelect
          label="Run profile"
          value={profile}
          onChange={(value) => changeFilter({ profile: value })}
          options={[
            { value: 'all', label: 'All profiles' },
            ...['smoke', 'quick', 'standard'].map((value) => ({
              value,
              label: profileTitle(value),
            })),
          ]}
        />
      </div>
      {filtered.length ? (
        <div className={styles.tableScroll}>
          <table>
            <thead>
              <tr>
                <th>Run / targets</th>
                <th>Mode</th>
                <th>Profile</th>
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
                            {targetLabel(target)} · {target.kind === 'mom' ? 'MoM' : 'Single'}
                          </span>
                        ))}
                      </div>
                    </td>
                    <td>
                      {run.manifest.mode === 'live'
                        ? 'Live'
                        : run.manifest.mode === 'preview'
                          ? 'Preview'
                          : 'Replay'}
                    </td>
                    <td>{profileTitle(run.manifest.profile)}</td>
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
      <BenchPagination
        label="Runs"
        total={filtered.length}
        page={currentPage}
        pageSize={10}
        onChange={(page) => onFilters({ page })}
      />
    </section>
  )
}
