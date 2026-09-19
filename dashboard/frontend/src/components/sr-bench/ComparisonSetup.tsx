import { useState } from 'react'
import ProductIcon from '../ProductIcon'
import BenchSelect from './BenchSelect'
import BenchPagination from './BenchPagination'
import { comparisonEligibility } from './comparisonEligibility'
import { number } from './model'
import { profileTitle } from './datasetPresentation'
import type { Run } from './types'
import styles from './SrBench.module.css'
import controls from './BenchControls.module.css'

export default function ComparisonSetup({
  runs,
  baseline,
  candidates,
  pending,
  onBaseline,
  onCandidates,
}: {
  runs: Run[]
  baseline: string
  candidates: string[]
  pending: boolean
  onBaseline: (id: string) => void
  onCandidates: (ids: string[]) => void
}) {
  const [query, setQuery] = useState('')
  const [showUnavailable, setShowUnavailable] = useState(false)
  const [page, setPage] = useState(0)
  const selected = runs.find((run) => run.id === baseline)
  const baselines = runs.filter(
    (run) =>
      run.status === 'completed' &&
      run.manifest.mode === 'live' &&
      run.manifest.targets.some((target) => target.kind === 'single'),
  )
  const choices = runs
    .filter((run) => run.id !== baseline)
    .map((run) => ({ run, ...comparisonEligibility(selected, run) }))
  const matching = choices.filter(({ run }) =>
    `${run.manifest.name} ${run.id} ${run.manifest.targets.map((target) => target.id).join(' ')}`
      .toLowerCase()
      .includes(query.toLowerCase()),
  )
  const available = matching.filter((item) => item.eligible)
  const visible = (showUnavailable ? matching : available).sort(
    (a, b) => Number(b.eligible) - Number(a.eligible),
  )
  const current = Math.min(page, Math.max(0, Math.ceil(visible.length / 8) - 1))
  const allSelected =
    available.length > 0 && available.every(({ run }) => candidates.includes(run.id))
  const selectedInvalid = candidates.filter(
    (id) => !choices.some((item) => item.run.id === id && item.eligible),
  )
  return (
    <div className={styles.comparisonSetup}>
      <section className={styles.comparisonStep} aria-labelledby="comparison-baseline-step">
        <div className={styles.stepHeading}>
          <span>1</span>
          <div>
            <h3 id="comparison-baseline-step">Choose a baseline</h3>
            <p>The service selects the strongest single model in this completed run.</p>
          </div>
        </div>
        <BenchSelect
          className={styles.baselineSelect}
          label="Baseline run"
          value={baseline}
          disabled={pending}
          searchable
          placeholder="Select single-model baseline"
          options={baselines.map((run) => ({
            value: run.id,
            label: run.manifest.name,
            description: `${run.manifest.targets
              .filter((target) => target.kind === 'single')
              .map((target) => target.id)
              .join(
                ', ',
              )} · ${profileTitle(run.manifest.profile)} · ${number(run.progress.total)} planned results`,
          }))}
          onChange={(value) => {
            onBaseline(value)
            onCandidates([])
            setPage(0)
          }}
        />
        {!baselines.length && (
          <p className={styles.notice}>
            Complete a live run containing a single model to create a baseline.
          </p>
        )}
        {selected && (
          <p className={styles.muted}>
            {profileTitle(selected.manifest.profile)} · {number(selected.progress.total)} planned
            results · Created {new Date(selected.created_at).toLocaleDateString()}. Changing the
            baseline clears the candidate selection.
          </p>
        )}
      </section>
      <section className={styles.comparisonStep} aria-labelledby="comparison-candidates-step">
        <div className={styles.stepHeading}>
          <span>2</span>
          <div>
            <h3 id="comparison-candidates-step">Choose runs to compare</h3>
            <p>Compare one or more single-model or MoM runs on the same frozen protocol.</p>
          </div>
        </div>
        {!selected ? (
          <p className={styles.emptyState}>Choose a baseline above to review available runs.</p>
        ) : (
          <>
            <div className={styles.sectionHeading}>
              <label className={styles.inlineLabel}>
                Find comparison runs
                <input
                  type="search"
                  value={query}
                  placeholder="Run, model or ID"
                  onChange={(event) => {
                    setQuery(event.target.value)
                    setPage(0)
                  }}
                />
              </label>
              <div className={styles.actions}>
                <span className={styles.badge}>{candidates.length} selected</span>
                <button
                  className={controls.compactButton}
                  disabled={pending || !available.length}
                  onClick={() =>
                    onCandidates(
                      allSelected
                        ? candidates.filter((id) => !available.some(({ run }) => run.id === id))
                        : [...new Set([...candidates, ...available.map(({ run }) => run.id)])],
                    )
                  }
                >
                  <ProductIcon name={allSelected ? 'close' : 'check'} />
                  {allSelected ? 'Clear selection' : 'Select all'}
                </button>
              </div>
            </div>
            <p className={styles.muted}>
              {choices.filter((item) => item.eligible).length} available for review ·{' '}
              {choices.filter((item) => !item.eligible).length} unavailable. Select all applies to
              available runs matching your search, across pages.
            </p>
            <label className={styles.toggleLabel}>
              <input
                type="checkbox"
                checked={showUnavailable}
                onChange={(event) => {
                  setShowUnavailable(event.target.checked)
                  setPage(0)
                }}
              />
              Show unavailable runs and reasons
            </label>
            <div className={styles.comparisonChoices} role="group" aria-label="Comparison runs">
              {visible.slice(current * 8, current * 8 + 8).map(({ run, eligible, reason }) => (
                <label
                  key={run.id}
                  className={`${styles.runChoice} ${eligible ? '' : styles.unavailableChoice}`}
                >
                  <input
                    type="checkbox"
                    disabled={pending || !eligible}
                    checked={candidates.includes(run.id)}
                    onChange={(event) =>
                      onCandidates(
                        event.target.checked
                          ? [...candidates, run.id]
                          : candidates.filter((id) => id !== run.id),
                      )
                    }
                  />
                  <span>
                    <strong>{run.manifest.name}</strong>
                    <small>
                      {run.manifest.targets
                        .map(
                          (target) => `${target.id} (${target.kind === 'mom' ? 'MoM' : 'Single'})`,
                        )
                        .join(', ')}{' '}
                      · {number(run.progress.total)} planned results ·{' '}
                      {profileTitle(run.manifest.profile)}
                    </small>
                    <small>{eligible ? 'Available' : reason}</small>
                  </span>
                </label>
              ))}
            </div>
            {!visible.length && (
              <p className={styles.emptyState}>
                {query
                  ? 'No runs match this search.'
                  : 'No runs are available for review yet. Show unavailable runs to inspect their reasons.'}
              </p>
            )}
            <BenchPagination
              label="Comparison runs"
              total={visible.length}
              page={current}
              pageSize={8}
              onChange={setPage}
            />
            {!!selectedInvalid.length && (
              <p className={styles.error} role="alert">
                Some saved selections are no longer available.{' '}
                <button
                  onClick={() =>
                    onCandidates(candidates.filter((id) => !selectedInvalid.includes(id)))
                  }
                >
                  Remove unavailable selections
                </button>
              </p>
            )}
            <p className={styles.muted}>
              The service verifies complete paired results, judges and frozen pricing before showing
              scores. Runs are ordered by creation time in the comparison.
            </p>
          </>
        )}
      </section>
    </div>
  )
}
