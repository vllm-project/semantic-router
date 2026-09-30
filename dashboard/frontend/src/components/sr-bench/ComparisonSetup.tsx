import { useState } from 'react'
import ProductIcon from '../ProductIcon'
import ProductLoadingState from '../ProductLoadingState'
import BenchSelect from './BenchSelect'
import BenchPagination from './BenchPagination'
import RunOptionsStatus from './RunOptionsStatus'
import type useExperimentComparison from './useExperimentComparison'
import { comparisonSelectionReady } from './experimentComparison'
import { number } from './model'
import { profileTitle } from './datasetPresentation'
import styles from './SrBench.module.css'
import controls from './BenchControls.module.css'

export default function ComparisonSetup({
  baseline,
  candidates,
  pending,
  options,
  onBaseline,
  onCandidates,
  onCompare,
}: {
  baseline: string
  candidates: string[]
  pending: boolean
  options: ReturnType<typeof useExperimentComparison>
  onBaseline: (id: string) => void
  onCandidates: (ids: string[]) => void
  onCompare: () => void
}) {
  const { baselines, choices } = options
  const [query, setQuery] = useState('')
  const [page, setPage] = useState(0)
  const available = choices.items.filter((item) =>
    `${item.name} ${item.run_id}`.toLowerCase().includes(query.toLowerCase()),
  )
  const current = Math.min(page, Math.max(0, Math.ceil(available.length / 8) - 1))
  const allSelected =
    available.length > 0 && available.every((item) => candidates.includes(item.run_id))
  const invalid = candidates.some((id) => !choices.items.some((item) => item.run_id === id))
  const ready =
    comparisonSelectionReady(baseline, candidates, baselines.items, choices.items) &&
    !choices.loading &&
    !choices.error &&
    !baselines.error &&
    !baselines.loading &&
    !options.membershipLoading &&
    !options.membershipError
  const baselineItems = baselines.items
  const noCombinations =
    baselines.loaded &&
    !baselines.loading &&
    !baselines.error &&
    !choices.error &&
    !choices.loading &&
    !options.membershipLoading &&
    !options.membershipError &&
    !baselines.next &&
    !baselineItems.length
  if (noCombinations)
    return (
      <p className={styles.optionEmpty} role="status">
        {baselines.scanLimited
          ? 'No verified comparisons available. Some saved evidence exceeded the verification limit.'
          : 'No comparable results yet. Finish reference and recipe runs on the same questions and settings.'}
      </p>
    )
  return (
    <div className={styles.comparisonSetup}>
      {options.membershipLoading && (
        <ProductLoadingState compact label="Finding comparable runs…" />
      )}
      {options.membershipError && (
        <div role="alert" className={styles.error}>
          <p>{options.membershipError}</p>
          <button onClick={options.reloadMembership}>Reload experiment members</button>
        </div>
      )}
      <RunOptionsStatus
        {...baselines}
        label="references"
        onRetry={baselines.reload}
        onMore={baselines.loadMore}
      />
      {!!baselineItems.length && (
        <BenchSelect
          className={styles.baselineSelect}
          label="Reference run"
          value={baseline}
          disabled={pending || baselines.loading || !!baselines.error}
          searchable
          placeholder="Choose reference results"
          options={baselineItems.map((item) => ({
            value: item.run_id,
            label: item.name,
            description: `${profileTitle(item.profile)} · ${number(item.case_count)} cases`,
          }))}
          onChange={(id) => {
            onBaseline(id)
            onCandidates([])
            setQuery('')
            setPage(0)
          }}
        />
      )}
      {baseline && (
        <>
          {!options.membershipLoading && !options.membershipError && !options.baselineInScope && (
            <p className={styles.notice}>
              The selected reference is outside this experiment. Choose reference results in this
              experiment, or explicitly include runs outside it.
            </p>
          )}
          <RunOptionsStatus
            {...choices}
            label="comparison runs"
            onRetry={choices.reload}
            onMore={choices.loadMore}
          />
          {choices.loaded &&
          !choices.loading &&
          !choices.error &&
          !choices.items.length &&
          !choices.next ? (
            <p className={styles.optionEmpty} role="status">
              {choices.scanLimited
                ? 'No verified results for this reference. Some evidence exceeded the verification limit.'
                : 'This reference no longer has comparable results. Choose another reference.'}
            </p>
          ) : (
            choices.loaded &&
            !choices.error &&
            !!choices.items.length && (
              <>
                <div className={styles.optionToolbar}>
                  <label className={styles.inlineLabel}>
                    Find comparison runs
                    <input
                      type="search"
                      value={query}
                      placeholder="Search loaded runs"
                      onChange={(event) => {
                        setQuery(event.target.value)
                        setPage(0)
                      }}
                    />
                  </label>
                  <div className={styles.actions}>
                    <span className={styles.muted}>{candidates.length} selected</span>
                    <button
                      className={controls.compactButton}
                      disabled={pending || choices.loading || !available.length}
                      onClick={() =>
                        onCandidates(
                          allSelected
                            ? candidates.filter(
                                (id) => !available.some((item) => item.run_id === id),
                              )
                            : [
                                ...new Set([
                                  ...candidates,
                                  ...available.map((item) => item.run_id),
                                ]),
                              ],
                        )
                      }
                    >
                      <ProductIcon name={allSelected ? 'close' : 'check'} />
                      {allSelected ? 'Clear selection' : 'Select all'}
                    </button>
                  </div>
                </div>
                <div className={styles.comparisonChoices} role="group" aria-label="Comparison runs">
                  {available.slice(current * 8, current * 8 + 8).map((item) => (
                    <label key={item.run_id} className={styles.runChoice}>
                      <input
                        type="checkbox"
                        checked={candidates.includes(item.run_id)}
                        disabled={pending || choices.loading}
                        onChange={(event) =>
                          onCandidates(
                            event.target.checked
                              ? [...candidates, item.run_id]
                              : candidates.filter((id) => id !== item.run_id),
                          )
                        }
                      />
                      <span>
                        <strong>{item.name}</strong>
                        <small>
                          {profileTitle(item.profile)} · {number(item.case_count)} cases
                        </small>
                      </span>
                    </label>
                  ))}
                </div>
                {!available.length && (
                  <p className={styles.muted}>No loaded runs match this search.</p>
                )}
                <BenchPagination
                  label="Comparison runs"
                  total={available.length}
                  page={current}
                  pageSize={8}
                  onChange={setPage}
                />
                {choices.next && (
                  <p className={styles.muted}>
                    Search and Select all apply to loaded results. Load more to see additional
                    compatible runs.
                  </p>
                )}
                {invalid && (
                  <p className={styles.notice}>
                    Some saved selections are not in the available results.{' '}
                    {choices.next
                      ? 'Load more to review them, or clear the selection.'
                      : 'Clear them before choosing another comparison.'}{' '}
                    <button
                      onClick={() =>
                        onCandidates(
                          candidates.filter((id) =>
                            choices.items.some((item) => item.run_id === id),
                          ),
                        )
                      }
                    >
                      Clear unavailable selections
                    </button>
                  </p>
                )}
                <div className={styles.actions}>
                  <button
                    className={styles.primary}
                    disabled={pending || !ready}
                    onClick={onCompare}
                  >
                    <ProductIcon name="chart" />
                    {pending ? 'Comparing…' : 'Compare runs'}
                  </button>
                  <span className={styles.muted}>
                    The strongest saved single model is the reference.
                  </span>
                </div>
              </>
            )
          )}
        </>
      )}
    </div>
  )
}
