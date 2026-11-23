import { useState } from 'react'
import { ChartLine, Check, Search, X } from 'lucide-react'
import { Button, Checkbox, InputGroup } from '../../ui'
import BenchSelect from './BenchSelect'
import BenchPagination from './BenchPagination'
import RunOptionsStatus from './RunOptionsStatus'
import type useExperimentComparison from './useExperimentComparison'
import { comparisonSelectionReady } from './experimentComparison'
import { number } from './model'
import { profileTitle } from './datasetPresentation'
import { ActionRow, BenchLoading, EmptyNote, Notice } from './benchUi'
import { benchText, choiceCardClass } from './benchStyles'

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
      <EmptyNote role="status" icon={ChartLine} className="bg-card">
        {baselines.scanLimited
          ? 'No verified comparisons available. Some saved evidence exceeded the verification limit.'
          : 'No comparable results yet. Finish reference and recipe runs on the same questions and settings.'}
      </EmptyNote>
    )
  return (
    <div className="my-4 min-w-0">
      {options.membershipLoading && <BenchLoading label="Finding comparable runs…" />}
      {options.membershipError && (
        <Notice tone="danger" role="alert">
          <p>{options.membershipError}</p>
          <div>
            <Button variant="outline" size="xs" onClick={options.reloadMembership}>
              Reload experiment members
            </Button>
          </div>
        </Notice>
      )}
      <RunOptionsStatus
        {...baselines}
        label="references"
        onRetry={baselines.reload}
        onMore={baselines.loadMore}
      />
      {!!baselineItems.length && (
        <div className="max-w-sm">
          <BenchSelect
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
        </div>
      )}
      {baseline && (
        <>
          {!options.membershipLoading && !options.membershipError && !options.baselineInScope && (
            <Notice tone="warning" as="p">
              The selected reference is outside this experiment. Choose reference results in this
              experiment, or explicitly include runs outside it.
            </Notice>
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
            <EmptyNote role="status">
              {choices.scanLimited
                ? 'No verified results for this reference. Some evidence exceeded the verification limit.'
                : 'This reference no longer has comparable results. Choose another reference.'}
            </EmptyNote>
          ) : (
            choices.loaded &&
            !choices.error &&
            !!choices.items.length && (
              <>
                <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
                  <InputGroup
                    type="search"
                    aria-label="Find comparison runs"
                    value={query}
                    placeholder="Search loaded runs"
                    leading={<Search aria-hidden="true" />}
                    containerClassName="w-full sm:w-72"
                    className="h-8"
                    onChange={(event) => {
                      setQuery(event.target.value)
                      setPage(0)
                    }}
                  />
                  <ActionRow>
                    <span className="text-xs text-muted-foreground tabular-nums">
                      {candidates.length} selected
                    </span>
                    <Button
                      variant="outline"
                      size="xs"
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
                      {allSelected ? <X aria-hidden="true" /> : <Check aria-hidden="true" />}
                      {allSelected ? 'Clear selection' : 'Select all'}
                    </Button>
                  </ActionRow>
                </div>
                <div
                  className="my-3 grid min-w-0 grid-cols-1 gap-2 sm:grid-cols-2"
                  role="group"
                  aria-label="Comparison runs"
                >
                  {available.slice(current * 8, current * 8 + 8).map((item) => (
                    <label key={item.run_id} className={choiceCardClass}>
                      <Checkbox
                        className="mt-0.5"
                        checked={candidates.includes(item.run_id)}
                        disabled={pending || choices.loading}
                        onCheckedChange={(checked) =>
                          onCandidates(
                            checked === true
                              ? [...candidates, item.run_id]
                              : candidates.filter((id) => id !== item.run_id),
                          )
                        }
                      />
                      <span className="min-w-0 break-words">
                        <strong className="block font-medium text-foreground">{item.name}</strong>
                        <small className="block text-xs text-muted-foreground">
                          {profileTitle(item.profile)} · {number(item.case_count)} cases
                        </small>
                      </span>
                    </label>
                  ))}
                </div>
                {!available.length && (
                  <p className={benchText.muted}>No loaded runs match this search.</p>
                )}
                <BenchPagination
                  label="Comparison runs"
                  total={available.length}
                  page={current}
                  pageSize={8}
                  onChange={setPage}
                />
                {choices.next && (
                  <p className={benchText.muted}>
                    Search and Select all apply to loaded results. Load more to see additional
                    compatible runs.
                  </p>
                )}
                {invalid && (
                  <Notice tone="warning" as="p">
                    Some saved selections are not in the available results.{' '}
                    {choices.next
                      ? 'Load more to review them, or clear the selection.'
                      : 'Clear them before choosing another comparison.'}{' '}
                    <Button
                      variant="link"
                      size="xs"
                      className="h-auto"
                      onClick={() =>
                        onCandidates(
                          candidates.filter((id) =>
                            choices.items.some((item) => item.run_id === id),
                          ),
                        )
                      }
                    >
                      Clear unavailable selections
                    </Button>
                  </Notice>
                )}
                <ActionRow className="mt-3">
                  <Button size="sm" disabled={pending || !ready} onClick={onCompare}>
                    <ChartLine aria-hidden="true" />
                    {pending ? 'Comparing…' : 'Compare runs'}
                  </Button>
                  <span className={benchText.muted}>
                    The strongest saved single model is the reference.
                  </span>
                </ActionRow>
              </>
            )
          )}
        </>
      )}
    </div>
  )
}
