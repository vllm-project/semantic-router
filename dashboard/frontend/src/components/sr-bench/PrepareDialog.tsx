import { useId, useRef, useState } from 'react'
import { ArrowUpRight, Download, RefreshCw, Search } from 'lucide-react'
import {
  Button,
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  InputGroup,
  RadioGroup,
  RadioGroupItem,
  SectionLabel,
  StatusDot,
  cn,
  type StatusTone,
} from '../../ui'
import { number } from './model'
import { benchmarkTitle, profileTitle } from './datasetPresentation'
import {
  preparationBenchmarkIDs,
  preparationIsActive,
  preparationPhase,
  type DatasetPreparationJob,
  type PreparationProfile,
} from './datasetPreparationApi'
import { retryPreparation, type useDatasetPreparations } from './useDatasetPreparations'
import type { Dataset } from './types'
import { BenchLoading, Disclosure, Notice } from './benchUi'
import { benchText } from './benchStyles'
import {
  BENCHMARK_SEARCH_THRESHOLD,
  PREPARATION_PROFILES,
  filterPreparationBenchmarks,
  largestPreparationSize,
} from './prepareDialogSupport'

const jobTone = (job: DatasetPreparationJob): StatusTone =>
  job.status === 'completed'
    ? 'success'
    : job.status === 'failed'
      ? 'danger'
      : preparationIsActive(job)
        ? 'info'
        : 'neutral'

export type PreparationState = ReturnType<typeof useDatasetPreparations>

export default function PrepareDialog({
  open,
  state,
  disabledReason,
  accessRefreshing,
  onRefreshAccess,
  onOpenDataset,
  onClose,
}: {
  open: boolean
  state: PreparationState
  disabledReason: string | null
  accessRefreshing: boolean
  onRefreshAccess: () => void
  onOpenDataset: (dataset: Dataset) => void
  onClose: () => void
}) {
  const id = useId()
  const content = useRef<HTMLDivElement>(null)
  const [benchmarkID, setBenchmarkID] = useState('')
  const [profile, setProfile] = useState<PreparationProfile>('smoke')
  const [query, setQuery] = useState('')
  const benchmark = state.benchmarks.find((item) => item.id === benchmarkID) ?? state.benchmarks[0]
  const listed = filterPreparationBenchmarks(state.benchmarks, query)
  const active = state.preparations.filter(preparationIsActive)
  const pending = state.preparations.filter((job) => job.status !== 'completed')
  const completed = state.preparations.filter((job) => job.status === 'completed')
  const duplicate = active.some(
    (job) => preparationBenchmarkIDs(job).includes(benchmark?.id ?? '') && job.profile === profile,
  )
  const selectionDisabled = state.submitting || state.loading
  const mutationDisabled = !!disabledReason || selectionDisabled
  const startDisabled = mutationDisabled || active.length > 0

  const start = () => {
    if (benchmark && !startDisabled) void state.prepare({ benchmark: benchmark.id, profile })
  }

  const jobs = (items: DatasetPreparationJob[], label: string) => (
    <ul
      className="m-0 min-w-0 list-none divide-y divide-border rounded-lg border border-border bg-card p-0"
      aria-label={label}
    >
      {items.map((job) => (
        <li key={job.id} className="min-w-0 px-3.5 py-3" data-status={job.status}>
          <div className="flex flex-wrap items-start justify-between gap-x-4 gap-y-2">
            <div className="flex min-w-0 flex-col gap-0.5">
              <strong className="break-words text-[13px] font-medium text-foreground">
                {preparationBenchmarkIDs(job)
                  .map(
                    (id) =>
                      state.benchmarks.find((item) => item.id === id)?.name ?? benchmarkTitle(id),
                  )
                  .join(', ')}
              </strong>
              <span className="text-xs text-muted-foreground">
                {profileTitle(job.profile)}
                {job.status === 'completed' &&
                  job.dataset &&
                  ` · ${number(job.dataset.case_count)} questions ready`}
              </span>
            </div>
            <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
              <span
                className="inline-flex items-center gap-1.5 text-xs text-foreground/85"
                role="status"
              >
                <StatusDot tone={jobTone(job)} pulse={preparationIsActive(job)} />
                {preparationPhase(job)}
              </span>
              <time
                dateTime={job.updated_at}
                title={new Date(job.updated_at).toLocaleString()}
                className="text-[11px] text-subtle-foreground tabular-nums"
              >
                {new Date(job.updated_at).toLocaleDateString()}
              </time>
              {job.status === 'completed' && job.dataset && (
                <Button
                  type="button"
                  variant="outline"
                  size="xs"
                  onClick={() => onOpenDataset(job.dataset!)}
                >
                  View prepared dataset
                </Button>
              )}
              {job.status === 'failed' && (
                <Button
                  type="button"
                  variant="outline"
                  size="xs"
                  disabled={mutationDisabled || active.length > 0}
                  onClick={() => {
                    if (!mutationDisabled && !active.length) retryPreparation(state, job)
                  }}
                >
                  Retry preparation
                </Button>
              )}
            </div>
          </div>
          {job.error && (
            <p className="mt-2 break-words text-xs text-destructive-foreground">{job.error}</p>
          )}
        </li>
      ))}
    </ul>
  )

  const hasNotice = Boolean(state.readError || disabledReason || state.submitError)
  return (
    <Dialog open={open} onOpenChange={(value) => !value && onClose()}>
      <DialogContent
        ref={content}
        size="xl"
        className="h-[min(680px,calc(100dvh-2rem))]"
        onOpenAutoFocus={(event) => {
          // The selected benchmark takes focus so the arrow keys browse the list at once.
          const selected = content.current?.querySelector<HTMLElement>(
            '[role="radio"][data-state="checked"]:not(:disabled)',
          )
          if (!selected) return
          event.preventDefault()
          selected.focus()
        }}
      >
        <DialogHeader className="pb-4 pr-12">
          <DialogTitle>Prepare a dataset</DialogTitle>
          <DialogDescription>
            New evaluations prepare missing datasets automatically. Download one ahead of time here;
            no model answers are generated.
          </DialogDescription>
        </DialogHeader>
        {hasNotice ? (
          <div className="flex shrink-0 flex-col gap-2 px-6 pb-4">
            {state.readError && (
              <Notice tone="danger" role="alert" className="my-0">
                <p>{state.readError}</p>
                <div>
                  <Button type="button" variant="outline" size="xs" onClick={state.refresh}>
                    Refresh preparation status
                  </Button>
                </div>
              </Notice>
            )}
            {disabledReason && (
              <Notice tone="warning" className="my-0">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <p className="min-w-0 flex-1">{disabledReason}</p>
                  <Button
                    type="button"
                    variant="outline"
                    size="xs"
                    onClick={onRefreshAccess}
                    disabled={accessRefreshing}
                  >
                    <RefreshCw aria-hidden="true" /> Refresh access
                  </Button>
                </div>
              </Notice>
            )}
            {state.submitError && (
              <Notice tone="danger" as="p" role="alert" className="my-0">
                {state.submitError}
              </Notice>
            )}
          </div>
        ) : null}
        <div
          className={cn(
            'grid min-h-0 flex-1 grid-cols-1 content-start overflow-y-auto border-t border-border',
            benchmark && 'md:grid-cols-[17rem_minmax(0,1fr)] md:grid-rows-1 md:overflow-hidden',
          )}
        >
          {benchmark ? (
            <div className="flex min-h-0 min-w-0 flex-col border-b border-border bg-surface-subtle/40 md:border-b-0 md:border-r">
              <div className="flex shrink-0 items-center justify-between gap-2 px-4 pb-2 pt-4">
                <SectionLabel>Benchmarks</SectionLabel>
                <span className="text-[11px] text-subtle-foreground tabular-nums">
                  {number(state.benchmarks.length)}
                </span>
              </div>
              {state.benchmarks.length > BENCHMARK_SEARCH_THRESHOLD ? (
                <div className="shrink-0 px-3 pb-2">
                  <InputGroup
                    type="search"
                    value={query}
                    onChange={(event) => setQuery(event.target.value)}
                    placeholder="Search benchmarks"
                    aria-label="Search benchmarks"
                    leading={<Search />}
                    className="h-8"
                  />
                </div>
              ) : null}
              <RadioGroup
                aria-label="Benchmark to download"
                value={benchmark.id}
                onValueChange={setBenchmarkID}
                disabled={selectionDisabled}
                className="min-h-0 content-start gap-0.5 overflow-y-auto px-2 pb-3 pt-1 max-md:max-h-64 md:flex-1 md:overscroll-contain"
              >
                {listed.map((item, index) => {
                  const checked = item.id === benchmark.id
                  const itemId = `${id}-benchmark-${index}`
                  return (
                    <label
                      key={item.id}
                      htmlFor={itemId}
                      className={cn(
                        'flex min-w-0 cursor-pointer items-center gap-3 rounded-md px-2.5 py-2 transition-colors has-[:focus-visible]:ring-[3px] has-[:focus-visible]:ring-ring/40 has-[:disabled]:cursor-not-allowed',
                        checked ? 'bg-accent' : 'hover:bg-accent/60',
                      )}
                    >
                      <RadioGroupItem
                        id={itemId}
                        value={item.id}
                        aria-labelledby={`${itemId}-name`}
                        aria-describedby={`${itemId}-size`}
                        className="focus-visible:ring-0"
                      />
                      <span className="flex min-w-0 flex-col">
                        <span
                          id={`${itemId}-name`}
                          className="truncate text-[13px] font-medium text-foreground"
                          title={item.name}
                        >
                          {item.name}
                        </span>
                        <span id={`${itemId}-size`} className="text-xs text-muted-foreground">
                          Up to {number(largestPreparationSize(item))} questions
                        </span>
                      </span>
                    </label>
                  )
                })}
                {listed.length === 0 ? (
                  <p className="px-2.5 py-6 text-center text-xs text-muted-foreground">
                    No benchmark matches “{query.trim()}”.
                  </p>
                ) : null}
              </RadioGroup>
            </div>
          ) : null}
          <div
            className={cn(
              'flex min-w-0 flex-col gap-6 px-6 py-5',
              benchmark && 'md:min-h-0 md:overflow-y-auto',
            )}
          >
            {benchmark ? (
              <section aria-labelledby={`${id}-benchmark`} className="flex min-w-0 flex-col gap-5">
                <div className="flex min-w-0 flex-col gap-1">
                  <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
                    <h3
                      id={`${id}-benchmark`}
                      className={cn(benchText.h2, 'min-w-0 [overflow-wrap:anywhere]')}
                    >
                      {benchmark.name}
                    </h3>
                    <a
                      href={benchmark.source_url}
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex shrink-0 items-center gap-1 text-xs font-medium text-primary hover:underline"
                    >
                      View source dataset
                      <ArrowUpRight aria-hidden="true" className="size-3.5" />
                    </a>
                  </div>
                  {benchmark.access_note ? (
                    <p className={benchText.muted}>{benchmark.access_note}</p>
                  ) : null}
                </div>
                <div className="flex min-w-0 flex-col gap-2">
                  <span id={`${id}-size`} className="text-xs font-medium text-muted-foreground">
                    Dataset size
                  </span>
                  <RadioGroup
                    aria-labelledby={`${id}-size`}
                    value={profile}
                    onValueChange={(value) => setProfile(value as PreparationProfile)}
                    disabled={selectionDisabled}
                    className="grid-cols-3"
                  >
                    {PREPARATION_PROFILES.map((value) => {
                      const checked = value === profile
                      const itemId = `${id}-size-${value}`
                      return (
                        <label
                          key={value}
                          htmlFor={itemId}
                          className={cn(
                            'flex min-w-0 cursor-pointer flex-col gap-1 rounded-lg border bg-card px-3 py-2.5 transition-[border-color,box-shadow] has-[:focus-visible]:ring-[3px] has-[:focus-visible]:ring-ring/40 has-[:disabled]:cursor-not-allowed',
                            checked
                              ? 'border-primary shadow-[0_0_0_3px_var(--ring)]'
                              : 'border-border hover:border-border-strong',
                          )}
                        >
                          <span className="flex items-center justify-between gap-2">
                            <span
                              id={`${itemId}-name`}
                              className="truncate text-[13px] font-medium text-foreground"
                            >
                              {profileTitle(value)}
                            </span>
                            <RadioGroupItem
                              id={itemId}
                              value={value}
                              aria-labelledby={`${itemId}-name ${itemId}-count`}
                              className="focus-visible:ring-0"
                            />
                          </span>
                          <span id={`${itemId}-count`} className="text-xs text-muted-foreground">
                            <strong className="text-[15px] font-semibold text-foreground tabular-nums">
                              {number(benchmark.profiles[value])}
                            </strong>{' '}
                            questions
                          </span>
                        </label>
                      )
                    })}
                  </RadioGroup>
                </div>
                <ul className="m-0 flex list-none flex-col gap-1 p-0 text-xs leading-relaxed text-muted-foreground">
                  {benchmark.dependencies.length > 0 && (
                    <li>
                      Dependencies: {benchmark.dependencies.join(', ')} · installed automatically
                    </li>
                  )}
                  <li>Downloads continue in the background, so you can close this dialog.</li>
                </ul>
              </section>
            ) : state.loading ? (
              <BenchLoading label="Loading available datasets…" lines={3} className="my-0" />
            ) : !state.readError ? (
              <Notice tone="neutral" as="p" className="my-0">
                No downloadable benchmarks are available from this service.
              </Notice>
            ) : null}
            {state.preparations.length > 0 && (
              <section
                aria-labelledby={`${id}-downloads`}
                className="flex min-w-0 flex-col gap-2 border-t border-border pt-5"
              >
                <h3 id={`${id}-downloads`} className={benchText.h3}>
                  Downloads
                </h3>
                {pending.length > 0 && jobs(pending, 'Dataset preparations')}
                {completed.length > 0 && (
                  <Disclosure
                    summary={`Completed (${number(completed.length)})`}
                    className={cn(pending.length ? 'mt-1' : 'mt-0', 'mb-0')}
                    bodyClassName="p-2"
                  >
                    {jobs(completed, 'Completed preparations')}
                  </Disclosure>
                )}
              </section>
            )}
          </div>
        </div>
        <DialogFooter>
          {benchmark ? (
            <span className="mr-auto hidden min-w-0 truncate text-xs text-muted-foreground sm:block">
              {`${benchmark.name} · ${profileTitle(profile)} · ${number(benchmark.profiles[profile])} questions`}
            </span>
          ) : null}
          <Button type="button" variant="ghost" size="sm" onClick={onClose}>
            Close
          </Button>
          {benchmark && (
            <Button type="button" size="sm" disabled={startDisabled} onClick={start}>
              <Download aria-hidden="true" />
              {state.submitting
                ? 'Starting preparation…'
                : duplicate
                  ? 'Preparation in progress'
                  : active.length
                    ? 'Another preparation is running'
                    : 'Download and prepare'}
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
