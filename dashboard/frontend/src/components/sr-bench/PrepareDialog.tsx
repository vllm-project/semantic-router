import { useState } from 'react'
import { ArrowUpRight, Download, RefreshCw } from 'lucide-react'
import {
  Button,
  Dialog,
  DialogBody,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  StatusDot,
  cn,
  type StatusTone,
} from '../../ui'
import BenchSelect from './BenchSelect'
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
  const [benchmarkID, setBenchmarkID] = useState('')
  const [profile, setProfile] = useState<PreparationProfile>('smoke')
  const benchmark = state.benchmarks.find((item) => item.id === benchmarkID) ?? state.benchmarks[0]
  const active = state.preparations.filter(preparationIsActive)
  const pending = state.preparations.filter((job) => job.status !== 'completed')
  const completed = state.preparations.filter((job) => job.status === 'completed')
  const duplicate = active.some(
    (job) => preparationBenchmarkIDs(job).includes(benchmark?.id ?? '') && job.profile === profile,
  )
  const profiles = ['smoke', 'quick', 'standard'] as const
  const selectionDisabled = state.submitting || state.loading
  const mutationDisabled = !!disabledReason || selectionDisabled

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

  return (
    <Dialog open={open} onOpenChange={(value) => !value && onClose()}>
      <DialogContent size="lg">
        <DialogHeader>
          <DialogTitle>Prepare a dataset</DialogTitle>
          <DialogDescription>
            New evaluations prepare missing datasets automatically. Download one ahead of time here;
            no model answers are generated.
          </DialogDescription>
        </DialogHeader>
        <DialogBody className="max-h-[min(70vh,640px)] overflow-y-auto">
          <section aria-label="Prepare datasets" className="flex min-w-0 flex-col gap-3">
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
            {state.loading && <BenchLoading label="Loading available datasets…" lines={3} />}
            {benchmark && (
              <form
                id="prepare-dataset-form"
                className="flex min-w-0 flex-col gap-3"
                onSubmit={(event) => {
                  event.preventDefault()
                  if (!mutationDisabled && !active.length)
                    void state.prepare({ benchmark: benchmark.id, profile })
                }}
              >
                <div className="grid min-w-0 grid-cols-1 gap-3 sm:grid-cols-2">
                  <BenchSelect
                    label="Benchmark to download"
                    value={benchmark.id}
                    options={state.benchmarks.map((item) => ({ value: item.id, label: item.name }))}
                    onChange={setBenchmarkID}
                    disabled={selectionDisabled}
                    searchable
                  />
                  <BenchSelect
                    label="Dataset size"
                    value={profile}
                    options={profiles.map((value) => ({
                      value,
                      label: profileTitle(value),
                      description: `${number(benchmark.profiles[value])} questions`,
                    }))}
                    onChange={(value) => setProfile(value as PreparationProfile)}
                    disabled={selectionDisabled}
                  />
                </div>
                <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-border bg-surface-subtle/60 px-3.5 py-3">
                  <strong className="text-[15px] font-semibold text-foreground tabular-nums">
                    {number(benchmark.profiles[profile])}{' '}
                    <span className="text-xs font-normal text-muted-foreground">questions</span>
                  </strong>
                  <a
                    href={benchmark.source_url}
                    target="_blank"
                    rel="noreferrer"
                    className="inline-flex w-fit items-center gap-1 text-xs font-medium text-primary hover:underline"
                  >
                    View source dataset
                    <ArrowUpRight aria-hidden="true" className="size-3.5" />
                  </a>
                </div>
                <Disclosure summary="Source and requirements" bare className="my-0">
                  <div className="grid gap-1.5 text-xs text-muted-foreground">
                    {benchmark.access_note && <p>{benchmark.access_note}</p>}
                    <p>Downloads continue in the background and do not generate model answers.</p>
                    {benchmark.dependencies.length > 0 && (
                      <p>
                        Dependencies: {benchmark.dependencies.join(', ')} · installed automatically
                      </p>
                    )}
                  </div>
                </Disclosure>
              </form>
            )}
            {!state.loading && !state.readError && !benchmark && (
              <Notice tone="neutral" as="p" className="my-0">
                No downloadable benchmarks are available from this service.
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
            {state.preparations.length > 0 && (
              <div className="mt-2 min-w-0">
                <h3 className={`${benchText.h3} mb-2`}>Downloads</h3>
                {pending.length > 0 && jobs(pending, 'Dataset preparations')}
                {completed.length > 0 && (
                  <Disclosure
                    summary={`Completed (${number(completed.length)})`}
                    className={cn(pending.length ? 'mt-3' : 'mt-0', 'mb-0')}
                    bodyClassName="p-2"
                  >
                    {jobs(completed, 'Completed preparations')}
                  </Disclosure>
                )}
              </div>
            )}
          </section>
        </DialogBody>
        <DialogFooter>
          <Button type="button" variant="ghost" size="sm" onClick={onClose}>
            Close
          </Button>
          {benchmark && (
            <Button
              type="submit"
              form="prepare-dataset-form"
              size="sm"
              disabled={mutationDisabled || active.length > 0}
            >
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
