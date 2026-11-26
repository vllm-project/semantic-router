import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Atom,
  CodeXml,
  GraduationCap,
  ListChecks,
  MessageCircleQuestion,
  Sigma,
  SquareTerminal,
  type LucideIcon,
} from 'lucide-react'
import {
  Badge,
  Button,
  Checkbox,
  RadioGroup,
  RadioGroupItem,
  SectionLabel,
  StatusDot,
  cn,
} from '../../ui'
import { benchApi } from './api'
import BenchSelect from './BenchSelect'
import {
  catalogBenchmarkTitle,
  friendlyDatasetName,
  profileDescription,
  profileTitle,
  selectedQuestionCount,
} from './datasetPresentation'
import type { Catalog, Dataset, DatasetSelection } from './types'
import { BenchBlock, BenchLoading, Notice } from './benchUi'
import { benchText, choiceCardClass } from './benchStyles'

export interface ResolvedDatasetScope {
  profile: string
  sourceIDs: string[]
  missingBenchmarks: string[]
  seed: number | null
  ready: boolean
  error: string
}

export default function RunDatasetScope({
  catalog,
  datasets,
  profile,
  onProfile,
  benchmarks,
  onBenchmarks,
  initialDataset,
  onResolved,
}: {
  catalog: Catalog
  datasets: Dataset[]
  profile: string
  onProfile: (value: string) => void
  benchmarks: string[]
  onBenchmarks: (value: string[]) => void
  initialDataset?: string
  onResolved: (value: ResolvedDatasetScope) => void
}) {
  const [custom, setCustom] = useState(!!initialDataset)
  const [datasetID, setDatasetID] = useState(initialDataset ?? '')
  const [selection, setSelection] = useState<DatasetSelection | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const initialized = useRef(!!initialDataset)
  const [revision, setRevision] = useState(0)
  useEffect(() => {
    if (custom) {
      setLoading(false)
      setError('')
      return
    }
    const controller = new AbortController()
    setLoading(true)
    setError('')
    setSelection(null)
    void benchApi
      .datasetSelection(profile, controller.signal)
      .then((value) => {
        if (controller.signal.aborted) return
        if (value.profile !== profile)
          throw new Error('The prepared scope did not match the selected profile.')
        setSelection(value)
        if (!initialized.current) {
          initialized.current = true
          onBenchmarks(value.benchmarks.filter((entry) => entry.eligible).map((entry) => entry.id))
        }
      })
      .catch((cause) => {
        if (!controller.signal.aborted)
          setError(
            cause instanceof Error ? cause.message : 'Prepared benchmarks could not be checked.',
          )
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false)
      })
    return () => controller.abort()
  }, [custom, profile, revision, onBenchmarks])
  const dataset = datasets.find((item) => item.id === datasetID)
  const entries = catalog.benchmarks.map((benchmark) => {
    if (custom) {
      const eligible = dataset?.profile === profile && !!dataset.benchmarks?.includes(benchmark.id)
      return {
        id: benchmark.id,
        eligible,
        reason: eligible ? null : 'Not included in this dataset',
        reason_code: eligible ? null : 'not_in_dataset',
        case_count: 0,
        source_ids: eligible ? [dataset!.id] : [],
      }
    }
    return selection?.profile === profile
      ? selection.benchmarks.find((entry) => entry.id === benchmark.id)
      : undefined
  })
  const available = entries
    .filter((entry) => entry?.eligible || (!custom && entry?.reason_code === 'not_prepared'))
    .map((entry) => entry!.id)
  const resolved = useMemo((): ResolvedDatasetScope => {
    if (custom) {
      const ready =
        !!dataset &&
        dataset.profile === profile &&
        benchmarks.length > 0 &&
        benchmarks.every((id) => dataset.benchmarks?.includes(id))
      return {
        profile,
        sourceIDs: ready ? [dataset.id] : [],
        missingBenchmarks: [],
        seed: dataset?.seed ?? null,
        ready,
        error: ready
          ? ''
          : 'Choose a dataset and benchmarks included in its profile. Existing selections are preserved until you change them.',
      }
    }
    if (loading || selection?.profile !== profile || error)
      return {
        profile,
        sourceIDs: [],
        missingBenchmarks: [],
        seed: null,
        ready: false,
        error: error || 'Wait for the service to check this profile.',
      }
    const chosen = benchmarks.map((id) => selection.benchmarks.find((entry) => entry.id === id))
    const ready =
      chosen.length > 0 &&
      chosen.every(
        (entry) =>
          (entry?.eligible && entry.source_ids.length > 0 && selection.seed !== null) ||
          entry?.reason_code === 'not_prepared',
      )
    return {
      profile,
      sourceIDs: ready ? [...new Set(chosen.flatMap((entry) => entry!.source_ids))] : [],
      missingBenchmarks: ready
        ? chosen.filter((entry) => entry?.reason_code === 'not_prepared').map((entry) => entry!.id)
        : [],
      seed: selection.seed ?? 20260918,
      ready,
      error: ready
        ? ''
        : benchmarks.length
          ? 'Some selected benchmarks are unavailable for this profile. Deselect them or choose a specific dataset.'
          : 'Select at least one benchmark.',
    }
  }, [custom, dataset, profile, benchmarks, loading, selection, error])
  useEffect(() => onResolved(resolved), [resolved, onResolved])
  const allSelected = available.length > 0 && available.every((id) => benchmarks.includes(id))
  const selectedQuestions =
    !custom && selection?.profile === profile ? selectedQuestionCount(selection, benchmarks) : null
  return (
    <div className="flex min-w-0 flex-col gap-6">
      <section className="flex min-w-0 flex-col gap-2.5">
        <SectionLabel>Size</SectionLabel>
        <RadioGroup
          value={profile}
          onValueChange={onProfile}
          aria-label="Evaluation size"
          className="grid-cols-1 gap-2.5 sm:grid-cols-3"
        >
          {['smoke', 'quick', 'standard']
            .filter((id) => catalog.profiles.some((item) => item.id === id))
            .map((id) => (
              <label key={id} className={cn(choiceCardClass, 'items-start')}>
                <RadioGroupItem value={id} className="mt-0.5" />
                <span className="flex min-w-0 flex-col gap-1">
                  <strong className="text-[13px] font-semibold text-foreground">
                    {profileTitle(id)}
                  </strong>
                  <span className="text-xs leading-snug text-muted-foreground">
                    {profileDescription(id)}
                  </span>
                  {id === profile && selectedQuestions !== null && (
                    <span className="mt-0.5 font-mono text-[11px] text-subtle-foreground tabular-nums">
                      {selectedQuestions.toLocaleString()} questions per target
                    </span>
                  )}
                </span>
              </label>
            ))}
        </RadioGroup>
      </section>
      <section className="flex min-w-0 flex-col gap-2.5">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <SectionLabel>Benchmarks</SectionLabel>
          <span className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground tabular-nums">
              {benchmarks.length} selected
            </span>
            <Button
              variant="ghost"
              size="xs"
              disabled={!available.length || loading}
              onClick={() =>
                onBenchmarks(
                  allSelected
                    ? benchmarks.filter((id) => !available.includes(id))
                    : [...new Set([...benchmarks, ...available])],
                )
              }
            >
              {allSelected ? 'Clear benchmarks' : 'Select all benchmarks'}
            </Button>
          </span>
        </div>
        {!custom && loading && <BenchLoading label="Checking prepared benchmarks…" />}
        {!custom && error && (
          <Notice tone="danger" role="alert" className="my-0">
            <p>{error}</p>
            <div>
              <Button variant="outline" size="xs" onClick={() => setRevision((value) => value + 1)}>
                Retry benchmark check
              </Button>
            </div>
          </Notice>
        )}
        <div
          className="grid min-w-0 grid-cols-1 gap-2.5 sm:grid-cols-2"
          role="group"
          aria-label="Included benchmarks"
        >
          {catalog.benchmarks.map((benchmark, index) => {
            const entry = entries[index]
            const selected = benchmarks.includes(benchmark.id)
            const missing = entry?.reason_code === 'not_prepared'
            const note = entry?.eligible
              ? custom
                ? 'Included in selected dataset'
                : `${entry.case_count} questions`
              : missing
                ? 'Prepared automatically'
                : (entry?.reason ?? 'Not available')
            const title = catalogBenchmarkTitle(benchmark.id, catalog.benchmarks)
            const Icon = benchmarkIcon(benchmark.kind)
            return (
              <label key={benchmark.id} className={cn(choiceCardClass, 'items-start gap-3 py-3')}>
                <Checkbox
                  checked={selected}
                  className="mt-0.5"
                  disabled={!selected && (!available.includes(benchmark.id) || loading)}
                  onCheckedChange={(checked) =>
                    onBenchmarks(
                      checked === true
                        ? [...benchmarks, benchmark.id]
                        : benchmarks.filter((id) => id !== benchmark.id),
                    )
                  }
                />
                <span className="flex min-w-0 flex-1 flex-col gap-1">
                  <span className="flex min-w-0 items-center gap-2">
                    <Icon aria-hidden="true" className="size-3.5 shrink-0 text-muted-foreground" />
                    <span className="truncate text-[13px] font-semibold text-foreground">
                      {title}
                    </span>
                    {benchmark.kind && benchmark.kind !== 'capability' ? (
                      <Badge tone="outline" size="sm" className="capitalize">
                        {benchmark.kind}
                      </Badge>
                    ) : null}
                  </span>
                  {benchmark.description ? (
                    <span className="line-clamp-2 text-xs leading-relaxed text-muted-foreground">
                      {benchmark.description}
                    </span>
                  ) : null}
                  <span className="inline-flex items-center gap-1.5 text-[11.5px] text-subtle-foreground">
                    <StatusDot
                      tone={entry?.eligible ? 'success' : missing ? 'warning' : 'neutral'}
                    />
                    <span className={cn(entry?.eligible && 'font-mono tabular-nums')}>{note}</span>
                  </span>
                </span>
              </label>
            )
          })}
        </div>
        {!custom && (
          <p className={benchText.muted}>
            Missing datasets and required dependencies are prepared automatically when you review.
          </p>
        )}
        {benchmarks.length > 0 && !resolved.ready && !loading && !error && (
          <Notice tone="warning" as="p" className="my-0">
            {resolved.error}
          </Notice>
        )}
      </section>
      {custom ? (
        <BenchBlock className="grid gap-3 bg-surface-subtle" aria-label="Specific dataset">
          <div className="max-w-xl">
            <BenchSelect
              label="Specific dataset"
              value={datasetID}
              searchable
              placeholder="Choose a frozen dataset"
              options={datasets
                .filter((item) => item.profile === profile)
                .map((item) => ({
                  value: item.id,
                  label: friendlyDatasetName(item),
                  description: `${item.case_count} questions · ${item.split ?? 'split unspecified'}`,
                }))}
              onChange={setDatasetID}
            />
          </div>
          <p className={benchText.muted}>
            This explicit dataset selection replaces automatic source resolution. Benchmark
            selections are preserved; no questions are resampled.
          </p>
          <div>
            <Button
              variant="link"
              size="sm"
              className="h-auto p-0"
              onClick={() => setCustom(false)}
            >
              Choose benchmarks automatically
            </Button>
          </div>
        </BenchBlock>
      ) : (
        <div>
          <Button variant="link" size="sm" className="h-auto p-0" onClick={() => setCustom(true)}>
            Use a specific dataset
          </Button>
        </div>
      )}
    </div>
  )
}

function benchmarkIcon(kind?: string): LucideIcon {
  const value = (kind ?? '').toLowerCase()
  if (value.includes('cod')) return CodeXml
  if (value.includes('math')) return Sigma
  if (value.includes('science')) return Atom
  if (value.includes('agent') || value.includes('terminal') || value.includes('tool'))
    return SquareTerminal
  if (value.includes('fact')) return MessageCircleQuestion
  if (value.includes('knowledge')) return GraduationCap
  return ListChecks
}
