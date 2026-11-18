import { useCallback, useMemo, useState } from 'react'
import { Copy, Database, Download, ListChecks, Play, RotateCw, Search } from 'lucide-react'
import {
  Badge,
  Button,
  DataTable,
  DropdownMenuItem,
  DropdownMenuSeparator,
  EmptyState,
  FilterPill,
  IconTile,
  InputGroup,
  StatusDot,
  Tooltip,
  toast,
  type Column,
} from '../../ui'
import BenchPagination from './BenchPagination'
import { DatasetSheet } from './DatasetDetails'
import DatasetExposure from './DatasetExposure'
import PrepareDialog from './PrepareDialog'
import {
  catalogBenchmarkTitle,
  datasetEvaluationTitle,
  friendlyDatasetName,
  profileTitle,
} from './datasetPresentation'
import {
  preparationBenchmarkIDs,
  preparationIsActive,
  preparationPhase,
  type DatasetPreparationJob,
} from './datasetPreparationApi'
import { number } from './model'
import { retryPreparation, useDatasetPreparations } from './useDatasetPreparations'
import type { Catalog, Dataset, Run } from './types'

const PAGE_SIZE = 20

type CatalogRow =
  | { kind: 'dataset'; id: string; dataset: Dataset; runs: number }
  | { kind: 'job'; id: string; job: DatasetPreparationJob }

export default function DatasetsView({
  datasets,
  loaded,
  readError,
  runs,
  runsLoaded,
  catalog,
  canRun,
  writeDisabledReason,
  accessRefreshing,
  onRefreshAccess,
  selectedDatasetID,
  onSelectDataset,
  prepareOpen,
  onPrepareOpen,
  onUse,
  onInventoryChanged,
}: {
  datasets: Dataset[]
  loaded: boolean
  readError?: string
  runs: Run[]
  runsLoaded: boolean
  catalog: Catalog | null
  canRun: boolean
  writeDisabledReason: string | null
  accessRefreshing: boolean
  onRefreshAccess: () => void
  selectedDatasetID?: string
  onSelectDataset: (id?: string) => void
  prepareOpen: boolean
  onPrepareOpen: (open: boolean) => void
  onUse: (dataset: Dataset) => void
  onInventoryChanged: () => void
}) {
  const completed = useCallback(() => onInventoryChanged(), [onInventoryChanged])
  const preparations = useDatasetPreparations(completed, prepareOpen)
  const [query, setQuery] = useState('')
  const [profiles, setProfiles] = useState<string[]>([])
  const [benchmarks, setBenchmarks] = useState<string[]>([])
  const [page, setPage] = useState(0)
  const title = (id: string) => catalogBenchmarkTitle(id, catalog?.benchmarks)

  const usage = useMemo(() => {
    const counts = new Map<string, number>()
    for (const run of runs) {
      const sha = run.manifest.dataset?.sha256
      if (sha) counts.set(sha, (counts.get(sha) ?? 0) + 1)
    }
    return counts
  }, [runs])
  const rows: CatalogRow[] = useMemo(() => {
    const text = query.trim().toLowerCase()
    const matches = (name: string, profile: string | undefined, ids: string[]) =>
      (!profiles.length || profiles.includes(profile ?? '')) &&
      (!benchmarks.length || ids.some((id) => benchmarks.includes(id))) &&
      (!text || `${name} ${ids.map(title).join(' ')}`.toLowerCase().includes(text))
    const jobs: CatalogRow[] = preparations.preparations
      .filter((job) => job.status !== 'completed')
      .filter((job) => {
        const ids = preparationBenchmarkIDs(job)
        return matches(ids.map(title).join(', '), job.profile, ids)
      })
      .map((job) => ({ kind: 'job', id: `job:${job.id}`, job }))
    const prepared: CatalogRow[] = datasets
      .filter((dataset) =>
        matches(friendlyDatasetName(dataset), dataset.profile, dataset.benchmarks ?? []),
      )
      .map((dataset) => ({
        kind: 'dataset',
        id: dataset.id,
        dataset,
        runs: dataset.sha256 ? (usage.get(dataset.sha256) ?? 0) : 0,
      }))
    return [...jobs, ...prepared]
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasets, preparations.preparations, query, profiles, benchmarks, usage, catalog])
  const pages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE))
  const current = Math.min(page, pages - 1)
  const totalQuestions = datasets.reduce((sum, dataset) => sum + (dataset.case_count ?? 0), 0)
  const benchmarkIDs = [...new Set(datasets.flatMap((dataset) => dataset.benchmarks ?? []))]

  const columns: Column<CatalogRow>[] = [
    {
      id: 'dataset',
      header: 'Dataset',
      cell: (row) => {
        const ids =
          row.kind === 'dataset' ? (row.dataset.benchmarks ?? []) : preparationBenchmarkIDs(row.job)
        const name =
          row.kind === 'dataset' ? friendlyDatasetName(row.dataset) : ids.map(title).join(', ')
        const profile = row.kind === 'dataset' ? row.dataset.profile : row.job.profile
        return (
          <div className="flex min-w-0 items-center gap-3">
            <IconTile
              icon={row.kind === 'dataset' ? ListChecks : Download}
              tone="orange"
              size="sm"
            />
            <div className="flex min-w-0 flex-col gap-0.5">
              <span className="flex min-w-0 items-center gap-2">
                {row.kind === 'dataset' ? (
                  <button
                    type="button"
                    className="truncate rounded-sm text-left text-[13px] font-medium outline-none hover:underline focus-visible:ring-[3px] focus-visible:ring-ring/40"
                    onClick={(event) => {
                      event.stopPropagation()
                      onSelectDataset(row.id)
                    }}
                  >
                    {name}
                  </button>
                ) : (
                  <span className="truncate text-[13px] font-medium">{name}</span>
                )}
                <Badge tone="outline" size="sm">
                  {profileTitle(profile)}
                </Badge>
              </span>
              {ids.length > 1 || row.kind === 'job' ? (
                <span className="truncate text-[11.5px] text-subtle-foreground">
                  {row.kind === 'job' ? 'Download' : ids.map(title).join(' · ')}
                </span>
              ) : null}
            </div>
          </div>
        )
      },
    },
    {
      id: 'questions',
      header: 'Questions',
      align: 'right',
      cell: (row) =>
        row.kind === 'dataset' ? (
          <span className="font-mono text-[13px] tabular">{number(row.dataset.case_count)}</span>
        ) : (
          <span className="text-subtle-foreground">—</span>
        ),
    },
    {
      id: 'role',
      header: 'Evaluation role',
      hideBelow: 'xl',
      cell: (row) => {
        if (row.kind !== 'dataset') return null
        const role = datasetEvaluationTitle(row.dataset.preparation, row.dataset.benchmarks)
        if (!role) return <span className="text-subtle-foreground">—</span>
        return (
          <span className="flex flex-col text-[13px] [&_p]:m-0 [&_p]:text-[11.5px]">
            <span>{role}</span>
            {role === 'Mixed evaluation roles' ? (
              <DatasetExposure
                compact
                preparation={row.dataset.preparation}
                benchmarks={row.dataset.benchmarks}
              />
            ) : null}
          </span>
        )
      },
    },
    {
      id: 'revision',
      header: 'Revision',
      hideBelow: 'md',
      cell: (row) =>
        row.kind === 'dataset' && row.dataset.sha256 ? (
          <Tooltip content="First characters of the frozen cases checksum">
            <span className="font-mono text-xs text-muted-foreground">
              {row.dataset.sha256.slice(0, 7)}
            </span>
          </Tooltip>
        ) : (
          <span className="text-subtle-foreground">—</span>
        ),
    },
    ...(runsLoaded
      ? [
          {
            id: 'runs',
            header: 'Used by',
            hideBelow: 'lg',
            cell: (row) =>
              row.kind === 'dataset' ? (
                <span className="text-[13px] text-muted-foreground tabular">
                  {row.runs ? `${row.runs} ${row.runs === 1 ? 'run' : 'runs'}` : '—'}
                </span>
              ) : null,
          } satisfies Column<CatalogRow>,
        ]
      : []),
    {
      id: 'status',
      header: 'Status',
      cell: (row) =>
        row.kind === 'dataset' ? (
          <span className="inline-flex items-center gap-1.5 text-[13px]">
            <StatusDot tone="success" />
            Ready
          </span>
        ) : row.job.status === 'failed' ? (
          <Tooltip content={row.job.error || 'Preparation failed'}>
            <span className="inline-flex items-center gap-1.5 text-[13px]">
              <StatusDot tone="danger" />
              Failed
            </span>
          </Tooltip>
        ) : (
          <span className="inline-flex items-center gap-1.5 text-[13px]">
            <StatusDot tone="info" pulse={preparationIsActive(row.job)} />
            {preparationPhase(row.job)}
          </span>
        ),
    },
  ]

  return (
    <section aria-label="Dataset library" className="min-w-0">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <InputGroup
          type="search"
          aria-label="Search datasets"
          placeholder="Search datasets…"
          value={query}
          onChange={(event) => {
            setQuery(event.target.value)
            setPage(0)
          }}
          leading={<Search aria-hidden="true" />}
          containerClassName="w-full sm:w-64"
          className="h-8"
        />
        <FilterPill
          label="Profile"
          options={['smoke', 'quick', 'standard'].map((value) => ({
            value,
            label: profileTitle(value),
            count: datasets.filter((dataset) => dataset.profile === value).length,
          }))}
          selected={profiles}
          onChange={(next) => {
            setProfiles(next)
            setPage(0)
          }}
        />
        <FilterPill
          label="Benchmark"
          options={benchmarkIDs.map((value) => ({
            value,
            label: title(value),
            count: datasets.filter((dataset) => dataset.benchmarks?.includes(value)).length,
          }))}
          selected={benchmarks}
          onChange={(next) => {
            setBenchmarks(next)
            setPage(0)
          }}
        />
        {loaded ? (
          <Tooltip content="Question totals are across prepared sets. Sets may overlap.">
            <span className="ml-auto text-xs text-muted-foreground tabular" role="status">
              {number(totalQuestions)} questions · {datasets.length}{' '}
              {datasets.length === 1 ? 'prepared set' : 'prepared sets'}
            </span>
          </Tooltip>
        ) : null}
      </div>
      {!loaded && readError ? (
        <div className="rounded-xl border border-border bg-card shadow-xs">
          <EmptyState
            compact
            icon={Database}
            title="Dataset inventory is unavailable"
            description="Prepared datasets stay on the sr-bench service. This list reconnects automatically."
          />
        </div>
      ) : (
        <DataTable
          data={rows.slice(current * PAGE_SIZE, (current + 1) * PAGE_SIZE)}
          columns={columns}
          getRowId={(row) => row.id}
          loading={!loaded}
          activeRowId={selectedDatasetID}
          onRowClick={(row) =>
            row.kind === 'dataset' ? onSelectDataset(row.id) : onPrepareOpen(true)
          }
          rowActions={(row) =>
            row.kind === 'dataset' ? (
              <>
                <DropdownMenuItem disabled={!canRun} onSelect={() => onUse(row.dataset)}>
                  <Play aria-hidden="true" /> Evaluate
                </DropdownMenuItem>
                <DropdownMenuItem onSelect={() => onSelectDataset(row.id)}>
                  <ListChecks aria-hidden="true" /> Browse questions
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  onSelect={() => {
                    void navigator.clipboard?.writeText(row.id).catch(() => undefined)
                    toast.success('Dataset ID copied')
                  }}
                >
                  <Copy aria-hidden="true" /> Copy dataset ID
                </DropdownMenuItem>
              </>
            ) : (
              <>
                <DropdownMenuItem onSelect={() => onPrepareOpen(true)}>
                  <Download aria-hidden="true" /> View downloads
                </DropdownMenuItem>
                {row.job.status === 'failed' ? (
                  <DropdownMenuItem
                    disabled={
                      !!writeDisabledReason ||
                      preparations.submitting ||
                      preparations.preparations.some(preparationIsActive)
                    }
                    onSelect={() => retryPreparation(preparations, row.job)}
                  >
                    <RotateCw aria-hidden="true" /> Retry preparation
                  </DropdownMenuItem>
                ) : null}
              </>
            )
          }
          empty={
            datasets.length === 0 && !preparations.preparations.length ? (
              <EmptyState
                icon={Database}
                tone="orange"
                title="No prepared datasets yet"
                description="New evaluations prepare the questions they need automatically, or download a benchmark ahead of time."
                actions={
                  <Button size="sm" variant="outline" onClick={() => onPrepareOpen(true)}>
                    <Download aria-hidden="true" /> Prepare dataset
                  </Button>
                }
              />
            ) : (
              <EmptyState
                compact
                icon={ListChecks}
                title="No matching datasets"
                description="Change the search or filters to see more prepared sets."
              />
            )
          }
        />
      )}
      <BenchPagination
        label="Datasets"
        total={rows.length}
        page={current}
        pageSize={PAGE_SIZE}
        onChange={setPage}
      />
      <DatasetSheet
        id={selectedDatasetID ?? null}
        datasets={datasets}
        canRun={canRun}
        onUse={onUse}
        onClose={() => onSelectDataset(undefined)}
      />
      <PrepareDialog
        open={prepareOpen}
        state={preparations}
        disabledReason={writeDisabledReason}
        accessRefreshing={accessRefreshing}
        onRefreshAccess={onRefreshAccess}
        onOpenDataset={(dataset) => {
          onPrepareOpen(false)
          onSelectDataset(dataset.id)
        }}
        onClose={() => onPrepareOpen(false)}
      />
    </section>
  )
}
