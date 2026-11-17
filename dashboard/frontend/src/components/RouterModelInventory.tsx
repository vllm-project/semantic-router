import React, { useEffect, useMemo, useState } from 'react'
import { ArrowUpRight, ChevronLeft, ChevronRight, Search } from 'lucide-react'

import { Badge, Button, InputGroup, cn } from '../ui'
import NativeSelect from './NativeSelect'
import {
  getLoadedModelCount,
  getPreviewRouterModels,
  getRouterModelAnchor,
  getRouterModelConsumers,
  getRouterModelResources,
  getRouterModelState,
  getRouterModelStateLabel,
  getTotalKnownModelCount,
  sortRouterModels,
  type RouterModelInfo,
  type RouterModelsInfo,
} from '../utils/routerRuntime'
import {
  formatRouterModelLabel,
  formatRouterModelTokens,
  getRouterModelInputLimits,
  getRouterModelKind,
  getRouterModelDevice,
  getRouterModelDisplayName,
  getRouterModelPreviewName,
} from './routerModelPresentation'
import { buildRouterModelDetailSections, type RouterModelDetailRow } from './routerModelDetails'
import {
  clampInventoryPage,
  filterAndSortRouterModels,
  type ModelInventorySort,
  type ModelInventoryStateFilter,
} from './routerModelInventorySupport'

interface RouterModelInventoryProps {
  modelsInfo?: RouterModelsInfo | null
  mode?: 'preview' | 'full' | 'detail'
  previewLimit?: number
  showSummary?: boolean
  emptyMessage?: string
  onSelectModel?: (model: RouterModelInfo) => void
}

type StateTone = 'success' | 'info' | 'warning' | 'neutral'

function getStateTone(model: RouterModelInfo): StateTone {
  switch (getRouterModelState(model)) {
    case 'ready':
      return 'success'
    case 'downloading':
      return 'info'
    case 'pending':
    case 'initializing':
      return 'warning'
    default:
      return 'neutral'
  }
}

const StateBadge: React.FC<{ model: RouterModelInfo }> = ({ model }) => (
  <Badge tone={getStateTone(model)} size="sm" dot className="shrink-0">
    {getRouterModelStateLabel(model)}
  </Badge>
)

function renderDetailRows(rows: RouterModelDetailRow[], wide = false): JSX.Element {
  return (
    <dl className="flex flex-col divide-y divide-border">
      {rows.map((row) => (
        <div
          key={`${row.label}-${row.value}`}
          className={cn(
            'grid gap-x-4 gap-y-0.5 py-2 text-[13px] first:pt-0 last:pb-0',
            wide
              ? 'grid-cols-1 sm:grid-cols-[minmax(10rem,0.4fr)_minmax(0,1fr)]'
              : 'grid-cols-1 sm:grid-cols-[minmax(7rem,0.9fr)_minmax(0,1fr)]',
          )}
        >
          <dt className="text-xs leading-5 text-subtle-foreground">{row.label}</dt>
          <dd className="min-w-0 leading-5 text-foreground [overflow-wrap:anywhere]">
            {row.value}
          </dd>
        </div>
      ))}
    </dl>
  )
}

const DeviceMark: React.FC<{ model: RouterModelInfo }> = ({ model }) => {
  const device = getRouterModelDevice(model)
  return (
    <span className="inline-flex items-center gap-2 text-xs text-muted-foreground">
      {device.isAmd && (
        <img src="/amd-logo.png" alt="AMD GPU" className="h-3.5 w-auto opacity-80 dark:invert" />
      )}
      <span>{device.label}</span>
    </span>
  )
}

const PreviewCardBody: React.FC<{ model: RouterModelInfo; consumers: RouterModelInfo[] }> = ({
  model,
  consumers,
}) => {
  const limits = getRouterModelInputLimits(model)
  const name = getRouterModelPreviewName(model)
  const budgetVaries = consumers.some((consumer) => {
    const other = getRouterModelInputLimits(consumer)
    return (other.document ?? other.input) !== (limits.document ?? limits.input)
  })
  return (
    <>
      <div className="flex items-center justify-between gap-2">
        <span className="truncate text-xs font-medium text-muted-foreground">
          {getRouterModelKind(model)}
        </span>
        <StateBadge model={model} />
      </div>
      <div className="flex min-w-0 flex-col gap-0.5">
        <h3 className="truncate text-sm font-semibold tracking-[-0.01em] text-foreground">
          {name.title}
        </h3>
        {name.subtitle && (
          <p className="truncate text-xs text-subtle-foreground">{name.subtitle}</p>
        )}
      </div>
      <div className="mt-auto flex flex-col gap-1 border-t border-border pt-3 text-xs text-muted-foreground tabular">
        {limits.window !== undefined && <p>Window: {formatRouterModelTokens(limits.window)}</p>}
        {limits.input !== undefined ? (
          <p>
            {limits.overflow === 'window' ? 'Document budget' : 'Input budget'}:{' '}
            {budgetVaries
              ? 'Varies by consumer'
              : formatRouterModelTokens(limits.document ?? limits.input)}
          </p>
        ) : (
          <p>
            {limits.publishedContext !== undefined ? 'Published model context' : 'Input budget'}:{' '}
            {formatRouterModelTokens(limits.publishedContext)}
          </p>
        )}
        {consumers.length > 1 && <p>Shared by {consumers.length} consumers</p>}
      </div>
    </>
  )
}

const FullCardBody: React.FC<{ model: RouterModelInfo; consumers: RouterModelInfo[] }> = ({
  model,
  consumers,
}) => {
  const sections = buildRouterModelDetailSections(model, consumers)

  return (
    <>
      <div className="flex flex-col gap-1">
        <div className="flex flex-wrap items-center gap-2">
          <h3 className="min-w-0 text-[15px] font-semibold tracking-[-0.012em] text-foreground [overflow-wrap:anywhere]">
            {getRouterModelDisplayName(model)}
          </h3>
          <StateBadge model={model} />
        </div>
        <p className="text-xs text-muted-foreground">{getRouterModelKind(model)}</p>
      </div>

      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        {sections.map((section) =>
          section.collapsible ? (
            <details
              key={section.title}
              className="group/details rounded-lg border border-border bg-surface-subtle/50 md:col-span-2"
            >
              <summary className="flex cursor-pointer list-none items-center gap-2.5 rounded-lg px-3.5 py-2.5 text-xs font-medium text-muted-foreground outline-none transition-colors before:size-1.5 before:-rotate-45 before:border-b-[1.5px] before:border-r-[1.5px] before:border-current before:transition-transform before:content-[''] hover:text-foreground focus-visible:ring-[3px] focus-visible:ring-ring/40 group-open/details:before:rotate-45 [&::-webkit-details-marker]:hidden">
                {section.title}
              </summary>
              <div className="border-t border-border px-3.5 py-3 font-mono text-xs [&_dd]:text-xs">
                {renderDetailRows(section.rows, true)}
              </div>
            </details>
          ) : (
            <section
              key={section.title}
              className={cn(
                'flex flex-col gap-2.5 rounded-lg border border-border bg-surface px-3.5 py-3',
                section.wide && 'md:col-span-2',
              )}
            >
              <h4 className="text-[11px] font-medium uppercase tracking-[0.06em] text-subtle-foreground">
                {section.title}
              </h4>
              {renderDetailRows(section.rows, section.wide)}
            </section>
          ),
        )}
      </div>

      <div
        className={cn(
          'flex flex-wrap items-center gap-3',
          model.registry?.model_card_url ? 'justify-between' : 'justify-end',
        )}
      >
        {model.registry?.model_card_url ? (
          <a
            className="inline-flex items-center gap-1 text-[13px] font-medium text-primary-soft-foreground underline-offset-4 hover:underline"
            href={model.registry.model_card_url}
            target="_blank"
            rel="noreferrer"
          >
            Open model card
            <ArrowUpRight className="size-3.5" aria-hidden="true" />
          </a>
        ) : null}

        <DeviceMark model={model} />
      </div>
    </>
  )
}

const RouterModelInventory: React.FC<RouterModelInventoryProps> = ({
  modelsInfo,
  mode = 'full',
  previewLimit,
  showSummary = true,
  emptyMessage = 'No router model metadata is available yet.',
  onSelectModel,
}) => {
  const [query, setQuery] = useState('')
  const [stateFilter, setStateFilter] = useState<ModelInventoryStateFilter>('all')
  const [sort, setSort] = useState<ModelInventorySort>('state')
  const [page, setPage] = useState(1)
  const pageSize = 8
  const resources = useMemo(
    () => getRouterModelResources(sortRouterModels(modelsInfo?.models ?? [])),
    [modelsInfo?.models],
  )
  const allModels = useMemo(() => resources.map(({ model }) => model), [resources])
  const filteredModels = useMemo(
    () =>
      filterAndSortRouterModels(
        resources
          .filter(
            ({ consumers }) =>
              filterAndSortRouterModels(consumers, query, 'all', 'state').length > 0,
          )
          .map(({ model }) => model),
        '',
        stateFilter,
        sort,
      ),
    [resources, query, sort, stateFilter],
  )
  const totalPages = Math.max(1, Math.ceil(filteredModels.length / pageSize))
  const currentPage = clampInventoryPage(page, filteredModels.length, pageSize)
  const models =
    mode === 'preview'
      ? getPreviewRouterModels(modelsInfo, previewLimit)
      : filteredModels.slice((currentPage - 1) * pageSize, currentPage * pageSize)

  const loadedCount = getLoadedModelCount(modelsInfo)
  const totalCount = getTotalKnownModelCount(modelsInfo)
  const phase = modelsInfo?.summary?.phase
  const summaryMessage = modelsInfo?.summary?.message

  useEffect(() => {
    setPage(1)
  }, [query, sort, stateFilter])

  useEffect(() => {
    if (mode !== 'full' || !window.location.hash) return
    const anchor = decodeURIComponent(window.location.hash.slice(1))
    const modelIndex = filteredModels.findIndex((model) => getRouterModelAnchor(model) === anchor)
    if (modelIndex >= 0) setPage(Math.floor(modelIndex / pageSize) + 1)
  }, [filteredModels, mode])

  if (allModels.length === 0) {
    return (
      <div className="rounded-lg border border-dashed border-border-strong px-4 py-8 text-center text-[13px] text-muted-foreground">
        {emptyMessage}
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-4">
      {mode === 'full' && (
        <>
          {showSummary && (
            <div className="flex flex-wrap items-center gap-3">
              <div className="flex min-w-28 flex-col gap-0.5 rounded-lg border border-border bg-surface px-3.5 py-2.5">
                <span className="text-xs text-subtle-foreground">Runtimes ready</span>
                <span className="text-base font-semibold tabular tracking-[-0.02em]">
                  {loadedCount}/{totalCount}
                </span>
              </div>
              {phase && (
                <div className="flex min-w-28 flex-col gap-0.5 rounded-lg border border-border bg-surface px-3.5 py-2.5">
                  <span className="text-xs text-subtle-foreground">Phase</span>
                  <span className="text-base font-semibold tracking-[-0.02em]">
                    {formatRouterModelLabel(phase)}
                  </span>
                </div>
              )}
              {summaryMessage && (
                <p className="min-w-56 flex-1 text-[13px] leading-relaxed text-muted-foreground">
                  {summaryMessage}
                </p>
              )}
            </div>
          )}

          <div className="flex flex-wrap items-center gap-2 border-b border-border pb-3">
            <label className="min-w-56 flex-1">
              <span className="sr-only">Search models</span>
              <InputGroup
                type="search"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Search name, type, repository, or tag"
                leading={<Search />}
                className="h-8"
              />
            </label>
            <label className="flex items-center gap-2 text-xs text-muted-foreground">
              Status
              <NativeSelect
                size="sm"
                containerClassName="w-36"
                value={stateFilter}
                onChange={(event) =>
                  setStateFilter(event.target.value as ModelInventoryStateFilter)
                }
              >
                <option value="all">All statuses</option>
                <option value="ready">Ready</option>
                <option value="loading">Loading</option>
                <option value="not_loaded">Not loaded</option>
              </NativeSelect>
            </label>
            <label className="flex items-center gap-2 text-xs text-muted-foreground">
              Sort
              <NativeSelect
                size="sm"
                containerClassName="w-32"
                value={sort}
                onChange={(event) => setSort(event.target.value as ModelInventorySort)}
              >
                <option value="state">Readiness</option>
                <option value="name">Name</option>
                <option value="type">Type</option>
              </NativeSelect>
            </label>
            <span
              className="ml-auto whitespace-nowrap text-xs tabular text-muted-foreground"
              aria-live="polite"
            >
              {filteredModels.length} of {allModels.length} runtimes
            </span>
          </div>
        </>
      )}

      {mode === 'full' && filteredModels.length === 0 ? (
        <div className="flex flex-col items-center gap-3 rounded-lg border border-dashed border-border-strong px-4 py-8 text-center text-[13px] text-muted-foreground">
          No models match the current search and status filters.
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => {
              setQuery('')
              setStateFilter('all')
              setSort('state')
            }}
          >
            Clear filters
          </Button>
        </div>
      ) : (
        <>
          <div
            className={cn(
              'grid gap-3',
              mode === 'preview'
                ? 'grid-cols-[repeat(auto-fit,minmax(min(100%,16rem),1fr))]'
                : 'grid-cols-1',
            )}
            data-testid={`router-model-grid-${mode}`}
          >
            {models.map((model) => {
              const consumers = getRouterModelConsumers(modelsInfo?.models ?? [], model)
              const interactive = Boolean(onSelectModel) && mode === 'preview'
              const className = cn(
                'relative flex min-w-0 flex-col text-left',
                mode === 'preview'
                  ? 'min-h-[10.5rem] gap-3 rounded-xl border border-border bg-card p-4 shadow-xs'
                  : mode === 'full'
                    ? 'gap-4 rounded-xl border border-border bg-card p-5 shadow-xs'
                    : 'gap-4',
                interactive &&
                  'group cursor-pointer outline-none transition-[border-color,box-shadow,transform] duration-200 hover:-translate-y-px hover:border-border-strong hover:shadow-md focus-visible:ring-[3px] focus-visible:ring-ring/40',
              )

              const cardContent =
                mode === 'preview' ? (
                  <PreviewCardBody model={model} consumers={consumers} />
                ) : (
                  <FullCardBody model={model} consumers={consumers} />
                )

              if (interactive && onSelectModel) {
                return (
                  <button
                    key={getRouterModelAnchor(model)}
                    type="button"
                    className={className}
                    data-testid={`router-model-${mode}-${model.name}`}
                    onClick={() => onSelectModel(model)}
                  >
                    {cardContent}
                    <ChevronRight
                      aria-hidden="true"
                      className="absolute bottom-4 right-4 size-4 text-subtle-foreground opacity-0 transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100"
                    />
                  </button>
                )
              }

              return (
                <article
                  key={getRouterModelAnchor(model)}
                  id={mode === 'full' ? getRouterModelAnchor(model) : undefined}
                  className={className}
                  data-testid={`router-model-${mode}-${model.name}`}
                >
                  {cardContent}
                </article>
              )
            })}
          </div>

          {mode === 'full' && totalPages > 1 && (
            <nav
              className="flex items-center justify-end gap-2 text-xs text-muted-foreground"
              aria-label="Model inventory pages"
            >
              <Button
                type="button"
                variant="outline"
                size="xs"
                disabled={currentPage === 1}
                onClick={() => setPage((value) => Math.max(1, value - 1))}
              >
                <ChevronLeft />
                Previous
              </Button>
              <span className="min-w-[5.5rem] text-center tabular">
                Page {currentPage} of {totalPages}
              </span>
              <Button
                type="button"
                variant="outline"
                size="xs"
                disabled={currentPage === totalPages}
                onClick={() => setPage((value) => Math.min(totalPages, value + 1))}
              >
                Next
                <ChevronRight />
              </Button>
            </nav>
          )}
        </>
      )}
    </div>
  )
}

export default RouterModelInventory
