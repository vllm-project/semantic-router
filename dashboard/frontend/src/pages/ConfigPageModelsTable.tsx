import { useEffect, useMemo, useState, type KeyboardEvent, type MouseEvent } from 'react'
import {
  Activity,
  Boxes,
  Copy,
  Eye,
  MoreHorizontal,
  Pencil,
  Plus,
  Star,
  Trash2,
} from 'lucide-react'

import { getPageWindow, paginateRows, updatePageSelection } from '../components/dataTableSupport'
import type { BuiltInModelCatalog } from '../types/modelCatalog'
import {
  Badge,
  Button,
  Checkbox,
  DataTable,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
  EmptyState,
  MiniBar,
  Tooltip,
  cn,
  toast,
  type Column,
} from '../ui'
import { ModelHealthCell } from './ConfigPageModelHealth'
import { healthSortValue } from './configPageModelHealthPresentation'
import { ModelLogo } from './ConfigPageModelLogo'
import {
  MODEL_KIND_LABELS,
  modelPriceSummary,
  modelServedBy,
  type ModelKind,
} from './configPageModelPresentation'
import ConfigPageTablePager from './ConfigPageTablePager'
import type { NormalizedModel } from './configPageSupport'
import type { BackendModelHealth } from './useBackendModelHealth'
import type { BackendModelTraffic } from './useBackendModelTraffic'

const PAGE_SIZES = [10, 25, 50, 100]

// Columns leave in this order as the table's container narrows; Traffic,
// when Router Replay supplies it, needs room of its own.
const COLUMN_HIDING = {
  served: ['@max-[1090px]:hidden', '@max-[970px]:hidden'],
  kind: ['@max-[875px]:hidden', '@max-[755px]:hidden'],
  decisions: ['@max-[775px]:hidden', '@max-[655px]:hidden'],
  price: ['@max-[680px]:hidden', '@max-[560px]:hidden'],
} as const

const hideWhenNarrow = (column: keyof typeof COLUMN_HIDING, traffic: boolean) => {
  const className = COLUMN_HIDING[column][traffic ? 0 : 1]
  return { className, headerClassName: className }
}

interface ConfigPageModelsTableProps {
  models: NormalizedModel[]
  totalModels: number
  defaultModel: string
  deployments: ReadonlyMap<string, string>
  referenceCounts: ReadonlyMap<string, number>
  kinds: ReadonlyMap<string, ModelKind>
  catalog: BuiltInModelCatalog
  health: BackendModelHealth
  traffic: BackendModelTraffic
  /** The catalog list price of each model that sets no pricing, as the Router prices it. */
  catalogPrices?: Readonly<Record<string, NonNullable<NormalizedModel['pricing']>>>
  isReadonly: boolean
  activeName: string | null
  resetKey: string
  filtersActive: boolean
  selectedKeys: ReadonlySet<string>
  onSelectedKeysChange: (keys: Set<string>) => void
  getDeleteBlocker: (modelName: string) => string | null
  canBeDefault: (model: NormalizedModel) => boolean
  onOpen: (model: NormalizedModel) => void
  onShowUsage: (model: NormalizedModel) => void
  onEdit: (model: NormalizedModel) => void
  onDelete: (model: NormalizedModel) => void
  onMakeDefault: (model: NormalizedModel) => void
  onCheckHealth: (model: NormalizedModel) => void
  onConnect: () => void
  onClearFilters: () => void
}

const stopRowClick = {
  role: 'presentation',
  onClick: (event: MouseEvent) => event.stopPropagation(),
  onKeyDown: (event: KeyboardEvent) => event.stopPropagation(),
} as const

/**
 * The model's logo, which turns into its selection box on hover, on focus
 * and while any model is selected, so bulk delete needs no checkbox column.
 */
function ModelSelectLogo({
  model,
  catalog,
  selected,
  selecting,
  blocker,
  onToggle,
}: {
  model: NormalizedModel
  catalog: BuiltInModelCatalog
  selected: boolean
  selecting: boolean
  blocker: string | null
  onToggle: (checked: boolean) => void
}) {
  const box = (
    <Checkbox
      aria-label={`Select model ${model.name}`}
      checked={selected}
      disabled={Boolean(blocker)}
      onCheckedChange={(checked) => onToggle(checked === true)}
    />
  )
  return (
    <span className="group/select relative grid size-[30px] shrink-0 place-items-center">
      <span
        className={cn(
          'transition-opacity duration-150',
          selecting ? 'opacity-0' : 'group-hover:opacity-0 group-focus-within/select:opacity-0',
        )}
      >
        <ModelLogo model={model} size={16} framed catalog={catalog} />
      </span>
      <span
        {...stopRowClick}
        className={cn(
          'absolute inset-0 grid place-items-center transition-opacity duration-150',
          selecting
            ? 'opacity-100'
            : 'opacity-0 group-hover:opacity-100 group-focus-within/select:opacity-100',
        )}
      >
        {blocker ? (
          <Tooltip content={blocker}>
            <span className="grid place-items-center">{box}</span>
          </Tooltip>
        ) : (
          box
        )}
      </span>
    </span>
  )
}

export default function ConfigPageModelsTable({
  models,
  totalModels,
  defaultModel,
  deployments,
  referenceCounts,
  kinds,
  catalog,
  health,
  traffic,
  catalogPrices,
  isReadonly,
  activeName,
  resetKey,
  filtersActive,
  selectedKeys,
  onSelectedKeysChange,
  getDeleteBlocker,
  canBeDefault,
  onOpen,
  onShowUsage,
  onEdit,
  onDelete,
  onMakeDefault,
  onCheckHealth,
  onConnect,
  onClearFilters,
}: ConfigPageModelsTableProps) {
  const [pageSize, setPageSize] = useState(25)
  const [page, setPage] = useState(1)
  useEffect(() => setPage(1), [resetKey])
  const pageWindow = getPageWindow(models.length, page, pageSize)
  useEffect(() => {
    if (pageWindow.page !== page) setPage(pageWindow.page)
  }, [page, pageWindow.page])
  const rows = paginateRows(models, pageWindow)
  const selectablePageKeys = rows
    .filter((model) => !getDeleteBlocker(model.name))
    .map((model) => model.name)
  const selectedOnPage = selectablePageKeys.filter((key) => selectedKeys.has(key)).length
  const selecting = !isReadonly && selectedKeys.size > 0
  const canCheckHealth = health.phase !== 'unavailable'

  const columns = useMemo(() => {
    const showTraffic = traffic.phase === 'ready' && traffic.requests > 0
    const maxShare = Math.max(1, ...traffic.shares.values())
    const shareOf = (model: NormalizedModel) =>
      traffic.shares.get(model.name) ?? (traffic.complete ? 0 : undefined)
    const list: Column<NormalizedModel>[] = [
      {
        id: 'model',
        header:
          selecting && selectablePageKeys.length ? (
            <span className="inline-flex items-center gap-3">
              <span className="grid w-[30px] place-items-center" {...stopRowClick}>
                <Checkbox
                  aria-label="Select all models on this page"
                  checked={
                    selectedOnPage === selectablePageKeys.length
                      ? true
                      : selectedOnPage > 0
                        ? 'indeterminate'
                        : false
                  }
                  onCheckedChange={(checked) =>
                    onSelectedKeysChange(
                      updatePageSelection(selectedKeys, selectablePageKeys, checked === true),
                    )
                  }
                />
              </span>
              Model
            </span>
          ) : (
            'Model'
          ),
        sortValue: selecting ? undefined : (model) => model.name,
        cell: (model) => {
          const secondary =
            model.provider_model_id ||
            model.catalog ||
            (deployments.get(model.name) ? `→ ${deployments.get(model.name)}` : '')
          return (
            <div className="flex min-w-0 max-w-[340px] items-center gap-3">
              {isReadonly ? (
                <ModelLogo model={model} size={16} framed catalog={catalog} />
              ) : (
                <ModelSelectLogo
                  model={model}
                  catalog={catalog}
                  selected={selectedKeys.has(model.name)}
                  selecting={selecting}
                  blocker={getDeleteBlocker(model.name)}
                  onToggle={(checked) =>
                    onSelectedKeysChange(updatePageSelection(selectedKeys, [model.name], checked))
                  }
                />
              )}
              <div className="flex min-w-0 flex-col">
                <span className="flex min-w-0 items-center gap-1.5">
                  <span className="truncate font-medium" title={model.name}>
                    {model.name}
                  </span>
                  {model.name === defaultModel ? (
                    <Badge size="sm" tone="brand">
                      default
                    </Badge>
                  ) : null}
                </span>
                {secondary ? (
                  <span
                    className="truncate font-mono text-[11.5px] text-subtle-foreground"
                    title={secondary}
                  >
                    {secondary}
                  </span>
                ) : null}
              </div>
            </div>
          )
        },
      },
      {
        id: 'served',
        header: 'Served by',
        ...hideWhenNarrow('served', showTraffic),
        cell: (model) => {
          const label = modelServedBy(model, deployments.get(model.name), catalog)
          return label ? (
            <span className="block max-w-[200px] truncate text-muted-foreground" title={label}>
              {label}
            </span>
          ) : (
            <span className="text-subtle-foreground">No backend</span>
          )
        },
      },
      {
        id: 'kind',
        header: 'Kind',
        ...hideWhenNarrow('kind', showTraffic),
        sortValue: (model) => MODEL_KIND_LABELS[kinds.get(model.name) ?? 'chat'],
        cell: (model) => (
          <Badge tone="outline">{MODEL_KIND_LABELS[kinds.get(model.name) ?? 'chat']}</Badge>
        ),
      },
      {
        id: 'price',
        header: 'Price / 1M',
        align: 'right',
        ...hideWhenNarrow('price', showTraffic),
        sortValue: (model) =>
          (model.pricing ?? catalogPrices?.[model.name])?.completion_per_1m ?? -1,
        cell: (model) => {
          const catalogPrice = model.pricing ? undefined : catalogPrices?.[model.name]
          const price = modelPriceSummary(model.pricing ?? catalogPrice)
          if (!price.input && !price.output)
            return <span className="text-subtle-foreground">—</span>
          return (
            <span
              className="whitespace-nowrap font-mono text-xs tabular text-muted-foreground"
              title={`Input / output price per one million tokens${catalogPrice ? ', the catalog list price' : ''}`}
            >
              {price.input ?? '—'} <span className="text-subtle-foreground">/</span>{' '}
              {price.output ?? '—'}
              {catalogPrice ? (
                <span className="block text-[11px] text-subtle-foreground">catalog list price</span>
              ) : null}
            </span>
          )
        },
      },
      {
        id: 'decisions',
        header: 'Decisions',
        align: 'right',
        ...hideWhenNarrow('decisions', showTraffic),
        sortValue: (model) => referenceCounts.get(model.name) ?? 0,
        cell: (model) => {
          const references = referenceCounts.get(model.name) ?? 0
          if (!references)
            return <span className="font-mono text-xs tabular text-subtle-foreground">0</span>
          return (
            <span {...stopRowClick}>
              <button
                type="button"
                onClick={() => onShowUsage(model)}
                aria-label={`${references} ${references === 1 ? 'decision uses' : 'decisions use'} ${model.name}`}
                className="rounded px-1 font-mono text-xs tabular text-foreground underline-offset-4 transition-colors hover:text-primary hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
              >
                {references}
              </button>
            </span>
          )
        },
      },
    ]
    if (canCheckHealth) {
      list.splice(3, 0, {
        id: 'health',
        header: 'Health',
        // Fixed, so a status changing under a check never shifts the columns after it.
        width: '132px',
        className: 'whitespace-nowrap',
        sortValue: (model) => healthSortValue(health.byModel.get(model.name)),
        cell: (model) => (
          <ModelHealthCell
            health={health.byModel.get(model.name)}
            phase={health.phase}
            issue={health.issue}
            checking={health.checking || health.checkingModels.has(model.name)}
          />
        ),
      })
    }
    if (showTraffic) {
      list.push({
        id: 'traffic',
        header: 'Traffic',
        align: 'right',
        width: '128px',
        sortValue: (model) => shareOf(model) ?? -1,
        cell: (model) => {
          const share = shareOf(model)
          if (share === undefined) return <span className="text-subtle-foreground">—</span>
          return (
            <div className="ml-auto flex w-24 flex-col items-end gap-1">
              <span className="font-mono text-xs tabular">{share.toFixed(1)}%</span>
              <MiniBar value={share} max={maxShare} tone="green" className="h-1" />
            </div>
          )
        },
      })
    }
    list.push({
      id: 'actions',
      width: '48px',
      header: <span className="sr-only">Actions</span>,
      cell: (model) => (
        <div className="flex justify-end" {...stopRowClick}>
          <DropdownMenu modal={false}>
            <DropdownMenuTrigger asChild>
              <Button
                type="button"
                variant="ghost"
                size="icon-xs"
                aria-label={`Actions for ${model.name}`}
                className="text-subtle-foreground group-hover:text-foreground data-[state=open]:bg-accent data-[state=open]:text-foreground"
              >
                <MoreHorizontal />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-48">
              <DropdownMenuItem onSelect={() => onOpen(model)}>
                <Eye />
                View details
              </DropdownMenuItem>
              {isReadonly ? null : (
                <DropdownMenuItem onSelect={() => onEdit(model)}>
                  <Pencil />
                  Edit
                </DropdownMenuItem>
              )}
              {canCheckHealth ? (
                <DropdownMenuItem
                  disabled={health.checking || health.checkingModels.has(model.name)}
                  onSelect={() => onCheckHealth(model)}
                >
                  <Activity />
                  Check health
                </DropdownMenuItem>
              ) : null}
              {isReadonly ? null : (
                <DropdownMenuItem
                  disabled={!canBeDefault(model)}
                  onSelect={() => onMakeDefault(model)}
                >
                  <Star />
                  Make default
                </DropdownMenuItem>
              )}
              <DropdownMenuItem
                onSelect={() => {
                  void navigator.clipboard?.writeText(model.name)
                  toast.success('Model name copied')
                }}
              >
                <Copy />
                Copy name
              </DropdownMenuItem>
              {isReadonly ? null : (
                <>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem variant="destructive" onSelect={() => onDelete(model)}>
                    <Trash2 />
                    Delete
                  </DropdownMenuItem>
                </>
              )}
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      ),
    })
    return list
  }, [
    canBeDefault,
    canCheckHealth,
    catalog,
    catalogPrices,
    defaultModel,
    deployments,
    getDeleteBlocker,
    health,
    isReadonly,
    kinds,
    onCheckHealth,
    onDelete,
    onEdit,
    onMakeDefault,
    onOpen,
    onSelectedKeysChange,
    onShowUsage,
    referenceCounts,
    selectablePageKeys,
    selectedKeys,
    selectedOnPage,
    selecting,
    traffic,
  ])

  return (
    <div className="@container">
      <DataTable
        data={rows}
        columns={columns}
        getRowId={(model) => model.name}
        selected={selectedKeys as Set<string>}
        activeRowId={activeName ?? undefined}
        onRowClick={onOpen}
        empty={
          totalModels === 0 ? (
            <EmptyState
              compact
              icon={Boxes}
              tone="green"
              title="No backend models yet"
              description="Connect a provider API or a self-hosted runtime so decisions can send it traffic."
              actions={
                isReadonly ? undefined : (
                  <Button size="sm" onClick={onConnect}>
                    <Plus />
                    Connect model
                  </Button>
                )
              }
            />
          ) : (
            <EmptyState
              compact
              icon={Boxes}
              tone="green"
              title="No models match"
              description={
                filtersActive
                  ? 'Try another search or clear the filters.'
                  : 'No models are configured.'
              }
              actions={
                filtersActive ? (
                  <Button variant="outline" size="sm" onClick={onClearFilters}>
                    Clear filters
                  </Button>
                ) : undefined
              }
            />
          )
        }
      />
      <ConfigPageTablePager
        itemLabel="models"
        page={pageWindow.page}
        totalPages={pageWindow.totalPages}
        total={models.length}
        start={pageWindow.start}
        end={pageWindow.end}
        onPageChange={setPage}
        pageSize={pageSize}
        pageSizeOptions={PAGE_SIZES}
        onPageSizeChange={(next) => {
          setPageSize(next)
          setPage(1)
        }}
      />
    </div>
  )
}
