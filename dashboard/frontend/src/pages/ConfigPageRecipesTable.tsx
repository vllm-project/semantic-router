import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Braces,
  Eye,
  MoreHorizontal,
  Network,
  Package,
  Pencil,
  Plus,
  Power,
  RefreshCw,
  Trash2,
} from 'lucide-react'

import {
  Alert,
  Badge,
  Button,
  DataTable,
  DataToolbar,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
  EmptyState,
  FilterPill,
  IconTile,
  StatusLabel,
  Switch,
  Tooltip,
  type Column,
} from '../ui'
import {
  filterRecipeRows,
  recipeRowStatus,
  recipeStatusLabel,
  type BuiltInRecipeRow,
  type ConfigRecipeRow,
  type PackageRecipeRow,
  type RecipeRow,
} from './configPageRecipesPageSupport'

interface Props {
  rows: RecipeRow[]
  activeId: string | null
  pendingRestart: ReadonlySet<string>
  isReadonly: boolean
  canDeploy: boolean
  builtInsFailed: boolean
  builtInsLoading: boolean
  packagesError: string | null
  onRetryBuiltIns: () => void
  onRetryPackages: () => void
  onOpen: (row: RecipeRow) => void
  onCreateRecipe: () => void
  onEdit: (row: ConfigRecipeRow) => void
  onDelete: (row: ConfigRecipeRow) => void
  onNewPublicModel: (row: ConfigRecipeRow | BuiltInRecipeRow) => void
  onToggle: (row: PackageRecipeRow, mode: 'activate' | 'deactivate') => void
}

const plural = (count: number, noun: string) => `${noun}${count === 1 ? '' : 's'}`

const SOURCE_LABEL: Record<RecipeRow['kind'], string> = {
  config: 'config.yaml',
  package: 'Package',
  builtin: 'Built-in',
}

function RowMenu({
  row,
  isReadonly,
  canDeploy,
  pendingRestart,
  onOpen,
  onEdit,
  onDelete,
  onNewPublicModel,
  onToggle,
}: Pick<
  Props,
  | 'isReadonly'
  | 'canDeploy'
  | 'pendingRestart'
  | 'onOpen'
  | 'onEdit'
  | 'onDelete'
  | 'onNewPublicModel'
  | 'onToggle'
> & { row: RecipeRow }) {
  const navigate = useNavigate()
  const status = recipeRowStatus(row, pendingRestart)
  return (
    <span
      className="flex justify-end"
      role="presentation"
      onClick={(event) => event.stopPropagation()}
      onKeyDown={(event) => event.stopPropagation()}
    >
      <DropdownMenu modal={false}>
        <DropdownMenuTrigger asChild>
          <Button
            variant="ghost"
            size="icon-xs"
            aria-label={`Actions for ${row.name}`}
            className="text-subtle-foreground group-hover:text-foreground data-[state=open]:bg-accent"
          >
            <MoreHorizontal />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-52">
          <DropdownMenuItem onSelect={() => onOpen(row)}>
            <Eye />
            Open details
          </DropdownMenuItem>
          {row.kind === 'config' ? (
            <>
              {isReadonly ? null : (
                <DropdownMenuItem onSelect={() => onNewPublicModel(row)}>
                  <Plus />
                  New public model
                </DropdownMenuItem>
              )}
              <DropdownMenuItem
                onSelect={() => navigate(`/routing/topology?scope=${encodeURIComponent(row.id)}`)}
              >
                <Network />
                View topology
              </DropdownMenuItem>
              <DropdownMenuItem onSelect={() => navigate('/routing/editor')}>
                <Braces />
                Open in config editor
              </DropdownMenuItem>
              {!isReadonly && !row.synthetic ? (
                <DropdownMenuItem onSelect={() => onEdit(row)}>
                  <Pencil />
                  Edit
                </DropdownMenuItem>
              ) : null}
              {!isReadonly && !row.synthetic && row.id !== 'default' ? (
                <>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem variant="destructive" onSelect={() => onDelete(row)}>
                    <Trash2 />
                    Delete
                  </DropdownMenuItem>
                </>
              ) : null}
            </>
          ) : null}
          {row.kind === 'package' && !isReadonly ? (
            <>
              <DropdownMenuSeparator />
              <DropdownMenuItem
                variant={row.summary.active ? 'destructive' : 'default'}
                disabled={!canDeploy || status === 'pending'}
                onSelect={() => onToggle(row, row.summary.active ? 'deactivate' : 'activate')}
              >
                <Power />
                {row.summary.active ? 'Deactivate' : 'Activate'}
              </DropdownMenuItem>
            </>
          ) : null}
          {row.kind === 'builtin' && !isReadonly ? (
            <DropdownMenuItem onSelect={() => onNewPublicModel(row)}>
              <Plus />
              Create public model
            </DropdownMenuItem>
          ) : null}
        </DropdownMenuContent>
      </DropdownMenu>
    </span>
  )
}

export default function ConfigPageRecipesTable(props: Props) {
  const {
    rows,
    activeId,
    pendingRestart,
    isReadonly,
    canDeploy,
    builtInsFailed,
    builtInsLoading,
    packagesError,
    onRetryBuiltIns,
    onRetryPackages,
    onOpen,
    onCreateRecipe,
    onToggle,
  } = props
  const [search, setSearch] = useState('')
  const [sources, setSources] = useState<string[]>([])
  const [statuses, setStatuses] = useState<string[]>([])
  const visible = filterRecipeRows(rows, search, sources, statuses, pendingRestart)
  const counts = (predicate: (row: RecipeRow) => boolean) => rows.filter(predicate).length
  const sourceOptions = (['config', 'package', 'builtin'] as const)
    .map((kind) => ({
      value: kind,
      label: SOURCE_LABEL[kind],
      count: counts((row) => row.kind === kind),
    }))
    .filter((option) => option.count > 0)
  const statusOptions = (['loaded', 'active', 'inactive', 'pending', 'available'] as const)
    .map((status) => ({
      value: status,
      label: recipeStatusLabel[status].label,
      count: counts((row) => recipeRowStatus(row, pendingRestart) === status),
    }))
    .filter((option) => option.count > 0)
  const running = counts((row) => {
    const status = recipeRowStatus(row, pendingRestart)
    return status === 'loaded' || status === 'active'
  })

  const columns = useMemo<Column<RecipeRow>[]>(
    () => [
      {
        id: 'name',
        header: 'Recipe',
        sortValue: (row) => row.name,
        cell: (row) => (
          <div className="flex min-w-0 items-center gap-3">
            <IconTile icon={Package} tone="blue" size="sm" />
            <div className="flex min-w-0 flex-col">
              <span className="flex min-w-0 items-center gap-1.5">
                <span className="truncate font-mono text-[13px] font-medium">{row.name}</span>
                {row.kind === 'config' && row.id === 'default' ? (
                  <Badge size="sm" tone="brand">
                    default
                  </Badge>
                ) : null}
                {row.kind !== 'config' ? (
                  <Badge size="sm" tone="outline">
                    {row.kind === 'builtin' ? 'built-in' : 'package'}
                  </Badge>
                ) : null}
              </span>
              <span className="max-w-[300px] truncate text-xs text-subtle-foreground">
                {row.description || 'No description'}
              </span>
            </div>
          </div>
        ),
      },
      {
        id: 'models',
        header: 'Public models',
        hideBelow: 'xl',
        cell: (row) => {
          if (row.kind === 'config') {
            const [first, ...more] = row.publicModels
            return first ? (
              <span
                className="flex max-w-[260px] items-center gap-1.5"
                title={row.publicModels.join(', ')}
              >
                <span className="truncate font-mono text-xs">{first}</span>
                {more.length ? (
                  <Badge size="sm" tone="neutral" className="tabular">
                    +{more.length}
                  </Badge>
                ) : null}
              </span>
            ) : (
              <span className="text-xs text-subtle-foreground">None yet</span>
            )
          }
          if (row.kind === 'package')
            return (
              <span className="text-xs tabular text-muted-foreground">
                {row.summary.counts.unified_models} in the package
              </span>
            )
          return <span className="text-xs text-subtle-foreground">None yet</span>
        },
      },
      {
        id: 'contents',
        header: 'Contents',
        hideBelow: 'xl',
        sortValue: (row) =>
          row.kind === 'config'
            ? row.decisions
            : row.kind === 'package'
              ? row.summary.counts.decisions
              : row.recipe.decisions.length,
        cell: (row) => {
          const [decisions, detail] =
            row.kind === 'config'
              ? [row.decisions, `${row.signals} ${plural(row.signals, 'signal')}`]
              : row.kind === 'package'
                ? [row.summary.counts.decisions, `${row.summary.counts.probes} probes`]
                : [row.recipe.decisions.length, '']
          return (
            <span className="whitespace-nowrap text-xs text-muted-foreground">
              <span className="font-medium tabular text-foreground">{decisions}</span>{' '}
              {plural(decisions, 'decision')}
              {detail ? <span className="tabular"> · {detail}</span> : null}
            </span>
          )
        },
      },
      {
        id: 'source',
        header: 'Source',
        hideBelow: '2xl',
        cell: (row) => (
          <span className="text-xs text-muted-foreground">{SOURCE_LABEL[row.kind]}</span>
        ),
      },
      {
        id: 'status',
        header: 'Status',
        width: '168px',
        cell: (row) => {
          const status = recipeRowStatus(row, pendingRestart)
          const label = recipeStatusLabel[status]
          if (row.kind === 'package') {
            const blocked = isReadonly || !canDeploy || status === 'pending'
            const tip =
              status === 'pending'
                ? 'Run vllm-sr serve to apply'
                : isReadonly || !canDeploy
                  ? 'Activating a recipe needs the config.deploy permission'
                  : row.summary.active
                    ? `Deactivate ${row.name}`
                    : `Activate ${row.name}`
            return (
              <span
                className="flex items-center gap-2.5"
                role="presentation"
                onClick={(event) => event.stopPropagation()}
                onKeyDown={(event) => event.stopPropagation()}
              >
                <Tooltip content={tip}>
                  <span>
                    <Switch
                      size="sm"
                      checked={status !== 'inactive'}
                      disabled={blocked}
                      aria-label={
                        row.summary.active ? `Deactivate ${row.name}` : `Activate ${row.name}`
                      }
                      onCheckedChange={(checked) =>
                        onToggle(row, checked ? 'activate' : 'deactivate')
                      }
                    />
                  </span>
                </Tooltip>
                <StatusLabel tone={label.tone}>{label.label}</StatusLabel>
              </span>
            )
          }
          return (
            <Tooltip
              content={
                status === 'loaded'
                  ? 'Part of the running configuration; public models resolve to it.'
                  : 'Publish it as a public model to use it.'
              }
            >
              <span className="inline-flex">
                <StatusLabel tone={label.tone}>{label.label}</StatusLabel>
              </span>
            </Tooltip>
          )
        },
      },
      {
        id: 'actions',
        width: '48px',
        header: <span className="sr-only">Actions</span>,
        cell: (row) => <RowMenu row={row} {...props} />,
      },
    ],
    [canDeploy, isReadonly, onToggle, pendingRestart, props],
  )

  return (
    <div className="flex flex-col gap-3">
      {packagesError ? (
        <Alert
          tone="warning"
          title="Recipe packages could not be listed"
          actions={
            <Button variant="outline" size="sm" onClick={onRetryPackages}>
              <RefreshCw />
              Retry
            </Button>
          }
        >
          {packagesError}
        </Alert>
      ) : null}
      <div>
        <DataToolbar
          search={search}
          onSearchChange={setSearch}
          placeholder="Search recipes…"
          filters={
            <>
              {sourceOptions.length > 1 ? (
                <FilterPill
                  label="Source"
                  selected={sources}
                  onChange={setSources}
                  options={sourceOptions}
                />
              ) : null}
              <FilterPill
                label="Status"
                selected={statuses}
                onChange={setStatuses}
                options={statusOptions}
              />
            </>
          }
          actions={
            <>
              <span className="text-xs text-muted-foreground">
                <span className="tabular">{running}</span> of{' '}
                <span className="tabular">{rows.length}</span> running
                {builtInsLoading ? ' · loading built-in recipes…' : ''}
              </span>
              {isReadonly ? null : (
                <Button variant="outline" size="sm" onClick={onCreateRecipe}>
                  <Plus />
                  New recipe
                </Button>
              )}
            </>
          }
        />
        <DataTable
          data={visible}
          columns={columns}
          getRowId={(row) => row.id}
          activeRowId={activeId ?? undefined}
          onRowClick={onOpen}
          empty={
            <EmptyState
              compact
              icon={Package}
              tone="blue"
              title="No recipes match"
              description="Try another search or clear the filters."
              actions={
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    setSearch('')
                    setSources([])
                    setStatuses([])
                  }}
                >
                  Clear filters
                </Button>
              }
            />
          }
        />
        {builtInsFailed ? (
          <p className="mt-2 flex flex-wrap items-center gap-x-1 px-1 text-xs text-subtle-foreground">
            Built-in recipes could not be read from the installed vLLM-SR package; recipes in
            config.yaml are unaffected.
            <Button variant="link" size="xs" className="h-auto px-0" onClick={onRetryBuiltIns}>
              Try again
            </Button>
          </p>
        ) : null}
      </div>
    </div>
  )
}
