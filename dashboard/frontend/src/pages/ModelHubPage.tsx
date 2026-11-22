import { useMemo, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  ArrowUpDown,
  Boxes,
  Building2,
  Cloud,
  LibraryBig,
  LockOpen,
  Plus,
  Route,
  Server,
} from 'lucide-react'
import useBuiltInModelCatalog from '../hooks/useBuiltInModelCatalog'
import {
  Alert,
  Badge,
  Button,
  Card,
  DataToolbar,
  EmptyState,
  FilterPill,
  IconTile,
  PageContainer,
  PageHeader,
  ProviderLogo,
  ProviderStack,
  Segmented,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  Skeleton,
  StatCard,
  cn,
} from '../ui'
import { HubLogo, ModelHubSheet, type ModelHubSheetView } from './ModelHubSheet'
import {
  MODEL_HUB_DEFAULT_FILTERS,
  modelHubBackendLink,
  modelHubCapabilities,
  modelHubContextLabel,
  modelHubCreators,
  modelHubDistributionLabel,
  modelHubMaxOutputLabel,
  modelHubParameterLabel,
  modelHubProviders,
  modelHubRows,
  modelHubStats,
  readableModelHubValue,
  type ModelHubFilters,
  type ModelHubRow,
  type ModelHubSort,
} from './modelHubSupport'

const PAGE = 24
const VIEWS: readonly ModelHubSheetView[] = ['overview', 'providers', 'evidence', 'pool']
const DISTRIBUTION_ICONS = { open_weights: LockOpen, proprietary_api: Cloud, router_recipe: Route }

function ModelCard({
  row,
  index,
  onOpen,
  onAdd,
}: {
  row: ModelHubRow
  index: number
  onOpen: () => void
  onAdd: () => void
}) {
  const model = row.model
  const DistributionIcon = DISTRIBUTION_ICONS[model.distribution.type]
  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: Math.min(index, 9) * 0.025, duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
    >
      <Card
        interactive
        role="button"
        tabIndex={0}
        aria-label={`Open ${model.display_name}`}
        onClick={onOpen}
        onKeyDown={(event) => {
          if (event.target !== event.currentTarget) return
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault()
            onOpen()
          }
        }}
        className="h-full gap-0 p-4 outline-none focus-visible:ring-[3px] focus-visible:ring-ring/40"
      >
        <div className="flex items-start gap-3">
          <HubLogo presentation={model.presentation} size={18} />
          <div className="flex min-w-0 flex-1 flex-col">
            <span className="truncate text-sm font-semibold tracking-[-0.01em]">
              {model.display_name}
            </span>
            <span className="truncate text-xs text-subtle-foreground">
              {model.publisher}
              {model.released_at ? ` · ${model.released_at}` : ''}
            </span>
          </div>
          <Badge
            tone={model.distribution.type === 'open_weights' ? 'outline' : 'neutral'}
            size="sm"
          >
            <DistributionIcon />
            {modelHubDistributionLabel[model.distribution.type]}
          </Badge>
        </div>
        <p className="mt-3 line-clamp-2 min-h-[2.6em] text-xs leading-relaxed text-muted-foreground">
          {model.description}
        </p>
        <dl className="mt-4 grid grid-cols-3 gap-2 rounded-lg border border-border bg-surface-subtle/60 px-3 py-2">
          {[
            ['Context', modelHubContextLabel(model)],
            ['Max output', modelHubMaxOutputLabel(model)],
            ['Parameters', modelHubParameterLabel(model)],
          ].map(([label, value]) => (
            <div key={label} className="flex min-w-0 flex-col gap-0.5">
              <dt className="text-[11px] text-subtle-foreground">{label}</dt>
              <dd className="truncate font-mono text-[12.5px] font-medium tabular" title={value}>
                {value}
              </dd>
            </div>
          ))}
        </dl>
        <div className="mt-3 flex flex-wrap gap-1">
          {model.capabilities.slice(0, 5).map((capability) => (
            <span
              key={capability}
              className="inline-flex h-5 items-center rounded-[5px] border border-border bg-surface px-1.5 text-[11px] text-muted-foreground"
            >
              {readableModelHubValue(capability)}
            </span>
          ))}
          {model.capabilities.length > 5 ? (
            <span className="inline-flex h-5 items-center px-1 text-[11px] text-subtle-foreground">
              +{model.capabilities.length - 5}
            </span>
          ) : null}
        </div>
        <div className="mt-auto flex items-center gap-2 pt-4">
          {row.providers.length ? (
            <span
              className="flex min-w-0 items-center gap-1.5"
              title={row.providers.map(({ provider }) => provider.display_name).join(', ')}
            >
              <ProviderStack
                providers={row.providers.map(({ provider }) =>
                  provider.presentation.logo.replace(/^package:/, ''),
                )}
                size={16}
                max={4}
              />
              <span className="truncate text-[11px] text-subtle-foreground">
                {row.providers.length} {row.providers.length === 1 ? 'provider' : 'providers'}
              </span>
            </span>
          ) : (
            <span className="text-[11px] text-subtle-foreground">
              {model.kind === 'virtual'
                ? 'Built from backend roles'
                : 'No serving provider cataloged'}
            </span>
          )}
          {model.kind === 'physical' ? (
            <Button
              variant="outline"
              size="xs"
              className="ml-auto"
              onClick={(event) => {
                event.stopPropagation()
                onAdd()
              }}
            >
              <Plus />
              Add as backend
            </Button>
          ) : null}
        </div>
      </Card>
    </motion.div>
  )
}

export default function ModelHubPage() {
  const navigate = useNavigate()
  const { catalog, error, loading, ready, source, retry } = useBuiltInModelCatalog()
  const [params, setParams] = useSearchParams()
  const [filters, setFilters] = useState<ModelHubFilters>(MODEL_HUB_DEFAULT_FILTERS)
  const [limit, setLimit] = useState(PAGE)
  const rows = useMemo(() => modelHubRows(catalog, filters), [catalog, filters])
  const stats = useMemo(() => modelHubStats(catalog), [catalog])
  const creators = useMemo(() => {
    const logos = new Map(
      catalog.models.map((model) => [
        model.publisher,
        model.presentation.logo.replace(/^package:/, ''),
      ]),
    )
    return modelHubCreators(catalog).map((name) => ({
      value: name,
      label: name,
      icon: <ProviderLogo provider={logos.get(name) ?? name} size={14} />,
      count: catalog.models.filter((model) => model.publisher === name).length,
    }))
  }, [catalog])
  const providers = useMemo(
    () =>
      modelHubProviders(catalog).map((provider) => ({
        value: provider.id,
        label: provider.display_name,
        icon: (
          <ProviderLogo provider={provider.presentation.logo.replace(/^package:/, '')} size={14} />
        ),
        count: provider.models?.length,
      })),
    [catalog],
  )
  const capabilities = useMemo(() => modelHubCapabilities(catalog), [catalog])
  const allRows = useMemo(
    () => modelHubRows(catalog, { ...MODEL_HUB_DEFAULT_FILTERS, lifecycle: 'all' }),
    [catalog],
  )
  const activeId = params.get('model')
  const active = activeId ? allRows.find((row) => row.model.id === activeId) : undefined
  const requestedView = params.get('view') as ModelHubSheetView | null
  const view: ModelHubSheetView =
    requestedView && VIEWS.includes(requestedView) ? requestedView : 'overview'
  const update = (patch: Partial<ModelHubFilters>) => {
    setFilters((current) => ({ ...current, ...patch }))
    setLimit(PAGE)
  }
  const open = (id: string, next: ModelHubSheetView = 'overview') =>
    setParams(next === 'overview' ? { model: id } : { model: id, view: next }, {
      replace: Boolean(activeId),
    })
  const filtered =
    filters.query ||
    filters.distribution !== 'all' ||
    filters.lifecycle !== 'supported' ||
    filters.publishers.length ||
    filters.providers.length ||
    filters.capabilities.length

  return (
    <PageContainer>
      <PageHeader
        icon={<IconTile icon={LibraryBig} tone="green" size="lg" />}
        title="Model Hub"
        description="Browse open-weight and API models with their context, protocols and providers, then add one as a backend."
        actions={
          <Button variant="outline" onClick={() => navigate('/models/backends')}>
            <Boxes />
            Your backends
          </Button>
        }
      />
      <div data-testid="model-hub-page" className="flex flex-col gap-6">
        {error ? (
          <Alert
            tone={ready ? 'warning' : 'danger'}
            title={
              !ready
                ? 'Model catalog details are unavailable.'
                : source === 'bundled'
                  ? 'Showing the catalog bundled with this Dashboard.'
                  : 'Showing the last catalog loaded from the server.'
            }
            actions={
              <Button
                variant="outline"
                size="sm"
                onClick={retry}
                disabled={loading}
                loading={loading}
              >
                {loading ? 'Retrying…' : 'Retry'}
              </Button>
            }
          >
            Server catalog could not be refreshed. {error}
          </Alert>
        ) : null}
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          <StatCard
            label="Models"
            icon={<Boxes className="size-4 text-chart-6" />}
            value={stats.models}
            hint={`${catalog.models.filter((model) => model.kind === 'virtual').length} router recipes`}
          />
          <StatCard
            label="Creators"
            icon={<Building2 className="size-4 text-chart-6" />}
            value={stats.creators}
            hint="Labs that publish weights or an API"
          />
          <StatCard
            label="Providers"
            icon={<Server className="size-4 text-chart-6" />}
            value={stats.providers}
            hint="APIs and hosts that serve them"
          />
        </div>
        <div className="flex flex-col gap-3">
          <DataToolbar
            className="mb-0"
            search={filters.query}
            onSearchChange={(query) => update({ query })}
            placeholder="Search models, creators or tags…"
            filters={
              <>
                <Segmented
                  aria-label="Distribution"
                  value={filters.distribution}
                  onValueChange={(distribution) => update({ distribution })}
                  options={[
                    { value: 'all', label: 'All' },
                    { value: 'open_weights', label: 'Open weights' },
                    { value: 'proprietary_api', label: 'API' },
                    { value: 'router_recipe', label: 'Recipes' },
                  ]}
                />
                <FilterPill
                  label="Creator"
                  selected={filters.publishers}
                  onChange={(publishers) => update({ publishers })}
                  options={creators}
                />
                <FilterPill
                  label="Provider"
                  selected={filters.providers}
                  onChange={(next) => update({ providers: next })}
                  options={providers}
                />
                <FilterPill
                  label="Capability"
                  selected={filters.capabilities}
                  onChange={(next) => update({ capabilities: next })}
                  options={capabilities.map((capability) => ({
                    value: capability,
                    label: readableModelHubValue(capability),
                    count: catalog.models.filter((model) => model.capabilities.includes(capability))
                      .length,
                  }))}
                />
              </>
            }
            actions={
              <>
                <Select
                  value={filters.lifecycle}
                  onValueChange={(lifecycle) =>
                    update({ lifecycle: lifecycle as ModelHubFilters['lifecycle'] })
                  }
                >
                  <SelectTrigger size="sm" aria-label="Lifecycle">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent align="end">
                    <SelectItem value="supported">Supported</SelectItem>
                    <SelectItem value="active">Active</SelectItem>
                    <SelectItem value="experimental">Experimental</SelectItem>
                    <SelectItem value="deprecated">Deprecated</SelectItem>
                    <SelectItem value="all">All lifecycles</SelectItem>
                  </SelectContent>
                </Select>
                <Select
                  value={filters.sort}
                  onValueChange={(sort) => update({ sort: sort as ModelHubSort })}
                >
                  <SelectTrigger size="sm" className="gap-2" aria-label="Sort">
                    <ArrowUpDown className="size-3.5 text-subtle-foreground" />
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent align="end">
                    <SelectItem value="released">Newest first</SelectItem>
                    <SelectItem value="name">Name</SelectItem>
                    <SelectItem value="context">Largest context</SelectItem>
                    <SelectItem value="providers">Most providers</SelectItem>
                  </SelectContent>
                </Select>
              </>
            }
          />
          <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
            <span className="tabular">
              {rows.length} of {catalog.models.length} models
            </span>
            {!ready && loading ? (
              <span
                role="status"
                className="inline-flex items-center gap-1.5 text-subtle-foreground"
              >
                <Skeleton className="size-2 rounded-full" />
                Loading model evaluations…
              </span>
            ) : null}
            {filtered ? (
              <Button
                variant="link"
                size="xs"
                className="h-auto text-xs"
                onClick={() => update(MODEL_HUB_DEFAULT_FILTERS)}
              >
                Clear filters
              </Button>
            ) : null}
          </div>
          {rows.length ? (
            <>
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
                {rows.slice(0, limit).map((row, index) => (
                  <ModelCard
                    key={row.model.id}
                    row={row}
                    index={index}
                    onOpen={() => open(row.model.id)}
                    onAdd={() => navigate(modelHubBackendLink(row.model.id))}
                  />
                ))}
              </div>
              {rows.length > limit ? (
                <div className="flex justify-center">
                  <Button variant="outline" onClick={() => setLimit((current) => current + PAGE)}>
                    Show {Math.min(PAGE, rows.length - limit)} more
                    <span className="text-subtle-foreground">· {rows.length - limit} left</span>
                  </Button>
                </div>
              ) : null}
            </>
          ) : (
            <div className={cn('rounded-xl border border-dashed border-border-strong')}>
              <EmptyState
                compact
                icon={LibraryBig}
                tone="green"
                title="No models match these filters"
                description="Drop a requirement or clear the filters to see the whole catalog."
                actions={
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => update(MODEL_HUB_DEFAULT_FILTERS)}
                  >
                    Clear filters
                  </Button>
                }
              />
            </div>
          )}
        </div>
      </div>
      <ModelHubSheet
        row={active}
        catalog={catalog}
        evidenceReady={ready}
        view={view}
        onViewChange={(next) => activeId && open(activeId, next)}
        onOpenChange={(value) => (value ? undefined : setParams({}, { replace: true }))}
      />
    </PageContainer>
  )
}
