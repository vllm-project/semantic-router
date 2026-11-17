import type { LucideIcon } from 'lucide-react'
import { Link, useNavigate } from 'react-router-dom'
import {
  ArrowRight,
  Boxes,
  ChevronRight,
  FileCode2,
  FlaskConical,
  Layers,
  MessagesSquare,
  Radar,
  RefreshCw,
  Rocket,
  ServerOff,
  Split,
  Waypoints,
} from 'lucide-react'

import {
  Badge,
  Button,
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  CopyButton,
  EmptyState,
  IconTile,
  MiniBar,
  ProviderLogo,
  Skeleton,
  StatusDot,
  Tooltip,
  cn,
  type Tone,
} from '../ui'
import type { SystemStatus } from '../utils/routerRuntime'
import { listRoutingScopes, type RoutingScopedConfigLike } from '../utils/routingScopes'
import { buildDecisionPreviewRows } from './dashboardPageOverview'
import {
  categorizeDecisions,
  countDecisions,
  countModels,
  countPlugins,
  countSignals,
} from './dashboardPageStats'
import type { RouterConfig } from './dashboardPageTypes'
import { HOME_RANGE_LABEL, topShares, type HomeRange } from './homeSupport'
import { modelProviderId } from './insightsConfigFacts'
import { modelLabel, useModelLabels } from './insightsModelLabelSupport'
import type { InsightsAggregateResponse } from './insightsPageTypes'
import { formatInsightsDecisionName } from './insightsPageSupport'
import { listPublicModels } from './startSupport'

// ---------------------------------------------------------------- config tiles

interface CountTile {
  id: string
  label: string
  count?: number
  hint: string
  icon: LucideIcon
  tone: Tone
  to: string
}

/** The routing configuration at a glance; each tile opens its manager. */
export function HomeConfigTiles({
  config,
  loading,
}: {
  config: RouterConfig | null
  loading: boolean
}) {
  const scoped = config as (RouterConfig & RoutingScopedConfigLike) | null
  const recipes = config ? listRoutingScopes(scoped).length : undefined
  const models = listPublicModels(config)
  const chatModels = models.filter((model) => model.api === 'chat').length
  const systemOneModels = models.length - chatModels
  const signals = config ? countSignals(config) : undefined
  const signalTypes = signals ? Object.keys(signals.byType).length : 0
  const plugins = config ? countPlugins(config) : 0
  const defaultModel =
    typeof config?.providers?.defaults === 'object' && config.providers.defaults
      ? (config.providers.defaults as { model?: unknown }).model
      : undefined
  const tiles: CountTile[] = [
    {
      id: 'recipes',
      label: 'Recipes',
      count: recipes,
      hint: `${chatModels} public model${chatModels === 1 ? '' : 's'}${systemOneModels ? ` + ${systemOneModels} System One` : ''}`,
      icon: Layers,
      tone: 'blue',
      to: '/routing/recipes',
    },
    {
      id: 'decisions',
      label: 'Decisions',
      count: config ? countDecisions(config) : undefined,
      hint: `${plugins} plugin${plugins === 1 ? '' : 's'} attached`,
      icon: Split,
      tone: 'violet',
      to: '/routing/decisions',
    },
    {
      id: 'signals',
      label: 'Signals',
      count: signals?.total,
      hint: `${signalTypes} signal type${signalTypes === 1 ? '' : 's'}`,
      icon: Radar,
      tone: 'gold',
      to: '/routing/signals',
    },
    {
      id: 'models',
      label: 'Backend models',
      count: config ? countModels(config) : undefined,
      hint: typeof defaultModel === 'string' ? `Default ${defaultModel}` : 'No default model',
      icon: Boxes,
      tone: 'green',
      to: '/models/backends',
    },
  ]
  return (
    <nav aria-label="Routing configuration" className="grid grid-cols-2 gap-3 xl:grid-cols-4">
      {tiles.map((tile) => (
        <Link
          key={tile.id}
          to={tile.to}
          data-testid={`home-count-${tile.id}`}
          className="group flex min-w-0 items-center gap-3 rounded-xl border border-border bg-card px-3.5 py-3 shadow-xs outline-none transition-[border-color,box-shadow,transform] duration-200 hover:-translate-y-px hover:border-border-strong hover:shadow-md focus-visible:ring-[3px] focus-visible:ring-ring/40"
        >
          <IconTile icon={tile.icon} tone={tile.tone} size="md" />
          <span className="flex min-w-0 flex-1 flex-col">
            <span className="flex items-baseline gap-2">
              {tile.count === undefined ? (
                loading ? (
                  <Skeleton className="h-5 w-8" />
                ) : (
                  <span className="text-lg font-semibold leading-6 text-subtle-foreground">—</span>
                )
              ) : (
                <span className="text-lg font-semibold leading-6 tracking-[-0.02em] tabular">
                  {tile.count}
                </span>
              )}
              <span className="truncate text-[13px] font-medium text-muted-foreground">
                {tile.label}
              </span>
            </span>
            <span
              className="truncate text-xs text-subtle-foreground"
              title={config ? tile.hint : undefined}
            >
              {config ? tile.hint : loading ? ' ' : 'Configuration unavailable'}
            </span>
          </span>
          <ChevronRight className="size-4 shrink-0 text-subtle-foreground transition-transform group-hover:translate-x-0.5" />
        </Link>
      ))}
    </nav>
  )
}

// ---------------------------------------------------------------- distributions

function CardLinkAction({ to, label }: { to: string; label: string }) {
  const navigate = useNavigate()
  return (
    <CardAction>
      <Button variant="ghost" size="xs" onClick={() => navigate(to)}>
        {label}
        <ArrowRight />
      </Button>
    </CardAction>
  )
}

const CATEGORY_LABEL = { guardrail: 'Guard', routing: 'Route', fallback: 'Default' } as const

/** Share of requests by decision; the configured decisions when no traffic is recorded. */
export function HomeRoutingMix({
  aggregate,
  config,
  range,
}: {
  aggregate: InsightsAggregateResponse | null
  config: RouterConfig | null
  range: HomeRange
}) {
  const shares = topShares(aggregate?.decision_distribution)
  const configured = config
    ? buildDecisionPreviewRows(
        (() => {
          const { guardrails, routing, fallbacks } = categorizeDecisions(config)
          return [...guardrails, ...routing, ...fallbacks]
        })(),
        6,
      )
    : []
  const total = config ? countDecisions(config) : 0
  return (
    <Card data-testid="home-routing-mix">
      <CardHeader>
        <CardTitle>Routing mix</CardTitle>
        <CardDescription>
          {shares.length
            ? `Share of requests · ${HOME_RANGE_LABEL[range]}`
            : config
              ? `${total} configured decision${total === 1 ? '' : 's'}, by priority`
              : 'Decisions the router can choose'}
        </CardDescription>
        <CardLinkAction to="/routing/decisions" label="All decisions" />
      </CardHeader>
      <CardContent className="flex flex-col gap-3.5 pt-3">
        {shares.length ? (
          shares.map((entry) => (
            <Link
              key={entry.name}
              to={`/observe/requests?decision=${encodeURIComponent(entry.name)}&range=${range}`}
              className="group flex flex-col gap-1.5 rounded-sm outline-none focus-visible:ring-[3px] focus-visible:ring-ring/40"
            >
              <span className="flex items-center justify-between gap-3 text-[13px]">
                <span className="flex min-w-0 items-center gap-2">
                  <Split className="size-3.5 shrink-0 text-chart-3" />
                  <span className="truncate group-hover:text-primary" title={entry.name}>
                    {formatInsightsDecisionName(entry.name)}
                  </span>
                </span>
                <span className="font-mono text-xs tabular text-muted-foreground">
                  {entry.share.toFixed(1)}%
                </span>
              </span>
              <MiniBar value={entry.share} max={shares[0].share} tone="violet" />
            </Link>
          ))
        ) : configured.length ? (
          <ul className="-my-1 flex flex-col">
            {configured.map((row) => (
              <li key={row.key} className="flex items-center gap-2.5 py-1.5" title={row.title}>
                <Split className="size-3.5 shrink-0 text-chart-3" />
                <span className="flex min-w-0 flex-1 flex-col">
                  <span className="truncate text-[13px]">{row.name}</span>
                  <span className="truncate text-xs text-subtle-foreground">
                    {row.modelNames === '—' ? (row.scopeLabel ?? 'No model') : row.modelNames}
                  </span>
                </span>
                <Badge
                  size="sm"
                  tone={
                    row.category === 'guardrail'
                      ? 'warning'
                      : row.category === 'fallback'
                        ? 'neutral'
                        : 'violet'
                  }
                >
                  {CATEGORY_LABEL[row.category]}
                </Badge>
                <span className="w-9 text-right font-mono text-xs tabular text-muted-foreground">
                  {row.priorityLabel}
                </span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="py-6 text-center text-[13px] text-muted-foreground">
            {config ? 'No decision is configured yet.' : 'The configuration did not load.'}
          </p>
        )}
      </CardContent>
    </Card>
  )
}

interface ConfiguredModel {
  name?: string
  backend_refs?: Array<{ provider?: string; name?: string }>
  [key: string]: unknown
}

/** Where traffic landed, by backend model; the configured backends when none is recorded. */
export function HomeModelsInUse({
  aggregate,
  config,
  range,
  modelProviders,
}: {
  aggregate: InsightsAggregateResponse | null
  config: RouterConfig | null
  range: HomeRange
  modelProviders: Record<string, string>
}) {
  const labels = useModelLabels()
  const shares = topShares(aggregate?.model_selection)
  const models = ((config?.providers?.models ?? []) as ConfiguredModel[]).filter((model) =>
    Boolean(model.name),
  )
  const defaults = config?.providers?.defaults as { model?: unknown } | undefined
  return (
    <Card data-testid="home-models">
      <CardHeader>
        <CardTitle>Models in use</CardTitle>
        <CardDescription>
          {shares.length
            ? `Where traffic landed · ${HOME_RANGE_LABEL[range]}`
            : config
              ? `${models.length} configured backend model${models.length === 1 ? '' : 's'}`
              : 'Backend models the router can call'}
        </CardDescription>
        <CardLinkAction to="/models/backends" label="Backends" />
      </CardHeader>
      <CardContent className="flex flex-col gap-0.5 pt-2">
        {shares.length ? (
          shares.map((entry) => {
            const provider = modelProviderId(entry.name, modelProviders)
            const label = modelLabel(labels, entry.name)
            return (
              <div key={entry.name} className="flex items-center gap-3 rounded-lg py-1.5">
                {provider ? (
                  <ProviderLogo provider={provider} size={18} framed />
                ) : (
                  <IconTile icon={Boxes} tone="green" size="sm" className="size-8" />
                )}
                <span className="flex min-w-0 flex-1 flex-col">
                  <span className="truncate text-[13px] font-medium" title={entry.name}>
                    {label}
                  </span>
                  <span className="truncate text-xs text-subtle-foreground tabular">
                    {entry.value.toLocaleString()} request{entry.value === 1 ? '' : 's'}
                  </span>
                </span>
                <span className="flex w-24 flex-col items-end gap-1">
                  <span className="font-mono text-xs tabular">{entry.share.toFixed(1)}%</span>
                  <MiniBar value={entry.share} max={shares[0].share} tone="green" className="h-1" />
                </span>
              </div>
            )
          })
        ) : models.length ? (
          models.slice(0, 6).map((model) => {
            const provider =
              modelProviderId(model.name, modelProviders) ??
              model.backend_refs?.find((ref) => ref.provider)?.provider
            const backends = model.backend_refs?.length ?? 0
            return (
              <div key={model.name} className="flex items-center gap-3 rounded-lg py-1.5">
                {provider ? (
                  <ProviderLogo provider={provider} size={18} framed />
                ) : (
                  <IconTile icon={Boxes} tone="green" size="sm" className="size-8" />
                )}
                <span className="flex min-w-0 flex-1 flex-col">
                  <span className="truncate text-[13px] font-medium">{model.name}</span>
                  <span className="truncate text-xs text-subtle-foreground">
                    {backends
                      ? `${backends} backend${backends === 1 ? '' : 's'}`
                      : 'Catalog-backed'}
                  </span>
                </span>
                {defaults?.model === model.name ? (
                  <Badge size="sm" tone="success">
                    default
                  </Badge>
                ) : null}
              </div>
            )
          })
        ) : (
          <p className="py-6 text-center text-[13px] text-muted-foreground">
            {config ? 'No backend model is configured yet.' : 'The configuration did not load.'}
          </p>
        )}
        {!shares.length && models.length > 6 ? (
          <Link
            to="/models/backends"
            className="pt-1 text-xs font-medium text-primary hover:underline"
          >
            +{models.length - 6} more
          </Link>
        ) : null}
      </CardContent>
    </Card>
  )
}

// ---------------------------------------------------------------- public models

/** What apps call: each recipe with its public model names, counts and topology. */
export function HomePublicModels({ config }: { config: RouterConfig | null }) {
  const navigate = useNavigate()
  const scopes = config ? listRoutingScopes(config as RouterConfig & RoutingScopedConfigLike) : []
  const publicModels = listPublicModels(config)
  const rows = scopes.map((scope) => ({
    scope,
    models: publicModels.filter((model) => model.recipe === scope.id),
  }))
  const shown = rows.slice(0, 5)
  return (
    <Card className="lg:col-span-2 xl:col-span-1" data-testid="home-public-models">
      <CardHeader>
        <CardTitle>Public models</CardTitle>
        <CardDescription>
          {config
            ? `What your apps call · ${scopes.length} recipe${scopes.length === 1 ? '' : 's'}`
            : 'What your apps call'}
        </CardDescription>
        <CardLinkAction to="/routing/recipes" label="Recipes" />
      </CardHeader>
      <CardContent className="flex flex-col gap-2 pt-3">
        {shown.length === 0 ? (
          <p className="py-6 text-center text-[13px] text-muted-foreground">
            The configuration did not load.
          </p>
        ) : (
          shown.map(({ scope, models }) => {
            const lead = models[0]
            const signals = Object.values(scope.routing.signals ?? {}).reduce<number>(
              (sum, value) => sum + (Array.isArray(value) ? value.length : 0),
              0,
            )
            const decisions = Array.isArray(scope.routing.decisions)
              ? scope.routing.decisions.length
              : 0
            return (
              <div
                key={scope.id}
                className="group rounded-lg border border-border bg-surface-subtle/60 px-3 py-2.5 transition-colors hover:border-border-strong"
              >
                <div className="flex min-w-0 items-center gap-2">
                  <Layers className="size-3.5 shrink-0 text-chart-1" />
                  {lead ? (
                    <span
                      className="truncate font-mono text-[12.5px] font-medium"
                      title={models.map((model) => model.name).join(', ')}
                    >
                      {lead.name}
                    </span>
                  ) : (
                    <span className="truncate text-[13px] font-medium">{scope.label}</span>
                  )}
                  {models.length > 1 ? (
                    <Tooltip
                      content={models
                        .slice(1)
                        .map((model) => model.name)
                        .join(', ')}
                    >
                      <span className="shrink-0 text-[11px] text-subtle-foreground">
                        +{models.length - 1}
                      </span>
                    </Tooltip>
                  ) : null}
                  {lead?.isDefault ? (
                    <Badge size="sm" tone="brand" className="shrink-0">
                      default
                    </Badge>
                  ) : null}
                  {lead?.api === 'systemone' ? (
                    <Badge size="sm" tone="violet" className="shrink-0">
                      System One
                    </Badge>
                  ) : null}
                  <span className="ml-auto flex shrink-0 items-center">
                    {lead ? (
                      <CopyButton
                        value={lead.name}
                        label="Copy model name"
                        className="size-6 opacity-0 transition-opacity group-hover:opacity-100 focus-visible:opacity-100"
                      />
                    ) : null}
                    <Tooltip content="View topology">
                      <Button
                        variant="ghost"
                        size="icon-xs"
                        className="size-6"
                        aria-label={`View ${scope.label} topology`}
                        onClick={() =>
                          navigate(`/routing/topology?scope=${encodeURIComponent(scope.id)}`)
                        }
                      >
                        <Waypoints />
                      </Button>
                    </Tooltip>
                  </span>
                </div>
                <div className="mt-1 flex min-w-0 items-center gap-3 text-xs text-muted-foreground">
                  <span className="truncate font-mono text-[11.5px]">{scope.id}</span>
                  <span className="shrink-0 tabular">
                    {decisions} decision{decisions === 1 ? '' : 's'}
                  </span>
                  <span className="shrink-0 tabular">
                    {signals} signal{signals === 1 ? '' : 's'}
                  </span>
                </div>
              </div>
            )
          })
        )}
        {rows.length > shown.length ? (
          <Link to="/routing/recipes" className="text-xs font-medium text-primary hover:underline">
            +{rows.length - shown.length} more recipes
          </Link>
        ) : null}
      </CardContent>
    </Card>
  )
}

// ---------------------------------------------------------------- shortcuts

const SHORTCUTS: Array<{
  to: string
  label: string
  description: string
  icon: LucideIcon
  tone: Tone
}> = [
  {
    to: '/get-started',
    label: 'Connect an app or agent',
    description: 'The endpoint, client snippets and agent setup',
    icon: Rocket,
    tone: 'blue',
  },
  {
    to: '/playground',
    label: 'Try a prompt',
    description: 'See which decision and model answer it',
    icon: MessagesSquare,
    tone: 'blue',
  },
  {
    to: '/routing/editor',
    label: 'Edit routing as code',
    description: 'Signals, decisions and plugins in one document',
    icon: FileCode2,
    tone: 'violet',
  },
  {
    to: '/routing/topology',
    label: 'Explore the topology',
    description: 'How signals lead to decisions and models',
    icon: Waypoints,
    tone: 'teal',
  },
  {
    to: '/observe/evaluations',
    label: 'Run an evaluation',
    description: 'Score recipes against a benchmark',
    icon: FlaskConical,
    tone: 'orange',
  },
]

/** The main tasks, as links the viewer's role can open. */
export function HomeShortcuts({ canOpen }: { canOpen: (path: string) => boolean }) {
  const shortcuts = SHORTCUTS.filter((shortcut) => canOpen(shortcut.to))
  if (shortcuts.length === 0) return null
  return (
    <Card data-testid="home-shortcuts">
      <CardHeader>
        <CardTitle>Jump to</CardTitle>
        <CardDescription>The main tasks</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-0.5 pt-2">
        {shortcuts.map((shortcut) => (
          <Link
            key={shortcut.to}
            to={shortcut.to}
            className="group -mx-2 flex items-center gap-3 rounded-lg px-2 py-2 outline-none transition-colors hover:bg-accent focus-visible:ring-[3px] focus-visible:ring-ring/40"
          >
            <IconTile icon={shortcut.icon} tone={shortcut.tone} size="sm" />
            <span className="flex min-w-0 flex-1 flex-col">
              <span className="truncate text-[13px] font-medium">{shortcut.label}</span>
              <span className="truncate text-xs text-subtle-foreground">
                {shortcut.description}
              </span>
            </span>
            <ArrowRight className="size-3.5 shrink-0 text-subtle-foreground opacity-0 transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100" />
          </Link>
        ))}
      </CardContent>
    </Card>
  )
}

// ---------------------------------------------------------------- services

/** The services `/api/status` reports, with their state. */
export function HomeServices({
  status,
  failed,
  onRetry,
}: {
  status: SystemStatus | null
  failed: boolean
  onRetry: () => void
}) {
  const services = status?.services ?? []
  const healthy = services.filter((service) => service.healthy).length
  return (
    <Card data-testid="home-services">
      <CardHeader>
        <CardTitle>Services</CardTitle>
        <CardDescription>
          {status
            ? services.length
              ? `${healthy} of ${services.length} operational`
              : 'No service reported'
            : failed
              ? 'Status is unavailable'
              : 'Checking…'}
        </CardDescription>
        <CardLinkAction to="/observe/status" label="Status" />
      </CardHeader>
      <CardContent className="flex flex-col gap-0.5 pt-2">
        {!status && failed ? (
          <EmptyState
            compact
            icon={ServerOff}
            tone="slate"
            title="The dashboard could not read the status"
            actions={
              <Button size="sm" variant="outline" onClick={onRetry}>
                <RefreshCw />
                Try again
              </Button>
            }
            className="py-6"
          />
        ) : !status ? (
          [0, 1, 2].map((index) => <Skeleton key={index} className="my-1.5 h-9 w-full" />)
        ) : (
          services.map((service) => (
            <div key={service.name} className="flex items-center gap-3 rounded-lg py-1.5">
              <StatusDot tone={service.healthy ? 'success' : 'danger'} className="ml-1" />
              <span className="flex min-w-0 flex-1 flex-col">
                <span className="truncate text-[13px] font-medium">{service.name}</span>
                {service.message ? (
                  <span className="truncate text-xs text-subtle-foreground">{service.message}</span>
                ) : null}
              </span>
              <span
                className={cn(
                  'shrink-0 text-xs capitalize',
                  service.healthy ? 'text-muted-foreground' : 'text-destructive-foreground',
                )}
              >
                {service.status}
              </span>
            </div>
          ))
        )}
      </CardContent>
    </Card>
  )
}
