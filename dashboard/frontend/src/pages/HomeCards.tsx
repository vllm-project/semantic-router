import type { LucideIcon } from 'lucide-react'
import { Link, useNavigate } from 'react-router-dom'
import {
  ArrowRight,
  Boxes,
  ChevronRight,
  Layers,
  Radar,
  RefreshCw,
  ServerOff,
  Split,
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
  EmptyState,
  IconTile,
  MiniBar,
  ProviderLogo,
  Skeleton,
  StatusDot,
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
import { HOME_RANGE_LABEL, topShares, type HomeAggregate, type HomeRange } from './homeSupport'
import type { HomeTraffic } from './homeTraffic'
import { HarnessLogo } from './InsightsHarness'
import { API_CLIENT_LABEL, harnessInfo, sortHarnessIds } from './insightsHarness'
import { modelProviderId } from './insightsConfigFacts'
import { modelLabel, useModelLabels } from './insightsModelLabelSupport'
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

/**
 * The routing configuration at a glance; each tile opens its manager. Stacked
 * beside a card on wide screens, the tiles share its height.
 */
export function HomeConfigTiles({
  config,
  loading,
  className,
}: {
  config: RouterConfig | null
  loading: boolean
  className?: string
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
    <nav
      aria-label="Routing configuration"
      className={cn(
        'grid grid-cols-2 gap-4 sm:grid-cols-4 xl:auto-rows-fr xl:grid-cols-1',
        className,
      )}
    >
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
  aggregate: HomeAggregate | null
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
  aggregate: HomeAggregate | null
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

// ---------------------------------------------------------------- agents in use

/** Where requests came from: each agent harness by its own logo, then plain API clients. */
export function HomeAgentsInUse({ traffic, range }: { traffic: HomeTraffic; range: HomeRange }) {
  const counts = traffic.harnessCounts ?? {}
  const entries = Object.entries(counts).filter(([, value]) => value > 0)
  const total = entries.reduce((sum, [, value]) => sum + value, 0)
  const ids = sortHarnessIds(entries.map(([id]) => id).filter((id) => id !== 'api_client'))
  const rows = [...ids, ...(counts.api_client ? ['api_client'] : [])].map((id) => ({
    id,
    value: counts[id] ?? 0,
    share: total > 0 ? ((counts[id] ?? 0) / total) * 100 : 0,
  }))
  const peak = Math.max(0, ...rows.map((row) => row.share))
  return (
    <Card className="lg:col-span-2 xl:col-span-1" data-testid="home-agents">
      <CardHeader>
        <CardTitle>Agents in use</CardTitle>
        <CardDescription>
          {rows.length
            ? `Requests by harness · ${HOME_RANGE_LABEL[range]}`
            : 'Coding agents and API clients that call the router'}
        </CardDescription>
        <CardLinkAction to="/observe/requests?tab=runs" label="Agent runs" />
      </CardHeader>
      <CardContent className="flex flex-col gap-0.5 pt-2">
        {rows.length ? (
          <ul className="flex flex-col gap-0.5" aria-label="Requests by agent harness">
            {rows.map((row) => {
              const info = harnessInfo(row.id === 'api_client' ? undefined : row.id)
              return (
                <li
                  key={row.id}
                  className="flex items-center gap-3 rounded-lg py-1.5"
                  data-harness={row.id}
                >
                  <HarnessLogo
                    harness={row.id === 'api_client' ? undefined : row.id}
                    size={18}
                    framed
                  />
                  <span className="flex min-w-0 flex-1 flex-col">
                    <span className="truncate text-[13px] font-medium">
                      {info?.label ?? API_CLIENT_LABEL}
                    </span>
                    <span className="truncate text-xs text-subtle-foreground tabular">
                      {row.value.toLocaleString()} request{row.value === 1 ? '' : 's'}
                    </span>
                  </span>
                  <span className="flex w-24 flex-col items-end gap-1">
                    <span className="font-mono text-xs tabular">{row.share.toFixed(1)}%</span>
                    <MiniBar value={row.share} max={peak} tone="blue" className="h-1" />
                  </span>
                </li>
              )
            })}
          </ul>
        ) : (
          <p className="py-6 text-center text-[13px] text-muted-foreground">
            {traffic.rowsLoading
              ? 'Counting requests…'
              : traffic.harnessCounts || traffic.rows.length === 0
                ? `No requests in the ${HOME_RANGE_LABEL[range]}.`
                : 'This router does not count requests by agent harness yet.'}
          </p>
        )}
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
