import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  BrainCircuit,
  ChartNoAxesColumn,
  ChevronRight,
  Code2,
  Eye,
  Layers,
  MoreHorizontal,
  Network,
  Package,
  Pencil,
  Plus,
  Trash2,
} from 'lucide-react'

import {
  Badge,
  Button,
  Card,
  CopyButton,
  DataToolbar,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
  EmptyState,
  IconTile,
  ProviderStack,
  Sparkline,
  Tooltip,
  cn,
} from '../ui'
import { formatInsightsCost } from '../utils/insightsCost'
import { formatLatency } from './configPageModelHealthPresentation'
import { modelProviderIds, providerLogoKey } from './configPageModelPresentation'
import ConfigPageRecipesTrafficChart from './ConfigPageRecipesTrafficChart'
import {
  apiLabel,
  publicModelResolution,
  recipeDisplayName,
  type PublicModelCard,
} from './configPageRecipesPageSupport'
import type { ConfigData, NormalizedModel } from './configPageSupport'
import {
  formatRequestCount,
  usePublicModelTraffic,
  type PublicModelTrafficEntry,
} from './usePublicModelTraffic'

const ease = [0.16, 1, 0.3, 1] as const

interface Props {
  cards: PublicModelCard[]
  config: ConfigData
  models: NormalizedModel[]
  isReadonly: boolean
  routerHealthy: boolean
  activeKey: string | null
  onOpen: (card: PublicModelCard, view?: 'connect') => void
  onEdit: (card: PublicModelCard) => void
  onDelete: (card: PublicModelCard) => void
  onCreate: () => void
}

function StatusBadge({ card, routerHealthy }: { card: PublicModelCard; routerHealthy: boolean }) {
  const resolution = publicModelResolution(card)
  if (resolution === 'missing')
    return (
      <Badge size="sm" tone="danger" dot>
        Recipe missing
      </Badge>
    )
  if (resolution === 'empty')
    return (
      <Badge size="sm" tone="warning" dot>
        No decisions
      </Badge>
    )
  return routerHealthy ? (
    <Badge size="sm" tone="success" dot>
      Live
    </Badge>
  ) : (
    <Tooltip content="The Router is not running; this name answers once it starts.">
      <span className="inline-flex">
        <Badge size="sm" tone="neutral" dot>
          Configured
        </Badge>
      </span>
    </Tooltip>
  )
}

function PublicModelCardView({
  card,
  index,
  config,
  models,
  isReadonly,
  routerHealthy,
  active,
  onOpen,
  onEdit,
  onDelete,
  traffic,
}: Omit<Props, 'cards' | 'activeKey' | 'onCreate'> & {
  card: PublicModelCard
  index: number
  active: boolean
  /** The last 24 h from Router Replay; without it the card shows its configuration. */
  traffic?: PublicModelTrafficEntry
}) {
  const navigate = useNavigate()
  const recipeName = recipeDisplayName(config, card.entrypoint.recipe)
  const targetModels = card.targets
    .map((target) => models.find((model) => model.name === target))
    .filter((model): model is NormalizedModel => Boolean(model))
  const providers = [
    ...new Set(
      targetModels.flatMap((model) =>
        modelProviderIds(model)
          .slice(0, 1)
          .map((id) => providerLogoKey(id)),
      ),
    ),
  ]
  const aliases = card.entrypoint.model_names.slice(1)
  const editable = !isReadonly && card.api === 'chat'
  const description =
    card.recipe?.description ||
    (card.derived
      ? 'The default routing profile, served under its built-in name.'
      : `Routes through ${recipeName}.`)
  const stats = traffic
    ? [
        { label: 'Requests · 24h', value: formatRequestCount(traffic.requests) },
        { label: 'p95', value: traffic.p95Ms !== undefined ? formatLatency(traffic.p95Ms) : '—' },
        {
          label: 'Cost · 24h',
          value: traffic.cost
            ? formatInsightsCost(traffic.cost.amount, traffic.cost.currency, { total: true })
            : '—',
        },
      ]
    : [
        { label: 'Signals', value: card.signals },
        { label: 'Decisions', value: card.decisions.length },
        { label: 'Backends', value: card.targets.length },
      ]
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05, duration: 0.45, ease }}
    >
      <Card
        interactive
        onClick={() => onOpen(card)}
        className={cn(
          'h-full gap-0 overflow-hidden p-0',
          active && 'border-primary/50 shadow-[0_0_0_3px_var(--ring)]',
        )}
      >
        <div className="flex items-start gap-3 px-4 pt-4">
          <IconTile icon={card.api === 'systemone' ? BrainCircuit : Layers} tone="blue" />
          <div className="flex min-w-0 flex-1 flex-col gap-1.5">
            <div className="flex min-w-0 items-center gap-0.5">
              <button
                type="button"
                onClick={(event) => {
                  event.stopPropagation()
                  onOpen(card)
                }}
                title={card.key}
                className="truncate rounded-sm text-left font-mono text-[13px] font-semibold outline-none hover:text-primary focus-visible:ring-[3px] focus-visible:ring-ring/40"
              >
                {card.key}
              </button>
              <span
                role="presentation"
                onClick={(event) => event.stopPropagation()}
                onKeyDown={(event) => event.stopPropagation()}
              >
                <CopyButton value={card.key} label="Copy model name" className="size-6" />
              </span>
            </div>
            <div className="flex flex-wrap items-center gap-1.5">
              <StatusBadge card={card} routerHealthy={routerHealthy} />
              <Badge size="sm" tone={card.api === 'systemone' ? 'violet' : 'outline'}>
                {apiLabel(card)}
              </Badge>
              {card.entrypoint.recipe === 'default' && card.api === 'chat' ? (
                <Badge size="sm" tone="brand">
                  default
                </Badge>
              ) : null}
              {aliases.length ? (
                <Tooltip content={aliases.join(', ')}>
                  <span className="inline-flex">
                    <Badge size="sm" tone="neutral">
                      +{aliases.length} {aliases.length === 1 ? 'alias' : 'aliases'}
                    </Badge>
                  </span>
                </Tooltip>
              ) : null}
            </div>
          </div>
          <span
            role="presentation"
            onClick={(event) => event.stopPropagation()}
            onKeyDown={(event) => event.stopPropagation()}
          >
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon-xs"
                  aria-label={`Actions for ${card.key}`}
                  className="-mr-1.5 -mt-0.5"
                >
                  <MoreHorizontal />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-48">
                <DropdownMenuItem onSelect={() => onOpen(card)}>
                  <Eye />
                  Open details
                </DropdownMenuItem>
                <DropdownMenuItem onSelect={() => onOpen(card, 'connect')}>
                  <Code2 />
                  Use this model
                </DropdownMenuItem>
                <DropdownMenuItem
                  onSelect={() =>
                    navigate(
                      `/routing/topology?scope=${encodeURIComponent(card.entrypoint.recipe)}`,
                    )
                  }
                >
                  <Network />
                  Open topology
                </DropdownMenuItem>
                <DropdownMenuItem
                  onSelect={() =>
                    navigate(`/observe/evaluations?model=${encodeURIComponent(card.key)}`)
                  }
                >
                  <ChartNoAxesColumn />
                  Evaluate
                </DropdownMenuItem>
                {editable ? (
                  <DropdownMenuItem onSelect={() => onEdit(card)}>
                    <Pencil />
                    Edit
                  </DropdownMenuItem>
                ) : null}
                {editable && !card.derived ? (
                  <>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem variant="destructive" onSelect={() => onDelete(card)}>
                      <Trash2 />
                      Delete
                    </DropdownMenuItem>
                  </>
                ) : null}
              </DropdownMenuContent>
            </DropdownMenu>
          </span>
        </div>
        <p className="mt-3 line-clamp-2 h-[2lh] px-4 text-xs leading-relaxed text-muted-foreground">
          {description}
        </p>
        <div className="mx-4 mt-3 flex min-w-0 items-center gap-1.5 rounded-lg border border-border bg-surface-subtle/60 py-1.5 pl-2.5 pr-1.5 text-xs">
          <Package className="size-3.5 shrink-0 text-chart-1" />
          <span className="truncate font-mono font-medium">{recipeName}</span>
          <ChevronRight className="size-3 shrink-0 text-subtle-foreground" />
          <span className="shrink-0 tabular text-muted-foreground">
            {card.decisions.length} {card.decisions.length === 1 ? 'decision' : 'decisions'}
          </span>
          {providers.length ? (
            <>
              <ChevronRight className="size-3 shrink-0 text-subtle-foreground" />
              <Tooltip content={card.targets.join(', ')}>
                <span className="flex items-center">
                  <ProviderStack providers={providers} size={14} max={4} />
                </span>
              </Tooltip>
            </>
          ) : null}
        </div>
        {traffic && traffic.hourly.length > 1 ? (
          <div className="mt-3">
            <Sparkline data={traffic.hourly} color="var(--chart-1)" height={34} />
          </div>
        ) : null}
        <div
          className={cn(
            'grid grid-cols-3 border-t border-border bg-card',
            traffic && traffic.hourly.length > 1 ? '' : 'mt-4',
          )}
        >
          {stats.map((stat, statIndex) => (
            <div
              key={stat.label}
              className={cn(
                'flex flex-col gap-0.5 px-4 py-2.5',
                statIndex > 0 && 'border-l border-border',
              )}
            >
              <span className="text-[11px] text-subtle-foreground">{stat.label}</span>
              <span className="font-mono text-[13px] font-medium tabular">{stat.value}</span>
            </div>
          ))}
        </div>
      </Card>
    </motion.div>
  )
}

/** Above this many public models the grid gets a search field. */
const SEARCH_THRESHOLD = 6

export default function ConfigPageRecipesPublicModels(props: Props) {
  const { cards: allCards, isReadonly, onCreate } = props
  const [search, setSearch] = useState('')
  const traffic = usePublicModelTraffic(allCards.map((card) => card.key))
  const query = search.trim().toLocaleLowerCase()
  const cards = query
    ? allCards.filter((card) =>
        [card.entrypoint.recipe, ...card.entrypoint.model_names].some((value) =>
          value.toLocaleLowerCase().includes(query),
        ),
      )
    : allCards
  if (allCards.length === 0) {
    return (
      <Card className="py-2">
        <EmptyState
          icon={Layers}
          tone="blue"
          title="No public models yet"
          description="A public model is the name apps send in the model field. Publish one for a recipe so clients can call it."
          actions={
            isReadonly ? undefined : (
              <Button onClick={onCreate}>
                <Plus />
                New public model
              </Button>
            )
          }
        />
      </Card>
    )
  }
  return (
    <div className="flex flex-col">
      {allCards.length > SEARCH_THRESHOLD ? (
        <DataToolbar
          search={search}
          onSearchChange={setSearch}
          placeholder="Search model names or recipes…"
          actions={
            <span className="text-xs text-muted-foreground">
              <span className="tabular">{cards.length}</span> of{' '}
              <span className="tabular">{allCards.length}</span> public models
            </span>
          }
        />
      ) : null}
      {cards.length === 0 ? (
        <p className="rounded-xl border border-dashed border-border-strong px-4 py-10 text-center text-[13px] text-muted-foreground">
          No public model matches “{search.trim()}”.
        </p>
      ) : null}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2 xl:grid-cols-3">
        {cards.map((card, index) => (
          <PublicModelCardView
            key={`${card.api}:${card.key}`}
            card={card}
            index={index}
            active={props.activeKey === card.key}
            config={props.config}
            models={props.models}
            isReadonly={props.isReadonly}
            routerHealthy={props.routerHealthy}
            onOpen={props.onOpen}
            onEdit={props.onEdit}
            onDelete={props.onDelete}
            traffic={traffic.byModel.get(card.key)}
          />
        ))}
      </div>
      {traffic.phase === 'ready' ? (
        <ConfigPageRecipesTrafficChart
          traffic={allCards.flatMap((card) => {
            const entry = traffic.byModel.get(card.key)
            return entry ? [[card.key, entry] as const] : []
          })}
        />
      ) : null}
    </div>
  )
}
