import { motion } from 'framer-motion'
import { ArrowRight, Layers, Plus, SlidersHorizontal, Split } from 'lucide-react'
import { Badge, Button, EmptyState, IconTile, Skeleton, cn } from '../../ui'
import type { ASTProgram, ASTRouteDecl } from '@/types/dsl'
import type { EntityKind } from '../builderPageTypes'
import type { BuilderRoutingScope } from '../builderPageRoutingScopeSupport'
import { GROUP_ORDER, groupMeta, type EditorEntity, type EditorGroup } from './editorModel'
import { ruleTreeToWhen, whenToRuleTree } from './editorRules'

interface EditorOverviewProps {
  ast: ASTProgram | null
  scope: BuilderRoutingScope | null
  entities: EditorEntity[]
  errors: number
  warnings: number
  readOnly: boolean
  onSelect: (entity: EditorEntity) => void
  onAdd: (kind: EntityKind) => void
  onShowProblems: () => void
  onGlobal: () => void
}

const addKind: Record<EditorGroup, EntityKind> = {
  model: 'model',
  signal: 'signal',
  projection: 'projection-partition',
  route: 'route',
  plugin: 'plugin',
}

function RouteRow({
  route,
  index,
  onOpen,
}: {
  route: ASTRouteDecl
  index: number
  onOpen: () => void
}) {
  const when = ruleTreeToWhen(whenToRuleTree(route.when))
  return (
    <motion.li
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: Math.min(index, 12) * 0.02, duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
    >
      <button
        type="button"
        onClick={onOpen}
        className="group grid w-full grid-cols-[3rem_minmax(0,1fr)] items-baseline gap-x-3 gap-y-0.5 px-4 py-2.5 text-left outline-none transition-colors hover:bg-surface-subtle focus-visible:bg-surface-subtle"
      >
        <span className="font-mono text-[11.5px] tabular text-subtle-foreground">
          P{route.priority}
        </span>
        <span className="flex min-w-0 items-baseline gap-3">
          <span className="min-w-0 shrink truncate font-mono text-[12.5px] font-medium text-foreground">
            {route.name}
          </span>
          <span className="ml-auto flex min-w-0 max-w-[55%] shrink-0 items-center gap-1.5">
            <ArrowRight className="size-3 shrink-0 self-center text-subtle-foreground" />
            {route.models.length ? (
              <span className="truncate font-mono text-[11.5px] text-foreground/80">
                {route.models.map((model) => model.model).join(', ')}
              </span>
            ) : route.plugins.length ? (
              <Badge tone="rose" size="sm">
                plugin answers
              </Badge>
            ) : (
              <span className="text-xs text-warning-foreground">no model</span>
            )}
          </span>
        </span>
        <span className="col-start-2 truncate font-mono text-[11.5px] text-muted-foreground">
          {when ? (
            <>
              <span className="text-subtle-foreground">WHEN </span>
              {when}
            </>
          ) : (
            <span className="font-sans italic text-subtle-foreground">always matches</span>
          )}
        </span>
      </button>
    </motion.li>
  )
}

/** The overview's shape while the compiler connects and the Router config loads. */
export function OverviewSkeleton() {
  return (
    <div
      className="mx-auto flex w-full max-w-[1040px] flex-col gap-5"
      aria-label="Loading the config"
    >
      <div className="flex items-start gap-3">
        <Skeleton className="size-9 rounded-lg" />
        <div className="flex flex-col gap-2">
          <Skeleton className="h-4 w-44" />
          <Skeleton className="h-3 w-72" />
        </div>
      </div>
      <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-3 lg:grid-cols-5">
        {[0, 1, 2, 3, 4].map((index) => (
          <Skeleton key={index} className="h-[104px] rounded-xl" />
        ))}
      </div>
      <div className="flex flex-col gap-2 rounded-xl border border-border bg-card p-4">
        {[0, 1, 2, 3, 4, 5].map((index) => (
          <Skeleton key={index} className="h-5" style={{ width: `${88 - index * 7}%` }} />
        ))}
      </div>
    </div>
  )
}

/** The active recipe at a glance: what it declares, how its decisions route, and what to fix. */
export function EditorOverview({
  ast,
  scope,
  entities,
  errors,
  warnings,
  readOnly,
  onSelect,
  onAdd,
  onShowProblems,
  onGlobal,
}: EditorOverviewProps) {
  const routes = [...(ast?.routes ?? [])].sort((left, right) => right.priority - left.priority)
  const counts = Object.fromEntries(
    GROUP_ORDER.map((group) => [
      group,
      entities.filter(
        (item) => (item.kind.startsWith('projection-') ? 'projection' : item.kind) === group,
      ).length,
    ]),
  ) as Record<EditorGroup, number>
  const entityFor = (route: ASTRouteDecl) =>
    entities.find((item) => item.kind === 'route' && item.name === route.name)

  if (!ast) {
    return (
      <EmptyState
        icon={Split}
        tone="violet"
        title="Nothing to show yet"
        description="Load the Router's config or import one to start editing its routing."
      />
    )
  }

  return (
    <div className="mx-auto flex w-full max-w-[1040px] flex-col gap-5">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div className="flex min-w-0 items-start gap-3">
          <IconTile icon={Layers} tone="blue" size="md" />
          <div className="flex min-w-0 flex-col gap-1">
            <h2 className="text-[15px] font-semibold tracking-[-0.01em]">
              {scope?.recipeName ? (
                <>
                  Recipe <span className="font-mono">{scope.recipeName}</span>
                </>
              ) : (
                (scope?.label ?? 'Default routing')
              )}
            </h2>
            <p className="text-[13px] text-muted-foreground">
              {scope?.recipeName
                ? 'An isolated routing program; its public models resolve here.'
                : 'Routing every request gets unless an entrypoint selects a recipe.'}
            </p>
            {scope?.modelNames.length ? (
              <div className="mt-1 flex flex-wrap items-center gap-1.5">
                <span className="text-xs text-subtle-foreground">Public models</span>
                {scope.modelNames.map((name) => (
                  <Badge key={name} tone="brand" size="sm" className="font-mono">
                    {name}
                  </Badge>
                ))}
              </div>
            ) : null}
          </div>
        </div>
        {errors || warnings ? (
          <Button variant="outline" size="sm" onClick={onShowProblems}>
            <span
              className={cn('size-1.5 rounded-full', errors ? 'bg-destructive' : 'bg-warning')}
            />
            {errors ? `${errors} ${errors === 1 ? 'error' : 'errors'}` : null}
            {errors && warnings ? ' · ' : null}
            {warnings ? `${warnings} ${warnings === 1 ? 'warning' : 'warnings'}` : null}
          </Button>
        ) : null}
      </header>

      <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-3 lg:grid-cols-5">
        {GROUP_ORDER.map((group, index) => {
          const meta = groupMeta[group]
          return (
            <motion.div
              key={group}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.03, duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
              className="group flex flex-col gap-3 rounded-xl border border-border bg-card px-3.5 py-3 shadow-xs"
            >
              <div className="flex items-center justify-between">
                <IconTile icon={meta.icon} tone={meta.tone} size="sm" />
                {readOnly ? null : (
                  <Button
                    variant="ghost"
                    size="icon-xs"
                    aria-label={`New ${meta.one.toLowerCase()}`}
                    className="opacity-70 group-hover:opacity-100"
                    onClick={() => onAdd(addKind[group])}
                  >
                    <Plus />
                  </Button>
                )}
              </div>
              <div className="flex flex-col">
                <span className="text-[22px] font-semibold leading-7 tabular tracking-[-0.02em]">
                  {counts[group]}
                </span>
                <span className="text-xs text-muted-foreground">{meta.label}</span>
              </div>
            </motion.div>
          )
        })}
      </div>

      <section className="overflow-hidden rounded-xl border border-border bg-card shadow-xs">
        <header className="flex items-center justify-between gap-3 border-b border-border px-4 py-3">
          <div className="flex flex-col gap-0.5">
            <h3 className="text-[13px] font-semibold">Decisions in priority order</h3>
            <p className="text-xs text-muted-foreground">
              The first decision whose rule matches answers the request.
            </p>
          </div>
          {readOnly ? null : (
            <Button variant="outline" size="xs" onClick={() => onAdd('route')}>
              <Plus />
              New decision
            </Button>
          )}
        </header>
        {routes.length ? (
          <ul className="divide-y divide-border">
            {routes.map((route, index) => (
              <RouteRow
                key={route.name}
                route={route}
                index={index}
                onOpen={() => {
                  const entity = entityFor(route)
                  if (entity) onSelect(entity)
                }}
              />
            ))}
          </ul>
        ) : (
          <EmptyState
            compact
            icon={Split}
            tone="violet"
            title="No decisions in this scope"
            description="Decisions pick a model when their signals match."
            actions={
              readOnly ? null : (
                <Button size="sm" onClick={() => onAdd('route')}>
                  <Plus />
                  New decision
                </Button>
              )
            }
          />
        )}
      </section>

      <button
        type="button"
        onClick={onGlobal}
        className="flex items-center gap-3 rounded-xl border border-dashed border-border-strong px-4 py-3 text-left outline-none transition-colors hover:bg-surface-subtle focus-visible:ring-[3px] focus-visible:ring-ring/40"
      >
        <IconTile icon={SlidersHorizontal} tone="slate" size="sm" />
        <span className="flex min-w-0 flex-1 flex-col">
          <span className="text-[13px] font-medium">Router settings</span>
          <span className="text-xs text-muted-foreground">
            Router-wide defaults sit outside recipes: view them here, edit them in Router settings.
          </span>
        </span>
        <ArrowRight className="size-4 text-subtle-foreground" />
      </button>
    </div>
  )
}
