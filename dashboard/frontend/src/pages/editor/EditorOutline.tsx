import * as React from 'react'
import { motion } from 'framer-motion'
import {
  ChevronRight,
  ChevronsDownUp,
  LayoutDashboard,
  Plus,
  Search,
  SlidersHorizontal,
  X,
} from 'lucide-react'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuTrigger,
  IconTile,
  Tooltip,
  cn,
} from '../../ui'
import type { DiagLevel } from '@/types/dsl'
import type { EntityKind } from '../builderPageTypes'
import { GROUP_ORDER, groupMeta, groupOf, type EditorEntity, type EditorGroup } from './editorModel'

const problemDot: Record<DiagLevel, string> = {
  error: 'bg-destructive',
  warning: 'bg-warning',
  constraint: 'bg-info',
}

const problemLabel: Record<DiagLevel, string> = {
  error: 'Has errors',
  warning: 'Has warnings',
  constraint: 'Has hints',
}

export type OutlineTarget = 'overview' | 'global' | { key: string }

interface EditorOutlineProps {
  entities: EditorEntity[]
  problems: Map<string, DiagLevel>
  activeKey: string | null
  target: 'overview' | 'global' | 'entity' | 'add'
  loading: boolean
  onOverview: () => void
  onGlobal: () => void
  onSelect: (entity: EditorEntity) => void
  onAdd: (kind: EntityKind) => void
}

function OutlineButton({
  active,
  onClick,
  icon,
  children,
}: {
  active: boolean
  onClick: () => void
  icon: React.ReactNode
  children: React.ReactNode
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-current={active ? 'page' : undefined}
      className={cn(
        'relative flex h-8 w-full items-center gap-2 rounded-md px-2 text-left text-[13px] font-medium outline-none transition-colors focus-visible:ring-[3px] focus-visible:ring-ring/40',
        active
          ? 'bg-accent text-foreground'
          : 'text-muted-foreground hover:bg-accent/60 hover:text-foreground',
      )}
    >
      {icon}
      <span className="min-w-0 flex-1 truncate">{children}</span>
    </button>
  )
}

function AddButton({ group, onAdd }: { group: EditorGroup; onAdd: (kind: EntityKind) => void }) {
  const meta = groupMeta[group]
  const className =
    'grid size-6 shrink-0 place-items-center rounded-md text-subtle-foreground outline-none transition-colors hover:bg-accent hover:text-foreground focus-visible:ring-[3px] focus-visible:ring-ring/40'
  if (group === 'projection') {
    return (
      <DropdownMenu>
        <Tooltip content="Add a projection">
          <DropdownMenuTrigger asChild>
            <button type="button" aria-label="Add projection" className={className}>
              <Plus className="size-3.5" />
            </button>
          </DropdownMenuTrigger>
        </Tooltip>
        <DropdownMenuContent align="end" className="w-56">
          <DropdownMenuLabel>New projection</DropdownMenuLabel>
          <DropdownMenuItem onSelect={() => onAdd('projection-partition')}>
            Partition
            <span className="ml-auto text-xs text-subtle-foreground">picks one member</span>
          </DropdownMenuItem>
          <DropdownMenuItem onSelect={() => onAdd('projection-score')}>
            Score
            <span className="ml-auto text-xs text-subtle-foreground">weighs signals</span>
          </DropdownMenuItem>
          <DropdownMenuItem onSelect={() => onAdd('projection-mapping')}>
            Mapping
            <span className="ml-auto text-xs text-subtle-foreground">bands a score</span>
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    )
  }
  return (
    <Tooltip content={`Add a ${meta.one.toLowerCase()}`}>
      <button
        type="button"
        aria-label={`Add ${meta.one.toLowerCase()}`}
        onClick={() => onAdd(group as EntityKind)}
        className={className}
      >
        <Plus className="size-3.5" />
      </button>
    </Tooltip>
  )
}

function OutlineGroup({
  group,
  items,
  total,
  open,
  filtering,
  problems,
  activeKey,
  onToggle,
  onSelect,
  onAdd,
}: {
  group: EditorGroup
  items: EditorEntity[]
  total: number
  open: boolean
  filtering: boolean
  problems: Map<string, DiagLevel>
  activeKey: string | null
  onToggle: () => void
  onSelect: (entity: EditorEntity) => void
  onAdd: (kind: EntityKind) => void
}) {
  const regionId = React.useId()
  const meta = groupMeta[group]
  const flagged = items.some((item) => problems.has(item.key))
  return (
    <div className="mb-0.5">
      <div className="group/header flex items-center gap-0.5 pr-1">
        <button
          type="button"
          onClick={onToggle}
          aria-expanded={open}
          aria-controls={regionId}
          className="flex h-8 min-w-0 flex-1 items-center gap-2 rounded-md px-1.5 text-left text-[13px] font-medium outline-none transition-colors hover:bg-accent focus-visible:ring-[3px] focus-visible:ring-ring/40"
        >
          <ChevronRight
            className={cn(
              'size-3.5 shrink-0 text-subtle-foreground transition-transform duration-200',
              open && 'rotate-90',
            )}
          />
          <IconTile
            icon={meta.icon}
            tone={meta.tone}
            size="xs"
            className="size-5 rounded-[5px] [&_svg]:size-3"
          />
          <span className="flex-1 truncate">{meta.label}</span>
          {flagged && !open ? (
            <span className="size-1.5 rounded-full bg-warning" aria-label="Has problems" />
          ) : null}
          <span className="rounded-full bg-secondary px-1.5 font-mono text-[10.5px] tabular text-muted-foreground">
            {total}
          </span>
        </button>
        <AddButton group={group} onAdd={onAdd} />
      </div>
      {open ? (
        <motion.ul
          id={regionId}
          initial={{ opacity: 0, y: -2 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
          className="relative ml-[13px] border-l border-border py-0.5 pl-1"
        >
          {items.map((item) => {
            const active = item.key === activeKey
            const issue = problems.get(item.key)
            return (
              <li key={item.key} className="relative">
                <button
                  type="button"
                  aria-current={active ? 'page' : undefined}
                  onClick={() => onSelect(item)}
                  className={cn(
                    'relative flex h-7 w-full items-center gap-1.5 rounded-md pl-2 pr-1.5 text-left outline-none transition-colors focus-visible:ring-[3px] focus-visible:ring-ring/40',
                    active
                      ? 'bg-accent text-foreground'
                      : 'text-muted-foreground hover:bg-accent/60 hover:text-foreground',
                  )}
                >
                  <span className="min-w-0 flex-1 truncate font-mono text-[12px]">{item.name}</span>
                  {item.meta ? (
                    <span className="max-w-[45%] shrink-0 truncate text-[10.5px] text-subtle-foreground">
                      {item.meta}
                    </span>
                  ) : null}
                  {issue ? (
                    <span
                      className={cn('size-1.5 shrink-0 rounded-full', problemDot[issue])}
                      aria-label={problemLabel[issue]}
                    />
                  ) : null}
                </button>
              </li>
            )
          })}
          {items.length === 0 ? (
            <li className="px-2 py-1.5 text-xs text-subtle-foreground">
              {filtering ? 'No match' : 'None declared'}
            </li>
          ) : null}
        </motion.ul>
      ) : null}
    </div>
  )
}

/** Entities of the active recipe, grouped by kind, with keyboard-operable entries. */
export function EditorOutline({
  entities,
  problems,
  activeKey,
  target,
  loading,
  onOverview,
  onGlobal,
  onSelect,
  onAdd,
}: EditorOutlineProps) {
  const [query, setQuery] = React.useState('')
  const [collapsed, setCollapsed] = React.useState<Set<EditorGroup>>(() => new Set())
  const needle = query.trim().toLowerCase()
  const matches = (item: EditorEntity) =>
    !needle ||
    item.name.toLowerCase().includes(needle) ||
    (item.meta ?? '').toLowerCase().includes(needle)

  // Selecting an entity from elsewhere (a problem, the overview) opens its group.
  React.useEffect(() => {
    const active = entities.find((item) => item.key === activeKey)
    if (!active) return
    const group = groupOf(active.kind)
    setCollapsed((current) => {
      if (!current.has(group)) return current
      const next = new Set(current)
      next.delete(group)
      return next
    })
  }, [activeKey, entities])

  return (
    <nav aria-label="Config outline" className="flex min-h-0 flex-1 flex-col">
      <div className="flex h-10 shrink-0 items-center justify-between pl-3 pr-2">
        <span className="text-[11px] font-medium uppercase tracking-[0.06em] text-subtle-foreground">
          Outline
        </span>
        <Tooltip content="Collapse all groups">
          <button
            type="button"
            aria-label="Collapse all groups"
            onClick={() => setCollapsed(new Set(GROUP_ORDER))}
            className="grid size-6 place-items-center rounded-md text-subtle-foreground outline-none transition-colors hover:bg-accent hover:text-foreground focus-visible:ring-[3px] focus-visible:ring-ring/40"
          >
            <ChevronsDownUp className="size-3.5" />
          </button>
        </Tooltip>
      </div>
      <div className="shrink-0 px-2.5 pb-2">
        <div className="relative">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-subtle-foreground" />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Escape') setQuery('')
            }}
            placeholder="Filter entities…"
            aria-label="Filter entities"
            className="h-8 w-full rounded-md border border-input bg-surface pl-8 pr-7 text-[12.5px] shadow-xs outline-none transition-[border-color,box-shadow] placeholder:text-subtle-foreground hover:border-border-strong focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/40"
          />
          {query ? (
            <button
              type="button"
              aria-label="Clear filter"
              onClick={() => setQuery('')}
              className="absolute right-1.5 top-1/2 grid size-5 -translate-y-1/2 place-items-center rounded text-subtle-foreground hover:text-foreground"
            >
              <X className="size-3" />
            </button>
          ) : null}
        </div>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto px-1.5 pb-3">
        <div className="mb-1 flex flex-col gap-0.5">
          <OutlineButton
            active={target === 'overview'}
            onClick={onOverview}
            icon={<LayoutDashboard className="size-3.5 text-subtle-foreground" />}
          >
            Overview
          </OutlineButton>
        </div>
        {loading && entities.length === 0 ? (
          <div className="flex flex-col gap-2 px-2 py-2" aria-hidden>
            {[72, 54, 64, 48, 58].map((width, index) => (
              <span
                key={index}
                className="h-3.5 animate-shimmer rounded bg-[linear-gradient(90deg,var(--muted)_0%,var(--surface-subtle)_50%,var(--muted)_100%)] bg-[length:200%_100%]"
                style={{ width: `${width}%` }}
              />
            ))}
          </div>
        ) : (
          GROUP_ORDER.map((group) => {
            const all = entities.filter((item) => groupOf(item.kind) === group)
            const visible = all.filter(matches)
            if (needle && visible.length === 0) return null
            return (
              <OutlineGroup
                key={group}
                group={group}
                items={visible}
                total={all.length}
                open={Boolean(needle) || !collapsed.has(group)}
                filtering={Boolean(needle)}
                problems={problems}
                activeKey={target === 'entity' ? activeKey : null}
                onToggle={() =>
                  setCollapsed((current) => {
                    const next = new Set(current)
                    if (next.has(group)) next.delete(group)
                    else next.add(group)
                    return next
                  })
                }
                onSelect={onSelect}
                onAdd={onAdd}
              />
            )
          })
        )}
        {needle && !entities.some(matches) ? (
          <p className="px-3 py-6 text-center text-xs text-subtle-foreground">
            No entities match “{query}”.
          </p>
        ) : null}
      </div>
      <div className="shrink-0 border-t border-border p-1.5">
        <OutlineButton
          active={target === 'global'}
          onClick={onGlobal}
          icon={<SlidersHorizontal className="size-3.5 text-subtle-foreground" />}
        >
          Router settings
        </OutlineButton>
      </div>
    </nav>
  )
}
