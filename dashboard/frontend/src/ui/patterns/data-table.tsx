import * as React from 'react'
import { createPortal } from 'react-dom'
import { AnimatePresence, motion } from 'framer-motion'
import {
  ArrowDown,
  ArrowUp,
  Check,
  ChevronsUpDown,
  MoreHorizontal,
  PlusCircle,
  Search,
  X,
} from 'lucide-react'
import { cn } from '../lib/cn'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '../components/table'
import { Checkbox } from '../components/form-controls'
import { Button } from '../components/button'
import { Kbd, Separator } from '../components/misc'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuTrigger,
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '../components/overlays'
import { TableSkeleton } from './feedback'

export interface Column<T> {
  id: string
  header: React.ReactNode
  cell: (row: T) => React.ReactNode
  sortValue?: (row: T) => string | number
  align?: 'left' | 'right' | 'center'
  className?: string
  headerClassName?: string
  width?: string
  hideBelow?: 'sm' | 'md' | 'lg' | 'xl' | '2xl'
}

const hideClass = {
  sm: 'max-sm:hidden',
  md: 'max-md:hidden',
  lg: 'max-lg:hidden',
  xl: 'max-xl:hidden',
  '2xl': 'max-2xl:hidden',
}

export function DataTable<T>({
  data,
  columns,
  getRowId,
  onRowClick,
  selectable = false,
  selected,
  onSelectedChange,
  rowActions,
  empty,
  loading = false,
  activeRowId,
  className,
}: {
  data: T[]
  columns: Column<T>[]
  getRowId: (row: T) => string
  onRowClick?: (row: T) => void
  selectable?: boolean
  selected?: Set<string>
  onSelectedChange?: (next: Set<string>) => void
  rowActions?: (row: T) => React.ReactNode
  empty?: React.ReactNode
  loading?: boolean
  activeRowId?: string
  className?: string
}) {
  const [sort, setSort] = React.useState<{ id: string; dir: 'asc' | 'desc' } | null>(null)
  const sorted = React.useMemo(() => {
    if (!sort) return data
    const column = columns.find((c) => c.id === sort.id)
    if (!column?.sortValue) return data
    const value = column.sortValue
    return [...data].sort((a, b) => {
      const av = value(a)
      const bv = value(b)
      const cmp =
        typeof av === 'number' && typeof bv === 'number'
          ? av - bv
          : String(av).localeCompare(String(bv))
      return sort.dir === 'asc' ? cmp : -cmp
    })
  }, [data, columns, sort])

  const ids = sorted.map(getRowId)
  const selectedSet = selected ?? new Set<string>()
  const allSelected = ids.length > 0 && ids.every((id) => selectedSet.has(id))
  const someSelected = ids.some((id) => selectedSet.has(id))

  const toggleAll = () => onSelectedChange?.(allSelected ? new Set() : new Set(ids))
  const toggleOne = (id: string) => {
    const next = new Set(selectedSet)
    if (next.has(id)) next.delete(id)
    else next.add(id)
    onSelectedChange?.(next)
  }

  return (
    <div
      className={cn('overflow-hidden rounded-xl border border-border bg-card shadow-xs', className)}
    >
      <Table>
        <TableHeader className="bg-surface-subtle/80">
          <TableRow>
            {selectable ? (
              <TableHead className="w-10">
                <Checkbox
                  aria-label="Select all"
                  checked={allSelected ? true : someSelected ? 'indeterminate' : false}
                  onCheckedChange={toggleAll}
                />
              </TableHead>
            ) : null}
            {columns.map((column) => {
              const active = sort?.id === column.id
              return (
                <TableHead
                  key={column.id}
                  style={column.width ? { width: column.width } : undefined}
                  className={cn(
                    column.align === 'right' && 'text-right',
                    column.align === 'center' && 'text-center',
                    column.hideBelow && hideClass[column.hideBelow],
                    column.headerClassName,
                  )}
                >
                  {column.sortValue ? (
                    <button
                      type="button"
                      onClick={() =>
                        setSort((current) =>
                          current?.id !== column.id
                            ? { id: column.id, dir: 'desc' }
                            : current.dir === 'desc'
                              ? { id: column.id, dir: 'asc' }
                              : null,
                        )
                      }
                      className={cn(
                        '-mx-1 inline-flex items-center gap-1 rounded px-1 py-0.5 transition-colors hover:text-foreground',
                        active && 'text-foreground',
                        column.align === 'right' && 'flex-row-reverse',
                      )}
                    >
                      {column.header}
                      {active ? (
                        sort?.dir === 'asc' ? (
                          <ArrowUp className="size-3" />
                        ) : (
                          <ArrowDown className="size-3" />
                        )
                      ) : (
                        <ChevronsUpDown className="size-3 opacity-40" />
                      )}
                    </button>
                  ) : (
                    column.header
                  )}
                </TableHead>
              )
            })}
            {rowActions ? <TableHead className="w-12" /> : null}
          </TableRow>
        </TableHeader>
        {loading ? null : (
          <TableBody>
            {sorted.map((row, index) => {
              const id = getRowId(row)
              const isSelected = selectedSet.has(id)
              return (
                <motion.tr
                  key={id}
                  initial={{ opacity: 0, y: 4 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{
                    delay: Math.min(index, 12) * 0.018,
                    duration: 0.3,
                    ease: [0.16, 1, 0.3, 1],
                  }}
                  data-state={isSelected || activeRowId === id ? 'selected' : undefined}
                  onClick={onRowClick ? () => onRowClick(row) : undefined}
                  className={cn(
                    'group border-b border-border transition-colors last:border-0 hover:bg-surface-subtle data-[state=selected]:bg-primary-soft/50',
                    onRowClick && 'cursor-pointer',
                  )}
                >
                  {selectable ? (
                    <TableCell className="w-10" onClick={(event) => event.stopPropagation()}>
                      <Checkbox
                        aria-label="Select row"
                        checked={isSelected}
                        onCheckedChange={() => toggleOne(id)}
                      />
                    </TableCell>
                  ) : null}
                  {columns.map((column) => (
                    <TableCell
                      key={column.id}
                      className={cn(
                        column.align === 'right' && 'text-right',
                        column.align === 'center' && 'text-center',
                        column.hideBelow && hideClass[column.hideBelow],
                        column.className,
                      )}
                    >
                      {column.cell(row)}
                    </TableCell>
                  ))}
                  {rowActions ? (
                    <TableCell
                      className="w-12 pr-3 text-right"
                      onClick={(event) => event.stopPropagation()}
                    >
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button
                            variant="ghost"
                            size="icon-xs"
                            aria-label="Row actions"
                            className="opacity-60 group-hover:opacity-100 data-[state=open]:opacity-100"
                          >
                            <MoreHorizontal />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end" className="w-44">
                          {rowActions(row)}
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </TableCell>
                  ) : null}
                </motion.tr>
              )
            })}
          </TableBody>
        )}
      </Table>
      {loading ? <TableSkeleton columns={Math.min(columns.length, 5)} /> : null}
      {!loading && sorted.length === 0 ? empty : null}
    </div>
  )
}

export function DataToolbar({
  search,
  onSearchChange,
  placeholder = 'Search…',
  filters,
  actions,
  className,
}: {
  search?: string
  onSearchChange?: (value: string) => void
  placeholder?: string
  filters?: React.ReactNode
  actions?: React.ReactNode
  className?: string
}) {
  return (
    <div className={cn('mb-3 flex flex-wrap items-center gap-2', className)}>
      {onSearchChange ? (
        <div className="relative">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-subtle-foreground" />
          <input
            value={search}
            onChange={(event) => onSearchChange(event.target.value)}
            placeholder={placeholder}
            className="h-8 w-64 rounded-md border border-input bg-surface pl-8 pr-8 text-[13px] shadow-xs outline-none transition-[border-color,box-shadow,width] duration-200 placeholder:text-subtle-foreground hover:border-border-strong focus:w-80 focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/40"
          />
          {search ? (
            <button
              type="button"
              aria-label="Clear search"
              onClick={() => onSearchChange('')}
              className="absolute right-2 top-1/2 grid size-5 -translate-y-1/2 place-items-center rounded text-subtle-foreground hover:text-foreground"
            >
              <X className="size-3.5" />
            </button>
          ) : (
            <Kbd className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2">/</Kbd>
          )}
        </div>
      ) : null}
      {filters}
      <div className="ml-auto flex items-center gap-2">{actions}</div>
    </div>
  )
}

export interface FilterOption {
  value: string
  label: string
  count?: number
  icon?: React.ReactNode
}

export function FilterPill({
  label,
  options,
  selected,
  onChange,
}: {
  label: string
  options: FilterOption[]
  selected: string[]
  onChange: (next: string[]) => void
}) {
  const [query, setQuery] = React.useState('')
  const visible = options.filter((option) =>
    option.label.toLowerCase().includes(query.toLowerCase()),
  )
  const selectedLabels = options.filter((option) => selected.includes(option.value))
  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          type="button"
          className={cn(
            'inline-flex h-8 items-center gap-1.5 rounded-md border border-dashed border-border-strong bg-transparent px-2.5 text-[13px] font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-foreground',
            selected.length > 0 && 'border-solid bg-surface text-foreground shadow-xs',
          )}
        >
          <PlusCircle
            className={cn('size-3.5', selected.length > 0 && 'rotate-45 transition-transform')}
          />
          {label}
          {selectedLabels.length > 0 ? (
            <>
              <Separator orientation="vertical" className="mx-0.5 h-4" />
              {selectedLabels.length > 2 ? (
                <span className="rounded bg-secondary px-1.5 text-xs">
                  {selectedLabels.length} selected
                </span>
              ) : (
                selectedLabels.map((option) => (
                  <span key={option.value} className="rounded bg-secondary px-1.5 text-xs">
                    {option.label}
                  </span>
                ))
              )}
            </>
          ) : null}
        </button>
      </PopoverTrigger>
      <PopoverContent align="start" className="w-60 p-0">
        <div className="border-b border-border p-1.5">
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder={`Filter ${label.toLowerCase()}…`}
            className="h-8 w-full rounded-md bg-transparent px-2 text-[13px] outline-none placeholder:text-subtle-foreground"
          />
        </div>
        <div className="max-h-64 overflow-y-auto p-1">
          {visible.map((option) => {
            const checked = selected.includes(option.value)
            return (
              <button
                key={option.value}
                type="button"
                role="menuitemcheckbox"
                aria-checked={checked}
                onClick={() =>
                  onChange(
                    checked
                      ? selected.filter((value) => value !== option.value)
                      : [...selected, option.value],
                  )
                }
                className="flex h-8 w-full items-center gap-2 rounded-md px-2 text-left text-[13px] hover:bg-accent"
              >
                <span
                  aria-hidden
                  className={cn(
                    'grid size-4 shrink-0 place-items-center rounded-[5px] border shadow-xs transition-colors',
                    checked
                      ? 'border-primary bg-primary text-primary-foreground'
                      : 'border-border-strong bg-surface',
                  )}
                >
                  {checked ? <Check className="size-3" strokeWidth={3} /> : null}
                </span>
                {option.icon}
                <span className="flex-1 truncate">{option.label}</span>
                {option.count !== undefined ? (
                  <span className="font-mono text-[11px] text-subtle-foreground">
                    {option.count}
                  </span>
                ) : null}
              </button>
            )
          })}
        </div>
        {selected.length > 0 ? (
          <div className="border-t border-border p-1">
            <button
              type="button"
              onClick={() => onChange([])}
              className="h-8 w-full rounded-md text-[13px] text-muted-foreground hover:bg-accent hover:text-foreground"
            >
              Clear filter
            </button>
          </div>
        ) : null}
      </PopoverContent>
    </Popover>
  )
}

export function BulkBar({
  count,
  onClear,
  children,
}: {
  count: number
  onClear: () => void
  children: React.ReactNode
}) {
  if (typeof document === 'undefined') return null
  return createPortal(
    <AnimatePresence>
      {count > 0 ? (
        <motion.div
          initial={{ y: 24, opacity: 0, x: '-50%' }}
          animate={{ y: 0, opacity: 1, x: '-50%' }}
          exit={{ y: 24, opacity: 0, x: '-50%' }}
          transition={{ type: 'spring', bounce: 0.2, duration: 0.45 }}
          className="fixed bottom-6 left-1/2 z-40 flex items-center gap-2 rounded-xl border border-border bg-popover py-1.5 pl-3 pr-1.5 shadow-xl"
        >
          <span className="text-[13px] font-medium tabular">{count} selected</span>
          <Separator orientation="vertical" className="mx-1 h-5" />
          {children}
          <Button variant="ghost" size="icon-sm" onClick={onClear} aria-label="Clear selection">
            <X />
          </Button>
        </motion.div>
      ) : null}
    </AnimatePresence>,
    document.body,
  )
}
