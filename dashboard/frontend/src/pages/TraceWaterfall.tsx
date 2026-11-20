import { useMemo, useRef, useState, type KeyboardEvent } from 'react'
import { ChevronRight, CircleAlert } from 'lucide-react'

import { cn } from '../ui'
import { formatMicros, serviceColor, type TraceSpan } from './jaegerTraces'
import { timelineTicks, visibleRows, type Waterfall, type WaterfallRow } from './traceWaterfall'

const COLUMNS = 'grid grid-cols-[minmax(13rem,38%)_minmax(0,1fr)]'

function tickStyle(fraction: number) {
  const translate = fraction === 0 ? '0' : fraction === 1 ? '-100%' : '-50%'
  return { left: `${fraction * 100}%`, transform: `translate(${translate}, -50%)` }
}

/** The right gutter leaves room for the duration label of a bar that ends at the edge. */
const TIMELINE = 'relative mr-16'

/**
 * The span tree of one trace with a timing bar per span. Arrow keys move the
 * selection, Right and Left expand and collapse, and a chevron click toggles.
 */
export function TraceWaterfall({
  waterfall,
  palette,
  selected,
  onSelect,
  className,
}: {
  waterfall: Waterfall
  palette: readonly string[]
  selected: TraceSpan | null
  onSelect: (span: TraceSpan) => void
  className?: string
}) {
  const [collapsed, setCollapsed] = useState<ReadonlySet<string>>(() => new Set())
  const rows = useMemo(() => visibleRows(waterfall.rows, collapsed), [waterfall.rows, collapsed])
  const items = useRef(new Map<number, HTMLDivElement>())
  const selectedIndex = rows.findIndex((row) => row.span === selected)
  const ticks = timelineTicks(waterfall.duration)

  const toggle = (spanID: string) =>
    setCollapsed((current) => {
      const next = new Set(current)
      if (next.has(spanID)) next.delete(spanID)
      else next.add(spanID)
      return next
    })

  const focusRow = (index: number) => {
    const row = rows[index]
    if (!row) return
    onSelect(row.span)
    items.current.get(index)?.focus()
  }

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>, index: number, row: WaterfallRow) => {
    const isCollapsed = collapsed.has(row.span.spanID)
    switch (event.key) {
      case 'ArrowDown':
        focusRow(index + 1)
        break
      case 'ArrowUp':
        focusRow(index - 1)
        break
      case 'Home':
        focusRow(0)
        break
      case 'End':
        focusRow(rows.length - 1)
        break
      case 'ArrowRight':
        if (row.childCount && isCollapsed) toggle(row.span.spanID)
        else if (row.childCount) focusRow(index + 1)
        break
      case 'ArrowLeft':
        if (row.childCount && !isCollapsed) toggle(row.span.spanID)
        else {
          const parent = row.ancestors[row.ancestors.length - 1]
          focusRow(rows.findIndex((candidate) => candidate.span.spanID === parent))
        }
        break
      case 'Enter':
      case ' ':
        onSelect(row.span)
        break
      default:
        return
    }
    event.preventDefault()
  }

  return (
    <div className={cn('flex min-h-0 flex-col', className)}>
      <div
        className={cn(
          COLUMNS,
          'h-8 shrink-0 items-center border-b border-border bg-surface-subtle/70 text-[11px] text-subtle-foreground',
        )}
      >
        <span className="px-4 font-medium">Service and operation</span>
        <span className={cn(TIMELINE, 'h-full')} aria-hidden="true">
          {ticks.map((tick) => (
            <span
              key={tick.fraction}
              className="absolute top-1/2 whitespace-nowrap tabular"
              style={tickStyle(tick.fraction)}
            >
              {tick.fraction === 0 ? '0' : formatMicros(tick.micros)}
            </span>
          ))}
        </span>
      </div>
      <div role="tree" aria-label="Spans" className="min-h-0 flex-1 overflow-y-auto">
        {rows.map((row, index) => {
          const isSelected = row.span === selected
          const isCollapsed = collapsed.has(row.span.spanID)
          const color = serviceColor(palette, row.span.serviceName)
          return (
            <div
              key={`${index}:${row.span.spanID}`}
              ref={(element) => {
                if (element) items.current.set(index, element)
                else items.current.delete(index)
              }}
              role="treeitem"
              aria-level={row.depth + 1}
              aria-expanded={row.childCount ? !isCollapsed : undefined}
              aria-selected={isSelected}
              tabIndex={isSelected || (selectedIndex < 0 && index === 0) ? 0 : -1}
              onClick={(event) => {
                if ((event.target as HTMLElement).closest('[data-toggle]')) toggle(row.span.spanID)
                else onSelect(row.span)
              }}
              onKeyDown={(event) => onKeyDown(event, index, row)}
              className={cn(
                COLUMNS,
                'cursor-pointer items-center border-b border-border/60 text-[12.5px] outline-none transition-colors hover:bg-surface-subtle focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring/50',
                isSelected && 'bg-primary-soft/50 hover:bg-primary-soft/60',
              )}
            >
              <span
                className="flex min-w-0 items-center gap-1.5 py-1.5 pr-3"
                style={{ paddingLeft: 12 + Math.min(row.depth, 16) * 14 }}
              >
                {row.childCount ? (
                  <span
                    data-toggle=""
                    aria-hidden="true"
                    className="grid size-4 shrink-0 place-items-center rounded text-subtle-foreground hover:bg-accent hover:text-foreground"
                  >
                    <ChevronRight
                      className={cn('size-3.5 transition-transform', !isCollapsed && 'rotate-90')}
                    />
                  </span>
                ) : (
                  <span className="size-4 shrink-0" aria-hidden="true" />
                )}
                <span
                  className="h-3.5 w-1 shrink-0 rounded-full"
                  style={{ background: color }}
                  aria-hidden="true"
                />
                <span
                  className="max-w-[50%] shrink-0 truncate font-medium"
                  title={row.span.serviceName}
                >
                  {row.span.serviceName}
                </span>
                <span className="truncate text-muted-foreground" title={row.span.operationName}>
                  {row.span.operationName}
                </span>
                {row.span.error ? (
                  <CircleAlert className="size-3.5 shrink-0 text-destructive" aria-label="Error" />
                ) : null}
                {isCollapsed && row.descendants ? (
                  <span className="shrink-0 rounded bg-muted px-1 text-[10.5px] text-muted-foreground tabular">
                    +{row.descendants}
                  </span>
                ) : null}
              </span>
              <span className={cn(TIMELINE, 'h-7')}>
                <span
                  className="absolute top-1/2 h-2.5 -translate-y-1/2 rounded-[3px]"
                  style={{
                    left: `${row.offset * 100}%`,
                    width: `max(2px, ${row.width * 100}%)`,
                    background: row.span.error ? 'var(--destructive)' : color,
                  }}
                />
                <span
                  className="absolute top-1/2 -translate-y-1/2 whitespace-nowrap text-[11px] text-muted-foreground tabular"
                  style={{ left: `calc(${(row.offset + row.width) * 100}% + 6px)` }}
                >
                  {formatMicros(row.span.duration)}
                </span>
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
