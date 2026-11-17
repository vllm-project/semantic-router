import * as React from 'react'
import { cn } from '../lib/cn'

export const chartAxis = {
  stroke: 'var(--border)',
  tick: { fill: 'var(--subtle-foreground)', fontSize: 11, fontFamily: 'var(--font-sans)' },
  tickLine: false,
  axisLine: false,
} as const

export const chartGrid = { stroke: 'var(--border)', strokeDasharray: '3 4', vertical: false } as const

interface TooltipPayloadItem {
  name?: string
  value?: number | string
  color?: string
  dataKey?: string | number
  payload?: Record<string, unknown>
}

export function ChartTooltip({
  active,
  payload,
  label,
  format = (value) => String(value),
  labelFormat = (value) => String(value),
}: {
  active?: boolean
  payload?: TooltipPayloadItem[]
  label?: string | number
  format?: (value: number | string, name?: string) => string
  labelFormat?: (label: string | number) => string
}) {
  if (!active || !payload?.length) return null
  return (
    <div className="min-w-40 rounded-lg border border-border bg-popover/95 px-3 py-2 text-xs shadow-lg backdrop-blur">
      {label !== undefined ? <div className="mb-1.5 font-medium text-foreground">{labelFormat(label)}</div> : null}
      <div className="flex flex-col gap-1">
        {payload.map((item) => (
          <div key={String(item.dataKey ?? item.name)} className="flex items-center gap-2">
            <span className="size-2 rounded-[3px]" style={{ background: item.color }} />
            <span className="flex-1 text-muted-foreground">{item.name}</span>
            <span className="font-mono font-medium tabular text-foreground">
              {item.value === null || item.value === undefined ? <span className="text-subtle-foreground">unknown</span> : format(item.value, item.name)}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

export function ChartLegend({ items, className }: { items: { label: string; color: string; value?: string }[]; className?: string }) {
  return (
    <div className={cn('flex flex-wrap items-center gap-x-4 gap-y-1.5 text-xs text-muted-foreground', className)}>
      {items.map((item) => (
        <span key={item.label} className="inline-flex items-center gap-1.5">
          <span className="size-2 rounded-[3px]" style={{ background: item.color }} />
          {item.label}
          {item.value ? <span className="font-mono text-foreground/80">{item.value}</span> : null}
        </span>
      ))}
    </div>
  )
}

export function useChartReady() {
  const [ready, setReady] = React.useState(false)
  React.useEffect(() => {
    const id = window.requestAnimationFrame(() => setReady(true))
    return () => window.cancelAnimationFrame(id)
  }, [])
  return ready
}
