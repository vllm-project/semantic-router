import * as React from 'react'
import type { LucideIcon } from 'lucide-react'
import { animate, useInView, useMotionValue, useTransform, motion } from 'framer-motion'
import { ArrowDownRight, ArrowUpRight } from 'lucide-react'
import { cn } from '../lib/cn'
import { Card } from '../components/card'

export type Tone = 'blue' | 'violet' | 'gold' | 'teal' | 'rose' | 'green' | 'orange' | 'slate'

const toneTile: Record<Tone, string> = {
  blue: 'bg-chart-1/12 text-chart-1 ring-chart-1/20',
  gold: 'bg-chart-2/16 text-[oklch(0.62_0.15_68)] ring-chart-2/30 dark:text-chart-2',
  violet: 'bg-chart-3/12 text-chart-3 ring-chart-3/20',
  teal: 'bg-chart-4/14 text-[oklch(0.55_0.1_195)] ring-chart-4/25 dark:text-chart-4',
  rose: 'bg-chart-5/12 text-chart-5 ring-chart-5/20',
  green: 'bg-chart-6/14 text-[oklch(0.55_0.13_152)] ring-chart-6/25 dark:text-chart-6',
  orange: 'bg-chart-7/14 text-[oklch(0.6_0.16_48)] ring-chart-7/25 dark:text-chart-7',
  slate: 'bg-chart-8/12 text-muted-foreground ring-chart-8/20',
}

export const toneText: Record<Tone, string> = {
  blue: 'text-chart-1',
  gold: 'text-chart-2',
  violet: 'text-chart-3',
  teal: 'text-chart-4',
  rose: 'text-chart-5',
  green: 'text-chart-6',
  orange: 'text-chart-7',
  slate: 'text-chart-8',
}

export const toneFill: Record<Tone, string> = {
  blue: 'var(--chart-1)',
  gold: 'var(--chart-2)',
  violet: 'var(--chart-3)',
  teal: 'var(--chart-4)',
  rose: 'var(--chart-5)',
  green: 'var(--chart-6)',
  orange: 'var(--chart-7)',
  slate: 'var(--chart-8)',
}

/** Entity tones keep every surface consistent: a decision is always violet, a signal always gold. */
export const entityTone = {
  recipe: 'blue',
  entrypoint: 'blue',
  decision: 'violet',
  signal: 'gold',
  projection: 'teal',
  model: 'green',
  plugin: 'rose',
  evaluation: 'orange',
  system: 'slate',
} as const satisfies Record<string, Tone>

export function IconTile({
  icon: Icon,
  tone = 'blue',
  size = 'md',
  className,
}: {
  icon: LucideIcon
  tone?: Tone
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl'
  className?: string
}) {
  const box = { xs: 'size-6 rounded-md', sm: 'size-7 rounded-md', md: 'size-9 rounded-lg', lg: 'size-11 rounded-xl', xl: 'size-14 rounded-2xl' }[size]
  const glyph = { xs: 'size-3.5', sm: 'size-4', md: 'size-[18px]', lg: 'size-5', xl: 'size-6' }[size]
  return (
    <span className={cn('grid shrink-0 place-items-center ring-1 ring-inset', box, toneTile[tone], className)}>
      <Icon className={glyph} strokeWidth={1.8} />
    </span>
  )
}

export type StatusTone = 'success' | 'warning' | 'danger' | 'info' | 'neutral'

const statusDot: Record<StatusTone, string> = {
  success: 'bg-success text-success',
  warning: 'bg-warning text-warning',
  danger: 'bg-destructive text-destructive',
  info: 'bg-info text-info',
  neutral: 'bg-subtle-foreground text-subtle-foreground',
}

export function StatusDot({ tone = 'success', pulse = false, className }: { tone?: StatusTone; pulse?: boolean; className?: string }) {
  return (
    <span className={cn('relative inline-flex size-2 shrink-0', className)}>
      {pulse ? <span className={cn('absolute inset-0 rounded-full animate-pulse-ring', statusDot[tone])} /> : null}
      <span className={cn('relative inline-flex size-2 rounded-full', statusDot[tone])} />
    </span>
  )
}

export function StatusLabel({ tone, children, pulse }: { tone: StatusTone; children: React.ReactNode; pulse?: boolean }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-[13px] text-foreground/85">
      <StatusDot tone={tone} pulse={pulse} />
      {children}
    </span>
  )
}

export function NumberTicker({
  value,
  format = (n) => Math.round(n).toLocaleString(),
  duration = 1.1,
  className,
}: {
  value: number
  format?: (value: number) => string
  duration?: number
  className?: string
}) {
  const ref = React.useRef<HTMLSpanElement>(null)
  const inView = useInView(ref, { once: true })
  const motionValue = useMotionValue(0)
  const text = useTransform(motionValue, (latest) => format(latest))
  React.useEffect(() => {
    if (!inView) return
    const controls = animate(motionValue, value, { duration, ease: [0.16, 1, 0.3, 1] })
    return () => controls.stop()
  }, [inView, motionValue, value, duration])
  return (
    <motion.span ref={ref} className={cn('tabular', className)}>
      {text}
    </motion.span>
  )
}

export function Sparkline({
  data,
  color = 'var(--chart-1)',
  height = 36,
  className,
  strokeWidth = 1.6,
}: {
  data: number[]
  color?: string
  height?: number
  className?: string
  strokeWidth?: number
}) {
  const id = React.useId().replace(/:/g, '')
  const width = 120
  const max = Math.max(...data)
  const min = Math.min(...data)
  const span = max - min || 1
  const points = data.map((value, index) => {
    const x = (index / Math.max(1, data.length - 1)) * width
    const y = height - 3 - ((value - min) / span) * (height - 6)
    return [x, y] as const
  })
  const line = points.map(([x, y], index) => `${index === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`).join(' ')
  const area = `${line} L${width},${height} L0,${height} Z`
  return (
    <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" className={cn('w-full overflow-visible', className)} style={{ height }}>
      <defs>
        <linearGradient id={`spark-${id}`} x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity={0.22} />
          <stop offset="100%" stopColor={color} stopOpacity={0} />
        </linearGradient>
      </defs>
      <path d={area} fill={`url(#spark-${id})`} />
      <path d={line} fill="none" stroke={color} strokeWidth={strokeWidth} strokeLinejoin="round" strokeLinecap="round" vectorEffect="non-scaling-stroke" />
    </svg>
  )
}

export function Delta({ value, suffix = '%', invert = false, className }: { value: number; suffix?: string; invert?: boolean; className?: string }) {
  const positive = value >= 0
  const good = invert ? !positive : positive
  const Icon = positive ? ArrowUpRight : ArrowDownRight
  return (
    <span
      className={cn(
        'inline-flex items-center gap-0.5 rounded-full px-1.5 py-0.5 text-[11px] font-semibold tabular',
        good ? 'bg-success-soft text-success-foreground' : 'bg-destructive-soft text-destructive-foreground',
        className,
      )}
    >
      <Icon className="size-3" strokeWidth={2.4} />
      {Math.abs(value).toFixed(1)}
      {suffix}
    </span>
  )
}

export function StatCard({
  label,
  value,
  format,
  unit,
  delta,
  deltaInvert,
  hint,
  icon,
  spark,
  sparkColor,
  className,
  onClick,
}: {
  label: string
  value: number
  format?: (value: number) => string
  unit?: string
  delta?: number
  deltaInvert?: boolean
  hint?: React.ReactNode
  icon?: React.ReactNode
  spark?: number[]
  sparkColor?: string
  className?: string
  onClick?: () => void
}) {
  return (
    <Card interactive={Boolean(onClick)} onClick={onClick} className={cn('overflow-hidden', className)}>
      <div className="flex flex-col gap-3 px-4 pb-3 pt-4">
        <div className="flex items-center justify-between gap-2">
          <span className="flex items-center gap-2 text-[13px] font-medium text-muted-foreground">
            {icon}
            {label}
          </span>
          {delta !== undefined ? <Delta value={delta} invert={deltaInvert} /> : null}
        </div>
        <div className="flex items-baseline gap-1.5">
          <NumberTicker value={value} format={format} className="text-[26px] font-semibold leading-none tracking-[-0.03em]" />
          {unit ? <span className="text-[13px] font-medium text-subtle-foreground">{unit}</span> : null}
        </div>
        {hint ? <div className="text-xs text-muted-foreground">{hint}</div> : null}
      </div>
      {spark ? <Sparkline data={spark} color={sparkColor} height={38} className="-mb-px" /> : null}
    </Card>
  )
}

export function KeyValueGrid({
  items,
  columns = 2,
  className,
}: {
  items: { label: React.ReactNode; value: React.ReactNode; mono?: boolean; span?: boolean }[]
  columns?: 1 | 2 | 3
  className?: string
}) {
  return (
    <dl
      className={cn(
        'grid gap-x-6 gap-y-4',
        columns === 1 && 'grid-cols-1',
        columns === 2 && 'grid-cols-1 sm:grid-cols-2',
        columns === 3 && 'grid-cols-1 sm:grid-cols-3',
        className,
      )}
    >
      {items.map((item, index) => (
        <div key={index} className={cn('flex min-w-0 flex-col gap-1', item.span && 'sm:col-span-full')}>
          <dt className="text-xs text-subtle-foreground">{item.label}</dt>
          <dd className={cn('min-w-0 text-[13px] text-foreground', item.mono && 'font-mono text-[12.5px]')}>{item.value}</dd>
        </div>
      ))}
    </dl>
  )
}

export type UptimeSample = 'ok' | 'degraded' | 'down' | 'none'

export function UptimeBar({ samples, className }: { samples: UptimeSample[]; className?: string }) {
  const color: Record<UptimeSample, string> = {
    ok: 'bg-success/80 hover:bg-success',
    degraded: 'bg-warning/85 hover:bg-warning',
    down: 'bg-destructive/85 hover:bg-destructive',
    none: 'bg-muted hover:bg-border-strong',
  }
  return (
    <div className={cn('flex h-7 items-stretch gap-[2px]', className)}>
      {samples.map((sample, index) => (
        <span
          key={index}
          title={`${samples.length - index}h ago · ${sample}`}
          className={cn('min-w-[2px] flex-1 rounded-[2px] transition-colors', color[sample])}
        />
      ))}
    </div>
  )
}

export function MiniBar({ value, max = 100, tone = 'blue', className }: { value: number; max?: number; tone?: Tone; className?: string }) {
  return (
    <span className={cn('relative block h-1.5 w-full overflow-hidden rounded-full bg-muted', className)}>
      <motion.span
        className="absolute inset-y-0 left-0 rounded-full"
        style={{ background: toneFill[tone] }}
        initial={{ width: 0 }}
        animate={{ width: `${Math.min(100, (value / max) * 100)}%` }}
        transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1] }}
      />
    </span>
  )
}
