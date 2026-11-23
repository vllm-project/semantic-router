import * as React from 'react'
import { animate, motion, useMotionValue, useTransform } from 'framer-motion'
import { cn } from '../lib/cn'
import { toneFill, type Tone } from './data-display'

const ease = [0.16, 1, 0.3, 1] as const

/** A 0–1 score bar with an optional threshold tick; the side of the threshold that matches is tinted. */
export function ScoreMeter({
  value,
  threshold,
  direction = 'above',
  tone = 'gold',
  matched,
  className,
  'aria-label': ariaLabel,
}: {
  value: number
  threshold?: number
  direction?: 'above' | 'below'
  tone?: Tone
  matched?: boolean
  className?: string
  'aria-label'?: string
}) {
  const pct = Math.max(0, Math.min(1, value)) * 100
  const tick = threshold === undefined ? undefined : Math.max(0, Math.min(1, threshold)) * 100
  return (
    <div
      className={cn('relative py-1', className)}
      role="meter"
      aria-label={ariaLabel}
      aria-valuemin={0}
      aria-valuemax={1}
      aria-valuenow={Number(value.toFixed(3))}
    >
      <div className="relative h-2 overflow-hidden rounded-full bg-muted">
        {tick !== undefined ? (
          <span
            className="absolute inset-y-0"
            style={{
              left: direction === 'above' ? `${tick}%` : 0,
              right: direction === 'above' ? 0 : `${100 - tick}%`,
              background: toneFill[tone],
              opacity: 0.14,
            }}
          />
        ) : null}
        <motion.span
          className="absolute inset-y-0 left-0 rounded-full"
          style={{ background: matched === false ? 'var(--chart-8)' : toneFill[tone] }}
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.7, ease }}
        />
      </div>
      {tick !== undefined ? (
        <span
          className="absolute top-0 h-4 w-[3px] -translate-x-1/2 rounded-full border border-surface bg-foreground/80"
          style={{ left: `${tick}%` }}
        />
      ) : null}
    </div>
  )
}

/** A number that animates to each new value (NumberTicker animates once, from zero). */
export function AnimatedNumber({
  value,
  format = (n) => n.toFixed(2),
  className,
}: {
  value: number
  format?: (value: number) => string
  className?: string
}) {
  const motionValue = useMotionValue(value)
  const text = useTransform(motionValue, (latest) => format(latest))
  React.useEffect(() => {
    const controls = animate(motionValue, value, { duration: 0.6, ease })
    return () => controls.stop()
  }, [motionValue, value])
  return <motion.span className={cn('tabular-nums', className)}>{text}</motion.span>
}

/** Text with marked ranges, such as the keywords a rule matched; overlapping ranges merge. */
export function HighlightedText({
  text,
  ranges,
  className,
}: {
  text: string
  ranges: readonly { start: number; end: number }[]
  className?: string
}) {
  const merged = [...ranges]
    .sort((a, b) => a.start - b.start)
    .reduce<{ start: number; end: number }[]>((out, range) => {
      const last = out[out.length - 1]
      if (last && range.start <= last.end) last.end = Math.max(last.end, range.end)
      else out.push({ start: range.start, end: range.end })
      return out
    }, [])
  const parts: React.ReactNode[] = []
  let cursor = 0
  merged.forEach((range, index) => {
    if (range.start > cursor) parts.push(text.slice(cursor, range.start))
    parts.push(
      <mark
        key={index}
        className="rounded-[4px] bg-chart-2/22 px-0.5 font-medium text-foreground ring-1 ring-chart-2/50 ring-inset"
      >
        {text.slice(range.start, range.end)}
      </mark>,
    )
    cursor = range.end
  })
  if (cursor < text.length) parts.push(text.slice(cursor))
  return (
    <p className={cn('text-[13px] leading-relaxed break-words whitespace-pre-wrap', className)}>
      {parts}
    </p>
  )
}
