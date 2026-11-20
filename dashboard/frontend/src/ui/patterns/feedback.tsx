import * as React from 'react'
import type { LucideIcon } from 'lucide-react'
import { motion } from 'framer-motion'
import { cn } from '../lib/cn'
import { IconTile, type Tone } from './data-display'
import { Skeleton } from '../components/misc'

export function EmptyState({
  icon,
  tone = 'blue',
  title,
  description,
  actions,
  footer,
  className,
  compact = false,
}: {
  icon: LucideIcon
  tone?: Tone
  title: React.ReactNode
  description?: React.ReactNode
  actions?: React.ReactNode
  footer?: React.ReactNode
  className?: string
  compact?: boolean
}) {
  return (
    <div
      className={cn(
        'flex flex-col items-center justify-center text-center',
        compact ? 'gap-3 py-10' : 'gap-4 py-16',
        className,
      )}
    >
      <div className="relative grid place-items-center">
        {[0, 1, 2].map((ring) => (
          <motion.span
            key={ring}
            className="absolute rounded-full border border-border"
            style={{ width: 64 + ring * 36, height: 64 + ring * 36 }}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1 - ring * 0.3, scale: 1 }}
            transition={{ delay: 0.05 * ring, duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
          />
        ))}
        <IconTile icon={icon} tone={tone} size="lg" className="relative bg-surface shadow-sm" />
      </div>
      <div className={cn('flex max-w-md flex-col gap-1.5', compact ? 'mt-3' : 'mt-6')}>
        <h3 className="text-[15px] font-semibold tracking-[-0.01em]">{title}</h3>
        {description ? (
          <p className="text-[13px] leading-relaxed text-muted-foreground">{description}</p>
        ) : null}
      </div>
      {actions ? (
        <div className="flex flex-wrap items-center justify-center gap-2">{actions}</div>
      ) : null}
      {footer ? <div className="text-xs text-subtle-foreground">{footer}</div> : null}
    </div>
  )
}

export function TableSkeleton({ rows = 6, columns = 5 }: { rows?: number; columns?: number }) {
  return (
    <div className="flex flex-col divide-y divide-border">
      {Array.from({ length: rows }).map((_, row) => (
        <div key={row} className="flex items-center gap-6 px-4 py-3.5">
          {Array.from({ length: columns }).map((__, column) => (
            <Skeleton key={column} className={cn('h-3.5', column === 0 ? 'w-40' : 'flex-1')} />
          ))}
        </div>
      ))}
    </div>
  )
}
