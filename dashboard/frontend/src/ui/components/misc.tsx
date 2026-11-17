import * as React from 'react'
import {
  Avatar as AvatarPrimitive,
  Collapsible as CollapsiblePrimitive,
  Progress as ProgressPrimitive,
  ScrollArea as ScrollAreaPrimitive,
  Separator as SeparatorPrimitive,
  ToggleGroup as ToggleGroupPrimitive,
} from 'radix-ui'
import { cva, type VariantProps } from 'class-variance-authority'
import { AlertTriangle, CheckCircle2, Info, OctagonAlert } from 'lucide-react'
import { cn } from '../lib/cn'

export function Skeleton({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      data-slot="skeleton"
      className={cn(
        'animate-shimmer rounded-md bg-[linear-gradient(90deg,var(--muted)_0%,var(--surface-subtle)_50%,var(--muted)_100%)] bg-[length:200%_100%]',
        className,
      )}
      {...props}
    />
  )
}

export function Separator({
  className,
  orientation = 'horizontal',
  ...props
}: React.ComponentPropsWithoutRef<typeof SeparatorPrimitive.Root>) {
  return (
    <SeparatorPrimitive.Root
      orientation={orientation}
      className={cn(
        'shrink-0 bg-border',
        orientation === 'horizontal' ? 'h-px w-full' : 'h-full w-px',
        className,
      )}
      {...props}
    />
  )
}

export function Avatar({
  className,
  src,
  name,
  size = 28,
}: {
  className?: string
  src?: string
  name: string
  size?: number
}) {
  const initials = name
    .split(/[\s@._-]+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase())
    .join('')
  return (
    <AvatarPrimitive.Root
      className={cn(
        'relative inline-flex shrink-0 overflow-hidden rounded-full ring-1 ring-border',
        className,
      )}
      style={{ width: size, height: size }}
    >
      {src ? <AvatarPrimitive.Image src={src} alt="" className="size-full object-cover" /> : null}
      <AvatarPrimitive.Fallback
        className="grid size-full place-items-center bg-[linear-gradient(135deg,var(--brand-400),var(--brand-700))] font-semibold text-white"
        style={{ fontSize: Math.max(10, Math.round(size * 0.38)) }}
      >
        {initials || '?'}
      </AvatarPrimitive.Fallback>
    </AvatarPrimitive.Root>
  )
}

export function Kbd({ className, ...props }: React.HTMLAttributes<HTMLElement>) {
  return (
    <kbd
      className={cn(
        'inline-flex h-5 min-w-5 select-none items-center justify-center gap-0.5 rounded-[5px] border border-border bg-surface-subtle px-1 font-sans text-[11px] font-medium text-muted-foreground shadow-[0_1px_0_var(--border)]',
        className,
      )}
      {...props}
    />
  )
}

export function Progress({
  className,
  value = 0,
  tone = 'brand',
  ...props
}: React.ComponentPropsWithoutRef<typeof ProgressPrimitive.Root> & {
  tone?: 'brand' | 'success' | 'warning' | 'danger' | 'neutral'
}) {
  const fill = {
    brand: 'bg-primary',
    success: 'bg-success',
    warning: 'bg-warning',
    danger: 'bg-destructive',
    neutral: 'bg-foreground/60',
  }[tone]
  return (
    <ProgressPrimitive.Root
      className={cn('relative h-1.5 w-full overflow-hidden rounded-full bg-muted', className)}
      value={value}
      {...props}
    >
      <ProgressPrimitive.Indicator
        className={cn(
          'h-full rounded-full transition-[width] duration-700 ease-[cubic-bezier(0.16,1,0.3,1)]',
          fill,
        )}
        style={{ width: `${Math.max(0, Math.min(100, value ?? 0))}%` }}
      />
    </ProgressPrimitive.Root>
  )
}

export function ScrollArea({
  className,
  children,
  ...props
}: React.ComponentPropsWithoutRef<typeof ScrollAreaPrimitive.Root>) {
  return (
    <ScrollAreaPrimitive.Root className={cn('relative overflow-hidden', className)} {...props}>
      <ScrollAreaPrimitive.Viewport className="size-full rounded-[inherit]">
        {children}
      </ScrollAreaPrimitive.Viewport>
      <ScrollAreaPrimitive.Scrollbar
        orientation="vertical"
        className="flex w-2 touch-none select-none p-px transition-colors"
      >
        <ScrollAreaPrimitive.Thumb className="relative flex-1 rounded-full bg-border-strong" />
      </ScrollAreaPrimitive.Scrollbar>
      <ScrollAreaPrimitive.Corner />
    </ScrollAreaPrimitive.Root>
  )
}

export const Collapsible = CollapsiblePrimitive.Root
export const CollapsibleTrigger = CollapsiblePrimitive.Trigger
export function CollapsibleContent({
  className,
  ...props
}: React.ComponentPropsWithoutRef<typeof CollapsiblePrimitive.Content>) {
  return (
    <CollapsiblePrimitive.Content
      className={cn(
        'overflow-hidden data-[state=closed]:animate-collapsible-up data-[state=open]:animate-collapsible-down',
        className,
      )}
      {...props}
    />
  )
}

/* Segmented control built on ToggleGroup (single selection). */
export function Segmented<T extends string>({
  value,
  onValueChange,
  options,
  size = 'default',
  className,
  'aria-label': ariaLabel,
}: {
  value: T
  onValueChange: (value: T) => void
  options: { value: T; label: React.ReactNode; icon?: React.ReactNode; disabled?: boolean }[]
  size?: 'sm' | 'default'
  className?: string
  'aria-label'?: string
}) {
  return (
    <ToggleGroupPrimitive.Root
      type="single"
      value={value}
      aria-label={ariaLabel}
      onValueChange={(next) => {
        if (next) onValueChange(next as T)
      }}
      className={cn(
        'inline-flex items-center gap-0.5 rounded-lg border border-border bg-surface-subtle p-0.5',
        className,
      )}
    >
      {options.map((option) => (
        <ToggleGroupPrimitive.Item
          key={option.value}
          value={option.value}
          disabled={option.disabled}
          className={cn(
            'inline-flex items-center gap-1.5 rounded-md px-2.5 font-medium text-muted-foreground outline-none transition-[color,background-color,box-shadow] hover:text-foreground focus-visible:ring-[3px] focus-visible:ring-ring/40 disabled:pointer-events-none disabled:opacity-40',
            'data-[state=on]:bg-surface data-[state=on]:text-foreground data-[state=on]:shadow-sm dark:data-[state=on]:bg-accent [&_svg]:size-3.5',
            size === 'sm' ? 'h-6 text-xs' : 'h-7 text-[13px]',
          )}
        >
          {option.icon}
          {option.label}
        </ToggleGroupPrimitive.Item>
      ))}
    </ToggleGroupPrimitive.Root>
  )
}

const alertVariants = cva(
  'relative flex gap-3 rounded-lg border px-3.5 py-3 text-[13px] [&>svg]:mt-0.5 [&>svg]:size-4 [&>svg]:shrink-0',
  {
    variants: {
      tone: {
        info: 'border-[oklch(0.645_0.18_251/0.25)] bg-info-soft text-foreground [&>svg]:text-info',
        success:
          'border-[oklch(0.62_0.145_155/0.25)] bg-success-soft text-foreground [&>svg]:text-success',
        warning:
          'border-[oklch(0.74_0.16_62/0.3)] bg-warning-soft text-foreground [&>svg]:text-warning',
        danger:
          'border-[oklch(0.59_0.215_25/0.25)] bg-destructive-soft text-foreground [&>svg]:text-destructive',
        neutral: 'border-border bg-surface-subtle text-foreground [&>svg]:text-muted-foreground',
      },
    },
    defaultVariants: { tone: 'info' },
  },
)

const alertIcons = {
  info: Info,
  success: CheckCircle2,
  warning: AlertTriangle,
  danger: OctagonAlert,
  neutral: Info,
}

export function Alert({
  tone = 'info',
  title,
  children,
  actions,
  icon,
  className,
}: VariantProps<typeof alertVariants> & {
  title?: React.ReactNode
  children?: React.ReactNode
  actions?: React.ReactNode
  icon?: React.ReactNode
  className?: string
}) {
  const Icon = alertIcons[tone ?? 'info']
  return (
    <div role="status" className={cn(alertVariants({ tone }), className)}>
      {icon ?? <Icon />}
      <div className="flex min-w-0 flex-1 flex-col gap-0.5">
        {title ? <p className="font-medium leading-5">{title}</p> : null}
        {children ? <div className="leading-5 text-muted-foreground">{children}</div> : null}
      </div>
      {actions ? <div className="flex shrink-0 items-start gap-2">{actions}</div> : null}
    </div>
  )
}
