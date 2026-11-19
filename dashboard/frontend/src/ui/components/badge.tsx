import * as React from 'react'
import { cva, type VariantProps } from 'class-variance-authority'
import { cn } from '../lib/cn'

export const badgeVariants = cva(
  "inline-flex w-fit shrink-0 items-center gap-1 whitespace-nowrap rounded-full border font-medium leading-none transition-colors [&_svg]:pointer-events-none [&_svg:not([class*='size-'])]:size-3",
  {
    variants: {
      tone: {
        neutral: 'border-border bg-secondary text-secondary-foreground',
        outline: 'border-border bg-transparent text-muted-foreground',
        brand: 'border-transparent bg-primary-soft text-primary-soft-foreground',
        gold: 'border-transparent bg-[oklch(0.84_0.16_80/0.16)] text-[oklch(0.5_0.12_66)] dark:text-[oklch(0.88_0.13_82)]',
        success: 'border-transparent bg-success-soft text-success-foreground',
        warning: 'border-transparent bg-warning-soft text-warning-foreground',
        danger: 'border-transparent bg-destructive-soft text-destructive-foreground',
        info: 'border-transparent bg-info-soft text-info-foreground',
        violet:
          'border-transparent bg-[oklch(0.6_0.2_295/0.12)] text-[oklch(0.5_0.19_295)] dark:text-[oklch(0.8_0.12_295)]',
        teal: 'border-transparent bg-[oklch(0.68_0.12_190/0.14)] text-[oklch(0.45_0.09_195)] dark:text-[oklch(0.82_0.09_190)]',
        rose: 'border-transparent bg-[oklch(0.64_0.2_12/0.12)] text-[oklch(0.52_0.19_12)] dark:text-[oklch(0.8_0.12_12)]',
      },
      size: {
        sm: 'h-5 px-1.5 text-[11px]',
        default: 'h-[22px] px-2 text-xs',
        lg: 'h-6 px-2.5 text-xs',
      },
    },
    defaultVariants: { tone: 'neutral', size: 'default' },
  },
)

const dotTone: Record<string, string> = {
  neutral: 'bg-subtle-foreground',
  outline: 'bg-subtle-foreground',
  brand: 'bg-primary',
  gold: 'bg-brand-gold',
  success: 'bg-success',
  warning: 'bg-warning',
  danger: 'bg-destructive',
  info: 'bg-info',
  violet: 'bg-chart-3',
  teal: 'bg-chart-4',
  rose: 'bg-chart-5',
}

export interface BadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {
  dot?: boolean
}

export const Badge = React.forwardRef<HTMLSpanElement, BadgeProps>(
  ({ className, tone, size, dot, children, ...props }, ref) => (
    <span
      ref={ref}
      data-slot="badge"
      className={cn(badgeVariants({ tone, size }), className)}
      {...props}
    >
      {dot ? <span className={cn('size-1.5 rounded-full', dotTone[tone ?? 'neutral'])} /> : null}
      {children}
    </span>
  ),
)
Badge.displayName = 'Badge'
