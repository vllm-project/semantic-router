import * as React from 'react'
import { Dialog as DialogPrimitive } from 'radix-ui'
import { X } from 'lucide-react'
import { cn } from '../lib/cn'
import { SheetContent } from '../components/overlays'

/** Right-side detail drawer with a consistent header, scrollable body and sticky footer. */
export function DetailSheet({
  open,
  onOpenChange,
  size = 'lg',
  children,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  size?: 'sm' | 'md' | 'lg' | 'xl'
  children: React.ReactNode
}) {
  return (
    <DialogPrimitive.Root open={open} onOpenChange={onOpenChange}>
      <SheetContent
        size={size}
        onOpenAutoFocus={(event) => {
          event.preventDefault()
          ;(event.currentTarget as HTMLElement | null)?.focus()
        }}
        tabIndex={-1}
      >
        {children}
      </SheetContent>
    </DialogPrimitive.Root>
  )
}

export function DetailSheetHeader({
  icon,
  title,
  description,
  badges,
  actions,
  className,
}: {
  icon?: React.ReactNode
  title: React.ReactNode
  description?: React.ReactNode
  badges?: React.ReactNode
  actions?: React.ReactNode
  className?: string
}) {
  return (
    <div className={cn('flex items-start gap-3 border-b border-border px-5 py-4', className)}>
      {icon}
      <div className="flex min-w-0 flex-1 flex-col gap-1">
        <div className="flex flex-wrap items-center gap-2">
          <DialogPrimitive.Title className="truncate text-base font-semibold tracking-[-0.01em]">
            {title}
          </DialogPrimitive.Title>
          {badges}
        </div>
        {description ? (
          <DialogPrimitive.Description className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-xs text-muted-foreground">
            {description}
          </DialogPrimitive.Description>
        ) : (
          <DialogPrimitive.Description className="sr-only">Details</DialogPrimitive.Description>
        )}
      </div>
      <div className="flex shrink-0 items-center gap-1.5">
        {actions}
        <DialogPrimitive.Close className="grid size-8 place-items-center rounded-md text-subtle-foreground transition-colors hover:bg-accent hover:text-foreground focus-visible:ring-[3px] focus-visible:ring-ring/40 focus-visible:outline-none">
          <X className="size-4" />
          <span className="sr-only">Close</span>
        </DialogPrimitive.Close>
      </div>
    </div>
  )
}

export function DetailSheetBody({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn('min-h-0 flex-1 overflow-y-auto', className)} {...props} />
}

export function DetailSheetFooter({
  highlight = false,
  className,
  children,
}: {
  highlight?: boolean
  className?: string
  children: React.ReactNode
}) {
  return (
    <div
      className={cn(
        'flex items-center gap-2 border-t border-border bg-surface-subtle/70 px-5 py-3 transition-colors duration-300',
        highlight && 'bg-primary-soft/50',
        className,
      )}
    >
      {children}
    </div>
  )
}

export function SectionLabel({ className, ...props }: React.HTMLAttributes<HTMLHeadingElement>) {
  return (
    <h3
      className={cn(
        'text-[11px] font-medium uppercase tracking-[0.06em] text-subtle-foreground',
        className,
      )}
      {...props}
    />
  )
}

export function MiniStat({
  label,
  value,
  hint,
  className,
}: {
  label: React.ReactNode
  value: React.ReactNode
  hint?: React.ReactNode
  className?: string
}) {
  return (
    <div
      className={cn('rounded-lg border border-border bg-surface-subtle/60 px-3 py-2.5', className)}
    >
      <div className="text-xs text-subtle-foreground">{label}</div>
      <div className="mt-1 text-lg font-semibold tabular tracking-[-0.02em]">{value}</div>
      {hint ? <div className="mt-0.5 text-xs text-muted-foreground">{hint}</div> : null}
    </div>
  )
}

/** Keeps the last non-empty value so a closing drawer can animate out with its content. */
export function useRetainedValue<T>(value: T | null | undefined): T | null {
  const ref = React.useRef<T | null>(value ?? null)
  if (value !== null && value !== undefined) ref.current = value
  return ref.current
}
