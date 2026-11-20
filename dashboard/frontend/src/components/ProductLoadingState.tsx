import { Loader2 } from 'lucide-react'

import { LogoMark, cn } from '../ui'

interface ProductLoadingStateProps {
  label?: string
  compact?: boolean
}

/**
 * Loading indicator for a region that has nothing to show yet. The compact
 * form sits in the flow of its container; the default form centers itself in
 * the remaining viewport height for page-level loading.
 */
export default function ProductLoadingState({
  label = 'Loading',
  compact = false,
}: ProductLoadingStateProps) {
  return (
    <div
      className={cn(
        'flex w-full items-center justify-center animate-in fade-in-0 duration-300',
        compact
          ? 'min-h-[7.5rem] gap-2.5 text-[13px] text-muted-foreground'
          : 'min-h-[calc(100dvh-8rem)] flex-col gap-4 px-6 text-center',
      )}
      role="status"
      aria-live="polite"
    >
      {compact ? (
        <Loader2
          className="size-4 shrink-0 animate-spin text-subtle-foreground"
          aria-hidden="true"
        />
      ) : (
        <span className="relative grid size-12 place-items-center rounded-2xl border border-border bg-surface shadow-sm">
          <span
            aria-hidden="true"
            className="absolute inset-0 rounded-2xl text-primary/50 animate-pulse-ring"
          />
          <LogoMark size={24} />
        </span>
      )}
      <span className={compact ? undefined : 'text-[13px] text-muted-foreground'}>{label}</span>
    </div>
  )
}
