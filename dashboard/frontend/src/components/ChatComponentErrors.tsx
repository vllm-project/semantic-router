import { OctagonAlert, RefreshCw, X } from 'lucide-react'

import { Button, cn } from '../ui'
import type { PlaygroundErrorPresentation } from './playgroundErrorPresentation'
import type { PlaygroundRoutingModelStatus } from './usePlaygroundRoutingModel'

interface ChatComponentErrorsProps {
  overlay: boolean
  onDismissError: () => void
  onRetryRoutingModelDiscovery: () => void
  routingModelStatus: PlaygroundRoutingModelStatus
  visibleError: PlaygroundErrorPresentation | null
}

export default function ChatComponentErrors({
  overlay,
  onDismissError,
  onRetryRoutingModelDiscovery,
  routingModelStatus,
  visibleError,
}: ChatComponentErrorsProps) {
  const routingModelUnavailable = routingModelStatus === 'error' && !visibleError
  if (!routingModelUnavailable && !visibleError) return null

  return (
    <div
      className={cn(
        'z-10 w-full shrink-0 px-4 pt-3',
        overlay ? 'absolute inset-x-0 top-0' : 'relative',
      )}
      data-testid="playground-error-region"
    >
      <div
        className="mx-auto flex w-full max-w-3xl items-start gap-3 rounded-lg border border-destructive/25 bg-destructive-soft px-3.5 py-3 text-[13px] text-foreground shadow-xs"
        role="alert"
      >
        <OctagonAlert className="mt-0.5 size-4 shrink-0 text-destructive" aria-hidden="true" />
        <div className="flex min-w-0 flex-1 flex-col gap-1">
          <span className="font-medium leading-5">
            {routingModelUnavailable
              ? 'The automatic routing model is unavailable.'
              : visibleError?.message}
          </span>
          {routingModelUnavailable ? (
            <span className="text-xs leading-5 text-muted-foreground">
              Messages can be sent once the router advertises a routing model.
            </span>
          ) : null}
          {!routingModelUnavailable && visibleError?.technicalDetails ? (
            <details
              className="text-xs text-muted-foreground [&>summary]:w-fit [&>summary]:cursor-pointer [&>summary]:select-none [&>summary]:rounded-sm [&>summary]:outline-none [&>summary]:hover:text-foreground [&>summary]:focus-visible:ring-[3px] [&>summary]:focus-visible:ring-ring/40"
              data-playground-technical-details="true"
            >
              <summary>Technical details</summary>
              <pre className="mt-1.5 max-h-40 overflow-auto whitespace-pre-wrap break-words rounded-md border border-border bg-surface px-2.5 py-2 font-mono text-[11.5px] leading-relaxed text-foreground/85">
                {visibleError.technicalDetails}
              </pre>
            </details>
          ) : null}
        </div>
        {routingModelUnavailable ? (
          <Button
            type="button"
            size="sm"
            variant="outline"
            className="shrink-0"
            onClick={onRetryRoutingModelDiscovery}
          >
            <RefreshCw />
            Retry discovery
          </Button>
        ) : (
          <Button
            type="button"
            variant="ghost"
            size="icon-xs"
            className="shrink-0"
            aria-label="Dismiss error"
            onClick={onDismissError}
          >
            <X />
          </Button>
        )}
      </div>
    </div>
  )
}
