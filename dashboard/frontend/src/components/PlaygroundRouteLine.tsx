import { Boxes, Clock, Coins, ListTree, Split } from 'lucide-react'

import { ProviderLogo, cn } from '../ui'
import { modelProviderId, useInsightsConfig } from '../pages/insightsConfigFacts'
import { modelLabel, modelText, useModelLabels } from '../pages/insightsModelLabelSupport'
import {
  formatInspectedCost,
  formatInspectedMs,
  readInspectedRoute,
} from './playgroundInspectorSupport'

/**
 * One line under an answer: the decision that matched, the model that
 * answered, its latency and cost. It opens the answer in the routing trace.
 */
export default function PlaygroundRouteLine({
  headers,
  inspected,
  onInspect,
}: {
  headers: Record<string, string>
  inspected: boolean
  onInspect?: () => void
}) {
  const { modelProviders } = useInsightsConfig()
  const labels = useModelLabels()
  const route = readInspectedRoute(headers)
  if (!route.decision && !route.model) return null
  const provider = route.model ? modelProviderId(route.model, modelProviders) : undefined
  const label = modelLabel(labels, route.model)
  const latency = formatInspectedMs(route.latencyMs)
  const cost = formatInspectedCost(route.cost)

  const content = (
    <>
      {onInspect ? <span className="sr-only">Inspect route: </span> : null}
      {route.decision ? (
        <>
          <span className="inline-flex min-w-0 items-center gap-1.5 font-medium text-chart-3">
            <Split className="size-3.5 shrink-0" aria-hidden />
            <span className="truncate">{route.decision.label}</span>
          </span>
          {route.model ? (
            <span className="text-subtle-foreground" aria-hidden>
              →
            </span>
          ) : null}
        </>
      ) : null}
      {route.model ? (
        <span
          className="inline-flex min-w-0 items-center gap-1.5 font-medium text-foreground"
          title={modelText(labels, route.model)}
        >
          {provider ? (
            <ProviderLogo provider={provider} size={14} />
          ) : (
            <Boxes className="size-3.5 shrink-0 text-chart-6" aria-hidden />
          )}
          <span className={cn('truncate', label === route.model && 'font-mono text-[11.5px]')}>
            {label}
          </span>
        </span>
      ) : null}
      {latency || cost ? <span className="h-3 w-px shrink-0 bg-border-strong" aria-hidden /> : null}
      {latency ? (
        <span className="inline-flex shrink-0 items-center gap-1 tabular text-muted-foreground">
          <Clock className="size-3" aria-hidden />
          {latency}
        </span>
      ) : null}
      {cost ? (
        <span className="inline-flex shrink-0 items-center gap-1 tabular text-muted-foreground">
          <Coins className="size-3" aria-hidden />
          {cost}
        </span>
      ) : null}
      {onInspect ? (
        <ListTree
          className="size-3.5 shrink-0 text-subtle-foreground transition-colors group-hover/route:text-foreground"
          aria-hidden
        />
      ) : null}
    </>
  )

  const className = cn(
    'inline-flex min-w-0 max-w-full flex-wrap items-center gap-x-2.5 gap-y-1 rounded-lg border px-2.5 py-1.5 text-left text-xs',
    inspected ? 'border-chart-3/40 bg-chart-3/[0.07]' : 'border-border bg-surface',
  )
  if (!onInspect) return <div className={className}>{content}</div>
  return (
    <button
      type="button"
      onClick={onInspect}
      aria-pressed={inspected}
      className={cn(
        className,
        'group/route outline-none transition-colors focus-visible:ring-[3px] focus-visible:ring-ring/40',
        !inspected && 'hover:border-border-strong',
      )}
    >
      {content}
    </button>
  )
}
