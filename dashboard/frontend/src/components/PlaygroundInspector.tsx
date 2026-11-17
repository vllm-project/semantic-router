import { useEffect } from 'react'
import { Link } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import { ArrowUpRight, Boxes, ListTree, PanelRightClose, Split, TriangleAlert } from 'lucide-react'

import {
  Badge,
  Button,
  CopyButton,
  EmptyState,
  IconTile,
  ProviderLogo,
  SectionLabel,
  Skeleton,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
  cn,
} from '../ui'
import { signalTypeIcon } from '../pages/configPageSignalCatalog'
import { modelProviderId, useInsightsConfig } from '../pages/insightsConfigFacts'
import type { Message } from './ChatComponentTypes'
import { buildFeedbackInsightsHref } from './chatComponentSupport'
import {
  readInspectedRoute,
  sortedHeaderEntries,
  type InspectedRoute,
} from './playgroundInspectorSupport'

interface PlaygroundInspectorProps {
  open: boolean
  /** Float over the chat, as on a narrow viewport, instead of docking beside it. */
  overlay: boolean
  message: Message | null
  /** The user message the inspected answer replies to. */
  prompt?: string
  insightsBasePath?: string
  onClose: () => void
}

const EASE = [0.16, 1, 0.3, 1] as const

function ModelName({ name }: { name: string }) {
  const { modelProviders } = useInsightsConfig()
  const provider = modelProviderId(name, modelProviders)
  return (
    <span className="flex min-w-0 items-center gap-2">
      {provider ? (
        <ProviderLogo provider={provider} size={16} />
      ) : (
        <Boxes className="size-4 shrink-0 text-chart-6" />
      )}
      <span className="truncate font-mono text-[12.5px] font-medium" title={name}>
        {name}
      </span>
    </span>
  )
}

function RouteView({
  route,
  prompt,
  recordHref,
}: {
  route: InspectedRoute
  prompt?: string
  recordHref?: string
}) {
  return (
    <div className="flex flex-col gap-5">
      {prompt ? (
        <p className="line-clamp-2 border-l-2 border-border-strong pl-2.5 text-xs leading-relaxed text-muted-foreground">
          {prompt}
        </p>
      ) : null}

      <section className="flex flex-col gap-2">
        <SectionLabel>Decision</SectionLabel>
        {route.decision ? (
          <div className="rounded-lg border border-chart-3/30 bg-chart-3/[0.06] p-3">
            <div className="flex items-start gap-2.5">
              <IconTile icon={Split} tone="violet" size="sm" />
              <div className="flex min-w-0 flex-col">
                <span className="truncate text-[13px] font-semibold">{route.decision.label}</span>
                <span className="truncate font-mono text-[11px] text-subtle-foreground">
                  {route.decision.id}
                </span>
              </div>
            </div>
            {route.recipe || route.algorithm || route.confidence ? (
              <div className="mt-2.5 flex flex-wrap gap-1.5">
                {route.recipe ? (
                  <Badge tone="outline" size="sm">
                    recipe {route.recipe}
                  </Badge>
                ) : null}
                {route.algorithm ? (
                  <Badge tone="violet" size="sm">
                    {route.algorithm}
                  </Badge>
                ) : null}
                {route.confidence ? (
                  <Badge tone="outline" size="sm" className="tabular">
                    confidence {route.confidence}
                  </Badge>
                ) : null}
              </div>
            ) : null}
          </div>
        ) : (
          <p className="text-[13px] text-muted-foreground">The router reported no decision.</p>
        )}
      </section>

      <section className="flex flex-col gap-2">
        <SectionLabel>Model</SectionLabel>
        {route.model ? (
          <div className="flex flex-col gap-2 rounded-lg border border-primary/30 bg-primary-soft/50 px-3 py-2.5">
            <ModelName name={route.model} />
            {route.looper ? (
              <div className="flex flex-col gap-1 border-t border-border pt-2">
                <span className="text-[11px] text-subtle-foreground">
                  Looper
                  {route.looper.iterations ? ` · ${route.looper.iterations} iterations` : ''}
                </span>
                {route.looper.models.map((model) => (
                  <ModelName key={model} name={model} />
                ))}
              </div>
            ) : null}
          </div>
        ) : (
          <p className="text-[13px] text-muted-foreground">The router reported no model.</p>
        )}
      </section>

      <section className="flex flex-col gap-2">
        <SectionLabel>Signals</SectionLabel>
        {route.signals.length ? (
          route.signals.map((signal, index) => {
            const Icon = signalTypeIcon(signal.type)
            return (
              <motion.div
                key={signal.type}
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.04 * index, duration: 0.24, ease: EASE }}
                className="flex flex-col gap-1.5 rounded-lg border border-chart-2/35 bg-chart-2/[0.07] px-2.5 py-2"
              >
                <span className="flex items-center gap-2 text-xs font-medium">
                  <Icon className="size-3.5 text-chart-2" />
                  {signal.label}
                </span>
                <span className="flex flex-wrap gap-1">
                  {signal.names.map((name) => (
                    <span
                      key={name}
                      className="rounded-[5px] border border-border bg-surface px-1.5 py-0.5 font-mono text-[11px] text-foreground/85"
                    >
                      {name}
                    </span>
                  ))}
                </span>
              </motion.div>
            )
          })
        ) : (
          <p className="text-xs leading-relaxed text-muted-foreground">
            No matched signal was reported. The router lists them when it allows this dashboard its
            debug trace.
          </p>
        )}
      </section>

      {route.metrics.length ? (
        <section className="grid grid-cols-2 gap-2">
          {route.metrics.map((metric) => (
            <div key={metric.id} className="rounded-lg border border-border bg-surface px-2.5 py-2">
              <div className="text-[11px] text-subtle-foreground">{metric.label}</div>
              <div
                className="truncate font-mono text-[13px] font-medium tabular"
                title={metric.value}
              >
                {metric.value}
              </div>
            </div>
          ))}
        </section>
      ) : null}

      {route.details.length || route.warnings.length || route.truncated || route.responsePath ? (
        <section className="flex flex-col gap-2">
          <SectionLabel>Details</SectionLabel>
          <dl className="flex flex-col gap-1.5 text-xs">
            {route.responsePath ? (
              <div className="flex items-center justify-between gap-3">
                <dt className="text-subtle-foreground">Response path</dt>
                <dd className="font-mono">{route.responsePath}</dd>
              </div>
            ) : null}
            {route.details.map((detail) => (
              <div key={detail.id} className="flex items-center justify-between gap-3">
                <dt className="text-subtle-foreground">{detail.label}</dt>
                <dd className="truncate font-mono">{detail.value}</dd>
              </div>
            ))}
          </dl>
          {route.warnings.length || route.truncated ? (
            <div className="flex flex-wrap gap-1.5">
              {route.warnings.map((warning) => (
                <Badge key={warning} tone="warning" size="sm">
                  <TriangleAlert />
                  {warning}
                </Badge>
              ))}
              {route.truncated ? (
                <Badge tone="warning" size="sm">
                  Trace truncated
                </Badge>
              ) : null}
            </div>
          ) : null}
        </section>
      ) : null}

      {recordHref ? (
        <Button variant="outline" size="sm" asChild>
          <Link to={recordHref}>
            Open in Requests
            <ArrowUpRight />
          </Link>
        </Button>
      ) : null}
    </div>
  )
}

function HeadersView({ headers }: { headers: Record<string, string> }) {
  return (
    <div className="flex flex-col gap-1.5">
      {sortedHeaderEntries(headers).map(([key, value]) => (
        <div
          key={key}
          className="flex items-center gap-2 rounded-lg border border-border bg-surface px-2.5 py-1.5"
        >
          <div className="flex min-w-0 flex-1 flex-col">
            <span className="truncate font-mono text-[11px] text-subtle-foreground">{key}</span>
            <span className="truncate font-mono text-xs" title={value}>
              {value}
            </span>
          </div>
          <CopyButton value={value} label={`Copy ${key}`} className="size-6" />
        </div>
      ))}
    </div>
  )
}

/** The playground's routing trace: how the router routed the selected answer. */
export default function PlaygroundInspector({
  open,
  overlay,
  message,
  prompt,
  insightsBasePath,
  onClose,
}: PlaygroundInspectorProps) {
  useEffect(() => {
    if (!open || !overlay) return undefined
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', closeOnEscape)
    return () => document.removeEventListener('keydown', closeOnEscape)
  }, [onClose, open, overlay])

  const headers = message?.headers
  const route = headers ? readInspectedRoute(headers) : null
  const recordHref = route?.replayId
    ? buildFeedbackInsightsHref(insightsBasePath, route.replayId)
    : undefined

  let body
  if (!message) {
    body = (
      <EmptyState
        compact
        icon={ListTree}
        tone="violet"
        title="Nothing to inspect yet"
        description="Send a message. Its signals, decision, model, latency and tokens show up here."
      />
    )
  } else if (!route) {
    body = message.isStreaming ? (
      <div className="flex flex-col gap-3" aria-busy="true">
        <p className="text-xs text-muted-foreground">
          The trace appears as soon as the answer finishes.
        </p>
        <Skeleton className="h-16 w-full rounded-lg" />
        <Skeleton className="h-11 w-full rounded-lg" />
        <Skeleton className="h-24 w-full rounded-lg" />
      </div>
    ) : (
      <EmptyState
        compact
        icon={ListTree}
        tone="slate"
        title="No routing headers on this answer"
        description="The answer failed, or a proxy in front of the router dropped its x-vsr headers."
      />
    )
  } else {
    body = null
  }

  return (
    <AnimatePresence initial={false}>
      {open && overlay ? (
        <motion.div
          key="scrim"
          aria-hidden
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="absolute inset-0 z-20 bg-background/60 backdrop-blur-[1px]"
          onClick={onClose}
        />
      ) : null}
      {open ? (
        <motion.aside
          key="inspector"
          aria-label="Routing trace"
          data-testid="playground-inspector"
          initial={{ opacity: 0, x: 16 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: 16 }}
          transition={{ duration: 0.28, ease: EASE }}
          className={cn(
            'flex w-[340px] shrink-0 flex-col border-l border-border',
            overlay
              ? 'absolute inset-y-0 right-0 z-30 max-w-[calc(100%-2.5rem)] bg-surface shadow-xl'
              : 'bg-surface-subtle/50',
          )}
        >
          <div className="flex h-12 shrink-0 items-center gap-2 border-b border-border px-4">
            <ListTree className="size-4 text-subtle-foreground" />
            <span className="text-[13px] font-semibold">Routing trace</span>
            <Button
              variant="ghost"
              size="icon-xs"
              className="ml-auto"
              onClick={onClose}
              aria-label="Close routing trace"
            >
              <PanelRightClose />
            </Button>
          </div>
          {route && headers ? (
            <Tabs defaultValue="route" className="min-h-0 flex-1 gap-0">
              <TabsList className="px-4">
                <TabsTrigger value="route">Route</TabsTrigger>
                <TabsTrigger value="headers">Headers</TabsTrigger>
              </TabsList>
              <div className="min-h-0 flex-1 overflow-y-auto">
                <TabsContent value="route" className="p-4">
                  <RouteView route={route} prompt={prompt} recordHref={recordHref} />
                </TabsContent>
                <TabsContent value="headers" className="p-4">
                  <HeadersView headers={headers} />
                </TabsContent>
              </div>
            </Tabs>
          ) : (
            <div className="min-h-0 flex-1 overflow-y-auto p-4">{body}</div>
          )}
        </motion.aside>
      ) : null}
    </AnimatePresence>
  )
}
