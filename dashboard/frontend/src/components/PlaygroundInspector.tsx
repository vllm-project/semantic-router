import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import {
  ArrowUpRight,
  Boxes,
  Check,
  ListTree,
  PanelRightClose,
  Split,
  TriangleAlert,
} from 'lucide-react'

import {
  Badge,
  Button,
  CopyButton,
  EmptyState,
  IconTile,
  MiniBar,
  ProviderLogo,
  SectionLabel,
  Skeleton,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
  cn,
} from '../ui'
import { X_VSR_REPLAY_ID } from '../generated/vsrHeaders'
import { signalTypeIcon } from '../pages/configPageSignalCatalog'
import { modelProviderId, useInsightsConfig } from '../pages/insightsConfigFacts'
import { modelLabel, useModelLabels } from '../pages/insightsModelLabelSupport'
import { fetchInsightsRecord, insightsUnavailableReason } from '../pages/insightsPageApi'
import type { InsightsRecord } from '../pages/insightsPageTypes'
import type { Message } from './ChatComponentTypes'
import { buildFeedbackInsightsHref } from './chatComponentSupport'
import HeaderDisplay from './HeaderDisplay'
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
// The Router writes the record as the answer ends, so a fresh one can lag a moment.
const RECORD_RETRY_MS = [0, 1_500, 4_000]
const records = new Map<string, InsightsRecord>()

/** The answer's Router Replay record, when the viewer may read it; null until then. */
function useReplayRecord(replayId: string | undefined, enabled: boolean) {
  const [record, setRecord] = useState<InsightsRecord | null>(
    replayId ? (records.get(replayId) ?? null) : null,
  )
  useEffect(() => {
    if (!replayId || !enabled) {
      setRecord(null)
      return undefined
    }
    const cached = records.get(replayId)
    setRecord(cached ?? null)
    if (cached) return undefined
    let cancelled = false
    let timer: number | undefined
    const attempt = (index: number) => {
      timer = window.setTimeout(() => {
        fetchInsightsRecord(replayId)
          .then((loaded) => {
            records.set(replayId, loaded)
            if (!cancelled) setRecord(loaded)
          })
          .catch((error: unknown) => {
            const reason = insightsUnavailableReason(error)
            const retry = reason !== 'forbidden' && reason !== 'router_unreachable'
            if (!cancelled && retry && index + 1 < RECORD_RETRY_MS.length) attempt(index + 1)
          })
      }, RECORD_RETRY_MS[index])
    }
    attempt(0)
    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [enabled, replayId])
  return record
}

function ModelName({ name, className }: { name: string; className?: string }) {
  const { modelProviders } = useInsightsConfig()
  const label = modelLabel(useModelLabels(), name)
  const provider = modelProviderId(name, modelProviders)
  return (
    <span className={cn('flex min-w-0 items-center gap-2', className)}>
      {provider ? (
        <ProviderLogo provider={provider} size={16} />
      ) : (
        <Boxes className="size-4 shrink-0 text-chart-6" />
      )}
      {label === name ? (
        <span className="truncate font-mono text-[12.5px] font-medium" title={name}>
          {name}
        </span>
      ) : (
        <span className="truncate text-xs font-medium" title={`Configured as ${name}`}>
          {label}
        </span>
      )}
    </span>
  )
}

function SignalsSection({ route }: { route: InspectedRoute }) {
  return (
    <section className="flex flex-col gap-2">
      <SectionLabel>Signals</SectionLabel>
      {route.signals.length ? (
        route.signals.map((signal, index) => {
          const Icon = signalTypeIcon(signal.type)
          return (
            <motion.div
              key={signal.key}
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.04 * index, duration: 0.24, ease: EASE }}
              className={cn(
                'flex flex-col gap-1.5 rounded-lg border px-2.5 py-2',
                signal.matched
                  ? 'border-chart-2/35 bg-chart-2/[0.07]'
                  : 'border-border bg-surface opacity-70',
              )}
              data-signal={signal.key}
              data-matched={signal.matched}
            >
              <span className="flex min-w-0 items-center gap-2 text-xs">
                <Icon
                  className={cn(
                    'size-3.5 shrink-0',
                    signal.matched ? 'text-chart-2' : 'text-subtle-foreground',
                  )}
                  aria-hidden
                />
                <span className="truncate font-mono" title={`${signal.typeLabel}: ${signal.name}`}>
                  {signal.name}
                </span>
                <span className="shrink-0 text-[11px] text-subtle-foreground">
                  {signal.typeLabel}
                </span>
                {signal.confidence !== undefined ? (
                  <span className="ml-auto shrink-0 font-mono tabular text-muted-foreground">
                    {signal.confidence.toFixed(2)}
                  </span>
                ) : null}
              </span>
              {signal.confidence !== undefined ? (
                <MiniBar
                  value={signal.confidence * 100}
                  tone={signal.matched ? 'gold' : 'slate'}
                  className="h-1"
                />
              ) : null}
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
  )
}

function DecisionSection({ route }: { route: InspectedRoute }) {
  return (
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
          {route.runnerUp ? (
            <p className="mt-2 text-xs text-muted-foreground">
              Beat <span className="font-mono text-foreground/85">{route.runnerUp}</span>
            </p>
          ) : null}
          {route.recipe || route.confidence ? (
            <div className="mt-2.5 flex flex-wrap gap-1.5">
              {route.recipe ? (
                <Badge tone="outline" size="sm">
                  recipe {route.recipe}
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
  )
}

function ModelSection({ route }: { route: InspectedRoute }) {
  return (
    <section className="flex flex-col gap-2">
      <SectionLabel>
        {route.algorithm ? `Model selection · ${route.algorithm}` : 'Model selection'}
      </SectionLabel>
      {route.candidates.length ? (
        route.candidates.map((candidate) => (
          <div
            key={candidate.model}
            className={cn(
              'flex items-center gap-2.5 rounded-lg border px-2.5 py-2',
              candidate.chosen
                ? 'border-primary/40 bg-primary-soft/60'
                : 'border-border bg-surface',
            )}
            data-candidate={candidate.model}
          >
            <ModelName name={candidate.model} className="flex-1" />
            {candidate.score !== undefined ? (
              <span className="font-mono text-xs tabular text-muted-foreground">
                {candidate.score.toFixed(2)}
              </span>
            ) : null}
            {candidate.chosen ? (
              <Check
                className="size-3.5 shrink-0 text-primary"
                strokeWidth={2.6}
                aria-label="Chosen"
              />
            ) : null}
          </div>
        ))
      ) : route.model ? (
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
  )
}

function RouteView({
  route,
  headers,
  prompt,
  recordHref,
}: {
  route: InspectedRoute
  headers: Record<string, string>
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

      <SignalsSection route={route} />
      <DecisionSection route={route} />
      <ModelSection route={route} />

      {route.plugins.length ? (
        <section className="flex flex-col gap-2">
          <SectionLabel>Plugins</SectionLabel>
          <div className="flex flex-wrap gap-1.5">
            {route.plugins.map((plugin) => (
              <Badge key={plugin.id} tone={plugin.tone}>
                {plugin.label}
                {plugin.detail ? <span className="opacity-75">· {plugin.detail}</span> : null}
              </Badge>
            ))}
          </div>
        </section>
      ) : null}

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

      <HeaderDisplay headers={headers} />

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
  const replayId = headers?.[X_VSR_REPLAY_ID]?.trim() || undefined
  const record = useReplayRecord(replayId, open && Boolean(message) && !message?.isStreaming)
  const route = headers ? readInspectedRoute(headers, record) : null
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
                  <RouteView
                    route={route}
                    headers={headers}
                    prompt={prompt}
                    recordHref={recordHref}
                  />
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
