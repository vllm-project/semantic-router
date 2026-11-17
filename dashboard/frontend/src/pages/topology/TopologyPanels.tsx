import * as React from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import type { Node } from 'reactflow'
import {
  ArrowRight,
  ArrowUpRight,
  ChevronDown,
  CircleAlert,
  PanelLeftClose,
  TriangleAlert,
  X,
} from 'lucide-react'
import {
  Badge,
  Button,
  Label,
  SectionLabel,
  Segmented,
  Switch,
  Tooltip,
  cn,
  toneFill,
  type Tone,
} from '../../ui'
import RoutingScopeSelector from '../../components/RoutingScopeSelector'
import type { RoutingScope } from '../../utils/routingScopes'
import { signalTypeIcon, signalTypeLabel } from '../configPageSignalCatalog'
import { algorithmMeta, pluginLabel } from '../configPageDecisionPresentation'
import type { TestQueryResult } from './types'
import {
  DEFAULT_ID,
  ENTRY_ID,
  MORE_ID,
  decisionKind,
  ruleConditions,
  ruleSummary,
  type Density,
  type Layer,
  type TopoNodeData,
  type TopologyModel,
} from './topologyGraph'

const ease = [0.16, 1, 0.3, 1] as const
export const PANEL_WIDTH = 240
const INSPECTOR_WIDTH = 300

const LAYERS: { layer: Layer; label: string; tone: Tone }[] = [
  { layer: 'signals', label: 'Signals', tone: 'gold' },
  { layer: 'projections', label: 'Projections', tone: 'teal' },
  { layer: 'decisions', label: 'Decisions', tone: 'violet' },
  { layer: 'models', label: 'Models', tone: 'green' },
]

const LEGEND: { label: string; tone: Tone }[] = [
  { label: 'Public model', tone: 'blue' },
  { label: 'Signal family', tone: 'gold' },
  { label: 'Projection', tone: 'teal' },
  { label: 'Decision', tone: 'violet' },
  { label: 'Guardrail', tone: 'rose' },
  { label: 'Model', tone: 'green' },
]

export function ViewPanel({
  scopes,
  scopeId,
  onScope,
  layers,
  onLayer,
  counts,
  density,
  onDensity,
  showAll,
  hiddenDecisions,
  onShowAll,
  onCollapse,
}: {
  scopes: RoutingScope[]
  scopeId: string
  onScope: (value: string) => void
  layers: Record<Layer, boolean>
  onLayer: (layer: Layer, on: boolean) => void
  counts: Record<Layer, number>
  density: Density
  onDensity: (value: Density) => void
  showAll: boolean
  hiddenDecisions: number
  onShowAll: (value: boolean) => void
  onCollapse: () => void
}) {
  return (
    <motion.aside
      initial={{ opacity: 0, x: -8 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -8 }}
      transition={{ duration: 0.3, ease }}
      style={{ width: PANEL_WIDTH }}
      aria-label="Topology view"
      className="absolute left-3 top-3 z-10 flex max-h-[calc(100%-1.5rem)] flex-col overflow-y-auto rounded-xl border border-border bg-popover/95 shadow-lg backdrop-blur-md"
    >
      <div className="flex items-center justify-between border-b border-border py-2 pl-3 pr-1.5">
        <span className="text-[13px] font-semibold">View</span>
        <Button variant="ghost" size="icon-xs" aria-label="Hide view panel" onClick={onCollapse}>
          <PanelLeftClose />
        </Button>
      </div>
      <div className="flex flex-col gap-4 p-3">
        {scopes.length ? (
          <section className="flex flex-col gap-1.5 [&>div]:w-full [&>div]:min-w-0">
            <SectionLabel>Recipe</SectionLabel>
            <RoutingScopeSelector
              label="Recipe"
              scopes={scopes}
              value={scopeId}
              onChange={onScope}
            />
          </section>
        ) : null}
        <section className="flex flex-col gap-1">
          <SectionLabel className="mb-1">Layers</SectionLabel>
          {LAYERS.map((item) => (
            <div key={item.layer} className="flex h-7 items-center gap-2">
              <span className="size-2 rounded-full" style={{ background: toneFill[item.tone] }} />
              <Label htmlFor={`layer-${item.layer}`} className="flex-1 cursor-pointer font-normal">
                {item.label}
              </Label>
              <span className="font-mono text-[11px] tabular text-subtle-foreground">
                {counts[item.layer]}
              </span>
              <Switch
                id={`layer-${item.layer}`}
                size="sm"
                checked={layers[item.layer]}
                onCheckedChange={(on) => onLayer(item.layer, on)}
              />
            </div>
          ))}
          {hiddenDecisions > 0 || showAll ? (
            <div className="mt-1 flex h-7 items-center gap-2 border-t border-border pt-2">
              <Label
                htmlFor="layer-all-decisions"
                className="flex-1 cursor-pointer font-normal text-muted-foreground"
              >
                Every decision
              </Label>
              <Switch
                id="layer-all-decisions"
                size="sm"
                checked={showAll}
                onCheckedChange={onShowAll}
              />
            </div>
          ) : null}
        </section>
        <section className="flex flex-col gap-1.5">
          <SectionLabel>Density</SectionLabel>
          <Segmented
            size="sm"
            aria-label="Density"
            value={density}
            onValueChange={onDensity}
            className="w-full [&>*]:flex-1 [&>*]:justify-center"
            options={[
              { value: 'compact', label: 'Compact' },
              { value: 'comfortable', label: 'Comfortable' },
            ]}
          />
        </section>
        <section className="flex flex-col gap-1.5">
          <SectionLabel>Legend</SectionLabel>
          <div className="grid grid-cols-2 gap-x-2 gap-y-1.5 text-xs text-muted-foreground">
            {LEGEND.map((item) => (
              <span key={item.label} className="flex items-center gap-1.5">
                <span
                  className="size-2 rounded-[3px]"
                  style={{ background: toneFill[item.tone] }}
                />
                {item.label}
              </span>
            ))}
          </div>
          <div className="mt-1.5 flex flex-col gap-1 text-xs text-muted-foreground">
            <span className="flex items-center gap-2">
              <svg width="26" height="6" aria-hidden>
                <line x1="1" y1="3" x2="25" y2="3" stroke="var(--chart-1)" strokeWidth="2" />
                <line
                  x1="1"
                  y1="3"
                  x2="25"
                  y2="3"
                  stroke="var(--surface)"
                  strokeWidth="2.2"
                  strokeDasharray="3 13"
                  className="animate-flow-dash"
                />
              </svg>
              Matched path
            </span>
            <span className="flex items-center gap-2">
              <svg width="26" height="6" aria-hidden>
                <line
                  x1="1"
                  y1="3"
                  x2="25"
                  y2="3"
                  stroke="var(--chart-6)"
                  strokeWidth="1.5"
                  strokeDasharray="4 5"
                  strokeOpacity="0.7"
                />
              </svg>
              Other candidate
            </span>
          </div>
        </section>
      </div>
    </motion.aside>
  )
}

function Fact({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <span className="flex min-w-0 flex-col">
      <span className="text-subtle-foreground">{label}</span>
      <span className="truncate">{children}</span>
    </span>
  )
}

function Chips({ items, tone }: { items: string[]; tone: Tone }) {
  const shown = items.slice(0, 8)
  return (
    <div className="flex flex-wrap gap-1">
      {shown.map((item) => (
        <span
          key={item}
          className="inline-flex h-6 max-w-full items-center gap-1.5 rounded-md border border-border bg-surface-subtle px-1.5 font-mono text-[11.5px]"
        >
          <span className="size-1.5 shrink-0 rounded-full" style={{ background: toneFill[tone] }} />
          <span className="truncate">{item}</span>
        </span>
      ))}
      {items.length > shown.length ? (
        <span className="px-1 text-xs leading-6 text-subtle-foreground">
          +{items.length - shown.length}
        </span>
      ) : null}
    </div>
  )
}

export function Inspector({
  node,
  model,
  scopeId,
  onShowAll,
  onClose,
}: {
  node: Node<TopoNodeData>
  model: TopologyModel
  scopeId?: string
  onShowAll: () => void
  onClose: () => void
}) {
  const navigate = useNavigate()
  const data = node.data
  const recipe = scopeId ? `recipe=${encodeURIComponent(scopeId)}` : ''
  const withRecipe = (path: string, query = '') =>
    `${path}${[recipe, query].filter(Boolean).length ? `?${[recipe, query].filter(Boolean).join('&')}` : ''}`
  const decision =
    data.kind === 'decision' && node.id !== DEFAULT_ID
      ? model.decisions.find((item) => `decision-${item.name}` === node.id)
      : undefined
  const projection =
    data.kind === 'projection'
      ? model.projections.find((item) => `projection-group-${item.mapping}` === node.id)
      : undefined
  const signals = data.signalType ? (model.signalsByType.get(data.signalType) ?? []) : []

  let link: { label: string; to: string } | null = null
  if (decision)
    link = {
      label: 'Open decision',
      to: withRecipe('/routing/decisions', `id=${encodeURIComponent(decision.name)}`),
    }
  else if (data.kind === 'signal' && data.signalType)
    link = {
      label: 'Open signals',
      to: withRecipe('/routing/signals', `type=${encodeURIComponent(data.signalType)}`),
    }
  else if (projection)
    link = {
      label: 'Open projection',
      to: withRecipe(
        '/routing/projections',
        `id=${encodeURIComponent(`mapping:${projection.mapping}`)}`,
      ),
    }
  else if (data.kind === 'model') link = { label: 'Open backend models', to: '/models/backends' }
  else if (data.kind === 'entry') link = { label: 'Open recipes', to: '/routing/recipes' }
  else if (node.id === DEFAULT_ID) link = { label: 'Open router settings', to: '/settings/router' }

  return (
    <motion.aside
      initial={{ opacity: 0, y: 4, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.25, ease }}
      style={{ width: INSPECTOR_WIDTH }}
      className="nodrag nopan nowheel flex cursor-default flex-col overflow-hidden rounded-xl border border-border bg-popover/95 text-left shadow-xl backdrop-blur-md"
    >
      <div className="flex items-start gap-2.5 border-b border-border px-3 py-2.5">
        <span
          className="mt-1.5 size-2 shrink-0 rounded-full"
          style={{ background: toneFill[data.tone] }}
        />
        <div className="flex min-w-0 flex-1 flex-col">
          <span
            className={cn(
              'truncate text-[13px] font-semibold',
              data.kind !== 'signal' && data.kind !== 'more' && 'font-mono',
            )}
          >
            {data.label}
          </span>
          <span className="truncate text-xs text-muted-foreground">
            {node.id === MORE_ID
              ? 'Decisions below the visible priority cut'
              : node.id === ENTRY_ID
                ? 'Public model names for this recipe'
                : data.sublabel}
          </span>
        </div>
        <Button variant="ghost" size="icon-xs" aria-label="Close details" onClick={onClose}>
          <X />
        </Button>
      </div>
      <div className="flex flex-col gap-3 px-3 py-3 text-[13px]">
        {data.warning ? (
          <p className="flex items-start gap-1.5 text-xs text-warning-foreground">
            <CircleAlert className="mt-px size-3.5 shrink-0" />
            {data.warning}
          </p>
        ) : null}
        {decision ? (
          <>
            {decision.description ? (
              <p className="text-xs leading-relaxed text-muted-foreground">
                {decision.description}
              </p>
            ) : null}
            <div className="rounded-md border border-border bg-surface-sunken/60 px-2 py-1.5 font-mono text-[11.5px] leading-relaxed text-foreground/90">
              <span className="text-subtle-foreground">WHEN </span>
              {ruleSummary(decision.rules)}
            </div>
            <div className="grid grid-cols-3 gap-2 text-xs">
              <Fact label="Priority">
                <span className="font-mono">P{decision.priority}</span>
              </Fact>
              <Fact label="Kind">{decisionKind(decision)}</Fact>
              <Fact label="Algorithm">
                {decision.algorithm?.type
                  ? (algorithmMeta(decision.algorithm.type)?.label ?? decision.algorithm.type)
                  : 'First model'}
              </Fact>
            </div>
            {decision.modelRefs.length ? (
              <Chips items={decision.modelRefs.map((ref) => ref.model)} tone="green" />
            ) : null}
            {decision.plugins?.length ? (
              <div className="flex flex-wrap gap-1">
                {decision.plugins.map((plugin) => (
                  <Badge key={plugin.type} tone="rose" size="sm">
                    {pluginLabel(plugin.type)}
                  </Badge>
                ))}
              </div>
            ) : null}
          </>
        ) : null}
        {data.kind === 'signal' ? (
          <>
            <p className="text-xs leading-relaxed text-muted-foreground">
              {signals.length} {signalTypeLabel(data.signalType ?? '').toLowerCase()}{' '}
              {signals.length === 1 ? 'signal' : 'signals'} in this recipe, read by{' '}
              {
                model.decisions.filter((item) =>
                  ruleConditions(item.rules).some(
                    (condition) => condition.type === data.signalType,
                  ),
                ).length
              }{' '}
              decisions.
            </p>
            {signals.length ? (
              <Chips items={signals.map((signal) => signal.name)} tone="gold" />
            ) : null}
          </>
        ) : null}
        {projection ? (
          <>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <Fact label="Score">
                <span className="font-mono">{projection.source || '—'}</span>
              </Fact>
              <Fact label="Method">{projection.method || '—'}</Fact>
            </div>
            <Chips items={projection.outputs} tone="teal" />
          </>
        ) : null}
        {data.kind === 'entry' ? (
          model.entryNames.length ? (
            <Chips items={model.entryNames} tone="blue" />
          ) : (
            <p className="text-xs leading-relaxed text-muted-foreground">
              Requests that name no recipe use this routing. Apps send the public model name in the
              model field.
            </p>
          )
        ) : null}
        {node.id === MORE_ID ? (
          <Button variant="outline" size="sm" className="w-full" onClick={onShowAll}>
            Show every decision
            <ChevronDown />
          </Button>
        ) : null}
        {link ? (
          <Button variant="outline" size="sm" className="w-full" onClick={() => navigate(link.to)}>
            {link.label}
            <ArrowUpRight />
          </Button>
        ) : null}
      </div>
    </motion.aside>
  )
}

function SignalPill({ type, name }: { type: string; name: string }) {
  const Icon = signalTypeIcon(type)
  return (
    <span className="inline-flex h-6 max-w-[11rem] items-center gap-1.5 rounded-md border border-border bg-surface px-1.5 text-xs">
      <Icon className="size-3 shrink-0 text-chart-2" />
      <span className="truncate font-mono text-[11.5px]">{name}</span>
    </span>
  )
}

export function ResultStrip({
  result,
  onPlayground,
}: {
  result: TestQueryResult
  onPlayground: () => void
}) {
  const step = (delay: number) => ({
    initial: { opacity: 0, x: -4 },
    animate: { opacity: 1, x: 0 },
    transition: { delay, duration: 0.3, ease },
  })
  const signals = result.matchedSignals
  const model = result.selectedModel || result.matchedModels[0]
  const ran = result.isAccurate || signals.length > 0 || Boolean(result.matchedDecision)
  const reason =
    result.selectionReason ||
    (result.matchedDecision
      ? result.isFallbackDecision
        ? `Fallback: ${result.fallbackReason ?? 'no decision matched'}`
        : `Matched ${result.matchedDecision}${result.selectionMethod ? ` · ${result.selectionMethod}` : ''}`
      : 'No decision matched this prompt.')
  return (
    <motion.div
      initial={{ opacity: 0, height: 0 }}
      animate={{ opacity: 1, height: 'auto' }}
      exit={{ opacity: 0, height: 0 }}
      transition={{ duration: 0.35, ease }}
      className="overflow-hidden"
      aria-live="polite"
    >
      {result.warning ? (
        <div className="flex items-start gap-2 border-b border-border bg-warning-soft/60 px-3.5 py-2 text-xs text-foreground">
          <TriangleAlert className="mt-px size-3.5 shrink-0 text-warning" />
          <span className="min-w-0 flex-1">
            {result.warning}
            {!result.isAccurate ? (
              <span className="text-muted-foreground"> Nothing is simulated.</span>
            ) : null}
          </span>
        </div>
      ) : null}
      {ran ? (
        <div className="flex flex-wrap items-center gap-x-1.5 gap-y-1.5 border-b border-border px-3 py-2.5">
          {signals.length ? (
            signals.slice(0, 3).map((signal, index) => (
              <motion.span key={`${signal.type}:${signal.name}`} {...step(0.04 * index)}>
                <Tooltip
                  content={`${signalTypeLabel(signal.type)}${signal.confidence != null ? ` · ${(signal.confidence * 100).toFixed(0)}%` : ''}`}
                >
                  <span>
                    <SignalPill type={signal.type} name={signal.name} />
                  </span>
                </Tooltip>
              </motion.span>
            ))
          ) : (
            <span className="text-xs text-muted-foreground">No signal fired</span>
          )}
          {signals.length > 3 ? (
            <span className="text-xs text-subtle-foreground">+{signals.length - 3}</span>
          ) : null}
          <ArrowRight className="size-3.5 shrink-0 text-subtle-foreground" />
          <motion.span {...step(0.1)} className="flex min-w-0 items-center gap-1">
            {result.matchedDecision ? (
              <span className="inline-flex h-6 max-w-[14rem] items-center gap-1.5 rounded-md border border-chart-3/30 bg-chart-3/10 px-1.5 font-mono text-[11.5px]">
                <span className="truncate">{result.matchedDecision}</span>
              </span>
            ) : (
              <span className="text-xs text-muted-foreground">No decision</span>
            )}
          </motion.span>
          <ArrowRight className="size-3.5 shrink-0 text-subtle-foreground" />
          <motion.span {...step(0.16)}>
            {model ? (
              <span className="inline-flex h-6 max-w-[14rem] items-center gap-1.5 rounded-md border border-chart-6/30 bg-chart-6/10 px-1.5 font-mono text-[11.5px]">
                <span className="truncate">{model}</span>
              </span>
            ) : (
              <Badge tone="neutral">No model</Badge>
            )}
          </motion.span>
          {typeof result.routingLatency === 'number' ? (
            <span className="ml-auto pl-2 font-mono text-[11px] tabular text-subtle-foreground">
              {result.routingLatency} ms
            </span>
          ) : null}
        </div>
      ) : null}
      <div className="flex items-center gap-3 border-b border-border px-3.5 py-1.5 text-xs text-muted-foreground">
        <span className="min-w-0 flex-1 truncate" title={ran ? reason : undefined}>
          {ran ? reason : 'Start the Router, then run the prompt again to see its path.'}
        </span>
        <Button variant="link" size="xs" className="h-auto shrink-0" onClick={onPlayground}>
          Open the playground
          <ArrowUpRight />
        </Button>
      </div>
    </motion.div>
  )
}
