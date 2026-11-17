import { ArrowRight, BrainCircuit } from 'lucide-react'

import {
  Badge,
  Button,
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  IconTile,
} from '../ui'
import RouterModelInventory from '../components/RouterModelInventory'
import { formatRouterModelLabel } from '../components/routerModelPresentation'
import {
  getLoadedModelCount,
  getTotalKnownModelCount,
  type RouterModelInfo,
  type RouterModelsInfo,
} from '../utils/routerRuntime'
import type { RouterConfig } from './dashboardPageTypes'
import {
  buildIntelligenceRoutingScopes,
  getDecisionRuntimeSummary,
} from './dashboardRouterIntelligenceSupport'

interface DashboardRouterIntelligenceProps {
  config: RouterConfig | null
  modelsInfo?: RouterModelsInfo | null
  onConfigure: () => void
  onOpenStatus: () => void
  onSelectModel: (model: RouterModelInfo) => void
}

const runtimeLabels = {
  ready: 'Shared runtime ready',
  attention: 'Runtime needs attention',
  unreported: 'Runtime not reported',
}

const runtimeTones = { ready: 'success', attention: 'warning', unreported: 'neutral' } as const

function Chips({ items }: { items: Array<{ name: string; kind?: string }> }) {
  return (
    <div className="flex flex-wrap gap-1.5">
      {items.map(({ name, kind }) => (
        <span
          key={`${kind ?? ''}:${name}`}
          className="inline-flex h-6 items-center gap-1.5 rounded-md border border-border bg-surface px-2 text-xs"
        >
          <span className="font-medium text-foreground">{name}</span>
          {kind ? <span className="text-subtle-foreground">{kind}</span> : null}
        </span>
      ))}
    </div>
  )
}

/** The decision model behind learned signals, each recipe's questions and signals, and the runtimes. */
export default function DashboardRouterIntelligence({
  config,
  modelsInfo,
  onConfigure,
  onOpenStatus,
  onSelectModel,
}: DashboardRouterIntelligenceProps) {
  const runtime = config ? getDecisionRuntimeSummary(config, modelsInfo) : null
  const scopes = config ? buildIntelligenceRoutingScopes(config) : []
  const knownModels = getTotalKnownModelCount(modelsInfo)

  return (
    <Card aria-labelledby="router-intelligence-title" role="region" className="xl:col-span-2">
      <CardHeader>
        <CardTitle id="router-intelligence-title">Router intelligence</CardTitle>
        <CardDescription>
          The decision model, what each recipe asks it, and the runtimes.
        </CardDescription>
        <CardAction>
          <Button variant="ghost" size="xs" onClick={onConfigure}>
            Manage decision model
            <ArrowRight />
          </Button>
        </CardAction>
      </CardHeader>
      <CardContent className="flex flex-col gap-5 pt-3">
        {runtime ? (
          <div
            className="flex flex-wrap items-center gap-x-4 gap-y-2 rounded-lg border border-border bg-surface-subtle/60 px-3.5 py-3"
            data-testid="decision-model-overview"
          >
            <IconTile icon={BrainCircuit} tone="violet" size="md" />
            <div className="flex min-w-0 flex-1 flex-col">
              <span className="text-xs text-subtle-foreground">Configured decision model</span>
              <h3 className="truncate font-mono text-[13.5px] font-semibold">{runtime.model}</h3>
              <p className="text-xs text-muted-foreground">
                Default model for learned signals, custom questions and decision selectors.
              </p>
            </div>
            <div className="flex flex-col items-start gap-1 sm:items-end">
              <Badge tone={runtimeTones[runtime.state]} dot>
                {runtimeLabels[runtime.state]}
              </Badge>
              {runtime.bindings > 0 ? (
                <span className="text-xs text-muted-foreground tabular">
                  {runtime.resources} runtime{runtime.resources === 1 ? '' : 's'} ·{' '}
                  {runtime.bindings} reported binding{runtime.bindings === 1 ? '' : 's'}
                </span>
              ) : null}
            </div>
          </div>
        ) : (
          <p className="text-[13px] text-muted-foreground">
            Decision model configuration is unavailable.
          </p>
        )}

        {scopes.length > 0 ? (
          <div className="flex flex-col gap-2">
            <h3 className="text-xs font-medium text-subtle-foreground">Configured routing</h3>
            <div className="flex flex-col divide-y divide-border rounded-lg border border-border">
              {scopes.map((scope) => {
                const signalCount = scope.signals.reduce(
                  (total, group) => total + group.names.length,
                  0,
                )
                return (
                  <details key={scope.id} className="group/scope">
                    <summary className="flex cursor-pointer list-none items-center gap-3 px-3.5 py-2.5 outline-none transition-colors before:size-1.5 before:shrink-0 before:-rotate-45 before:border-b-[1.5px] before:border-r-[1.5px] before:border-subtle-foreground before:transition-transform before:content-[''] hover:bg-accent/50 focus-visible:ring-[3px] focus-visible:ring-ring/40 group-open/scope:before:rotate-45 [&::-webkit-details-marker]:hidden">
                      <span className="flex min-w-0 flex-1 flex-col">
                        <strong className="truncate text-[13px] font-medium">{scope.label}</strong>
                        {scope.entrypoints.length > 0 ? (
                          <span className="truncate font-mono text-[11.5px] text-subtle-foreground">
                            {scope.entrypoints.join(', ')}
                          </span>
                        ) : null}
                      </span>
                      <span className="shrink-0 text-xs text-muted-foreground tabular">
                        {scope.questions.length} questions · {signalCount} other signals ·{' '}
                        {scope.projections.length} projections
                      </span>
                    </summary>
                    <div className="flex flex-col gap-3 border-t border-border bg-surface-subtle/40 px-3.5 py-3">
                      {scope.questions.length > 0 ? (
                        <div className="flex flex-col gap-1.5">
                          <h4 className="text-xs text-subtle-foreground">Custom questions</h4>
                          <Chips items={scope.questions} />
                        </div>
                      ) : null}
                      {scope.signals.map(({ type, names }) => (
                        <div className="flex flex-col gap-1.5" key={type}>
                          <h4 className="text-xs text-subtle-foreground">
                            {formatRouterModelLabel(type)} signals
                          </h4>
                          <Chips items={names.map((name) => ({ name }))} />
                        </div>
                      ))}
                      {scope.projections.length > 0 ? (
                        <div className="flex flex-col gap-1.5">
                          <h4 className="text-xs text-subtle-foreground">Projections</h4>
                          <Chips items={scope.projections} />
                        </div>
                      ) : null}
                      {!scope.questions.length && !signalCount && !scope.projections.length ? (
                        <p className="text-[13px] text-muted-foreground">
                          No signals or projections configured.
                        </p>
                      ) : null}
                    </div>
                  </details>
                )
              })}
            </div>
          </div>
        ) : null}

        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-2">
            <h3 className="text-xs font-medium text-subtle-foreground">Runtime inventory</h3>
            {knownModels > 0 ? (
              <span className="text-xs text-muted-foreground tabular">
                {getLoadedModelCount(modelsInfo)}/{knownModels} runtimes ready
              </span>
            ) : null}
            <Button variant="ghost" size="xs" className="ml-auto" onClick={onOpenStatus}>
              Status
              <ArrowRight />
            </Button>
          </div>
          <RouterModelInventory
            mode="preview"
            modelsInfo={modelsInfo}
            emptyMessage="No runtime inventory reported yet."
            onSelectModel={onSelectModel}
          />
        </div>
      </CardContent>
    </Card>
  )
}
