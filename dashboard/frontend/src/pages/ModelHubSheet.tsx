import { useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { ArrowUpRight, Cloud, LockOpen, Plus, Route } from 'lucide-react'
import type {
  BuiltInModelCatalog,
  BuiltInModelMetadata,
  CatalogBenchmark,
  CatalogEvaluation,
  CatalogPresentation,
  CatalogProtocol,
  CatalogProvider,
} from '../types/modelCatalog'
import {
  Badge,
  Button,
  DetailSheet,
  DetailSheetBody,
  DetailSheetFooter,
  DetailSheetHeader,
  EmptyState,
  KeyValueGrid,
  MiniStat,
  ProviderLogo,
  SectionLabel,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
  cn,
  useRetainedValue,
} from '../ui'
import {
  modelHubBenchmarkRawValueLabel,
  modelHubBenchmarkValueLabel,
} from './modelHubBenchmarkNormalization'
import { providerProtocolOperations } from './modelHubProviderOperations'
import { resolveModelCatalogIcon } from './modelProviderIcons'
import {
  modelHubContextLabel,
  modelHubDistributionLabel,
  modelHubEvaluationConditionLabel,
  modelHubMaxOutputLabel,
  modelHubParameterLabel,
  modelHubPublicEvaluations,
  modelHubBackendLink,
  modelHubRelationshipLabel,
  readableModelHubValue as readable,
  type ModelHubRow,
} from './modelHubSupport'

export type ModelHubSheetView = 'overview' | 'providers' | 'evidence' | 'pool'

export function HubLogo({
  presentation,
  size = 20,
}: {
  presentation: CatalogPresentation
  size?: number
}) {
  if (presentation.logo.startsWith('package:'))
    return <ProviderLogo provider={presentation.logo.slice('package:'.length)} size={size} framed />
  const src = resolveModelCatalogIcon(presentation.logo)
  return (
    <span
      className="grid shrink-0 place-items-center rounded-lg border border-border bg-surface shadow-xs"
      style={{ width: size + 14, height: size + 14 }}
    >
      {src ? (
        <img
          src={src}
          alt=""
          width={size}
          height={size}
          className={cn('object-contain', presentation.monochrome && 'dark:invert')}
        />
      ) : (
        <span className="text-[11px] font-semibold text-muted-foreground">
          {presentation.monogram}
        </span>
      )}
    </span>
  )
}

export function EvaluationCard({
  model,
  evaluation,
  benchmark,
}: {
  model: BuiltInModelMetadata
  evaluation: CatalogEvaluation
  benchmark: CatalogBenchmark | undefined
}) {
  return (
    <article className="flex flex-col gap-2.5 rounded-lg border border-border bg-card p-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <span className="text-[13px] font-medium">
          {benchmark?.display_name ?? evaluation.benchmark}
        </span>
        <span className="text-xs text-subtle-foreground">
          {readable(evaluation.benchmark_profile)} ·{' '}
          {modelHubEvaluationConditionLabel(model, evaluation.reasoning_effort)}
        </span>
      </div>
      <dl className="grid grid-cols-2 gap-2 sm:grid-cols-3">
        {Object.entries(evaluation.metrics).map(([metric, value]) => {
          const definition = benchmark?.metrics.find((candidate) => candidate.id === metric)
          const label = definition
            ? modelHubBenchmarkValueLabel(value, definition)
            : Number.isInteger(value)
              ? value.toLocaleString()
              : value.toFixed(2)
          const raw = definition ? modelHubBenchmarkRawValueLabel(value, definition) : undefined
          return (
            <div key={metric} className="flex flex-col gap-0.5">
              <dt className="text-[11px] text-subtle-foreground">{readable(metric)}</dt>
              <dd
                className="font-mono text-[13px] font-medium tabular"
                title={raw ? `Raw: ${raw}` : undefined}
              >
                {label}
              </dd>
            </div>
          )
        })}
      </dl>
      <footer className="flex flex-wrap items-center justify-between gap-2 text-xs text-muted-foreground">
        <span>
          {readable(evaluation.evidence.provenance)} · {evaluation.evidence.verification}
          {evaluation.measured_at
            ? ` · measured ${evaluation.measured_at}`
            : evaluation.observed_at
              ? ` · observed ${evaluation.observed_at}`
              : ''}
        </span>
        {evaluation.evidence.source ? (
          <a
            href={evaluation.evidence.source}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-0.5 font-medium text-primary hover:underline"
          >
            Source
            <ArrowUpRight className="size-3" />
          </a>
        ) : null}
      </footer>
    </article>
  )
}

function ProtocolOperations({
  provider,
  protocol,
}: {
  provider: CatalogProvider
  protocol: CatalogProtocol
}) {
  const operations = providerProtocolOperations(provider, protocol)
  if (!operations.length) return null
  return (
    <div className="flex flex-col gap-1.5 rounded-md border border-border bg-surface-subtle/60 p-2.5">
      <div className="flex flex-wrap items-baseline gap-2">
        <strong className="text-xs font-medium">{protocol.display_name}</strong>
        <code className="font-mono text-[11px] text-subtle-foreground">{protocol.id}</code>
      </div>
      <div className="flex flex-col gap-1">
        {operations.map((operation) => (
          <div
            key={operation.reference}
            className="grid grid-cols-[3.25rem_8rem_minmax(0,1fr)] items-center gap-2 text-xs"
          >
            <span className="font-mono text-[10.5px] font-semibold text-chart-1">
              {operation.method}
            </span>
            <strong className="truncate font-medium">{readable(operation.id)}</strong>
            <code className="truncate font-mono text-[11.5px] text-muted-foreground">
              {operation.path}
            </code>
          </div>
        ))}
      </div>
    </div>
  )
}

export function ModelAccess({ row, catalog }: { row: ModelHubRow; catalog: BuiltInModelCatalog }) {
  if (!row.providers.length)
    return (
      <EmptyState
        compact
        icon={Route}
        tone="green"
        title="No direct provider binding"
        description="Virtual models are materialized through their router recipe and backend roles."
      />
    )
  return (
    <div className="flex flex-col gap-3">
      {row.providers.map(({ provider, model }) => {
        const protocols = model.protocols.flatMap((id) => {
          const protocol = catalog.protocols.find((candidate) => candidate.id === id)
          return protocol ? [protocol] : []
        })
        return (
          <section
            key={`${provider.id}/${model.id}`}
            className="flex flex-col gap-3 rounded-lg border border-border bg-card p-3.5"
          >
            <header className="flex items-start gap-3">
              <HubLogo presentation={provider.presentation} size={16} />
              <div className="flex min-w-0 flex-1 flex-col">
                <strong className="truncate text-[13px] font-medium">
                  {provider.display_name}
                </strong>
                <span className="text-xs text-subtle-foreground">
                  {provider.support_tier} support · {readable(model.lifecycle)}
                </span>
              </div>
              <Badge tone="outline" size="sm">
                {modelHubRelationshipLabel[model.relationship]}
              </Badge>
            </header>
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
              <code className="font-mono text-[12px]">{model.id}</code>
              {provider.default_base_url ? (
                <span className="truncate font-mono text-[11.5px] text-subtle-foreground">
                  {provider.default_base_url}
                </span>
              ) : null}
            </div>
            {protocols.length ? (
              <div className="flex flex-col gap-2">
                {protocols.map((protocol) => (
                  <ProtocolOperations key={protocol.id} provider={provider} protocol={protocol} />
                ))}
              </div>
            ) : null}
            {model.pricing && Object.keys(model.pricing).length ? (
              <dl className="grid grid-cols-2 gap-2 sm:grid-cols-3">
                {Object.entries(model.pricing).map(([key, value]) => (
                  <div key={key} className="flex flex-col gap-0.5">
                    <dt className="text-[11px] text-subtle-foreground">{readable(key)}</dt>
                    <dd className="font-mono text-[12.5px] tabular">{String(value)}</dd>
                  </div>
                ))}
              </dl>
            ) : null}
          </section>
        )
      })}
    </div>
  )
}

export function VirtualPool({ row, catalog }: { row: ModelHubRow; catalog: BuiltInModelCatalog }) {
  const models = useMemo(
    () => new Map(catalog.models.map((model) => [model.id, model])),
    [catalog.models],
  )
  if (row.model.kind !== 'virtual')
    return (
      <p className="text-xs text-muted-foreground">
        Backend pools apply only to virtual model recipes.
      </p>
    )
  return (
    <div className="flex flex-col gap-4">
      <KeyValueGrid
        columns={3}
        items={[
          { label: 'Entrypoint', value: row.model.entrypoint ?? '—', mono: true },
          { label: 'Recipe', value: row.model.recipe ?? 'Built-in' },
          { label: 'Roles', value: row.model.roles?.length ?? 0 },
        ]}
      />
      {row.model.roles?.map((role) => (
        <section
          key={role.name}
          className="flex flex-col gap-2.5 rounded-lg border border-border bg-card p-3.5"
        >
          <header className="flex items-start justify-between gap-3">
            <div className="flex flex-col">
              <strong className="text-[13px] font-medium capitalize">{readable(role.name)}</strong>
              <span className="text-xs text-subtle-foreground">
                {role.required ? 'Required role' : 'Optional role'}
              </span>
            </div>
            <Badge tone="outline" size="sm">
              min {role.minimum_candidates}
            </Badge>
          </header>
          {role.traits.length ? (
            <div className="flex flex-wrap gap-1">
              {role.traits.map((trait) => (
                <span
                  key={trait}
                  className="inline-flex h-5 items-center rounded-[5px] border border-border px-1.5 text-[11px] text-muted-foreground"
                >
                  {readable(trait)}
                </span>
              ))}
            </div>
          ) : null}
          {role.recommended_pool.length ? (
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
              {role.recommended_pool.map((id) => {
                const model = models.get(id)
                return (
                  <div
                    key={id}
                    className="flex items-center gap-2.5 rounded-md border border-border bg-surface-subtle/60 px-2.5 py-2"
                  >
                    {model ? (
                      <HubLogo presentation={model.presentation} size={14} />
                    ) : (
                      <span className="grid size-7 shrink-0 place-items-center rounded-lg border border-dashed border-border-strong text-[11px] text-subtle-foreground">
                        C
                      </span>
                    )}
                    <span className="flex min-w-0 flex-col">
                      <span className="truncate text-[12.5px] font-medium">
                        {model?.display_name ?? id}
                      </span>
                      <span className="truncate text-[11px] text-subtle-foreground">
                        {model ? model.publisher : 'Custom model slot'}
                      </span>
                    </span>
                  </div>
                )
              })}
            </div>
          ) : null}
        </section>
      ))}
    </div>
  )
}

function Overview({ row, catalog }: { row: ModelHubRow; catalog: BuiltInModelCatalog }) {
  const model = row.model
  const family = catalog.reasoning_families.find(
    (candidate) => candidate.id === model.reasoning_family,
  )
  return (
    <div className="flex flex-col gap-6">
      <p className="text-[13px] leading-relaxed text-muted-foreground">{model.description}</p>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <MiniStat label="Context" value={modelHubContextLabel(model)} hint="tokens" />
        <MiniStat label="Max output" value={modelHubMaxOutputLabel(model)} hint="tokens" />
        <MiniStat label="Parameters" value={modelHubParameterLabel(model)} />
        <MiniStat
          label="Released"
          value={model.released_at ?? '—'}
          hint={model.knowledge_cutoff ? `cutoff ${model.knowledge_cutoff}` : undefined}
        />
      </div>
      <section className="flex flex-col gap-3">
        <SectionLabel>Specs</SectionLabel>
        <KeyValueGrid
          items={[
            {
              label: 'Creator',
              value: (
                <span className="flex items-center gap-1.5">
                  <HubLogo presentation={model.presentation} size={12} />
                  {model.publisher}
                </span>
              ),
            },
            { label: 'Family', value: model.family, mono: true },
            { label: 'Distribution', value: modelHubDistributionLabel[model.distribution.type] },
            { label: 'Lifecycle', value: readable(model.lifecycle) },
            {
              label: 'Input',
              value: (
                <span className="flex flex-wrap gap-1">
                  {model.modalities.input.map((modality) => (
                    <Badge key={modality} tone="neutral" size="sm">
                      {readable(modality)}
                    </Badge>
                  ))}
                </span>
              ),
            },
            {
              label: 'Output',
              value: (
                <span className="flex flex-wrap gap-1">
                  {model.modalities.output.map((modality) => (
                    <Badge key={modality} tone="neutral" size="sm">
                      {readable(modality)}
                    </Badge>
                  ))}
                </span>
              ),
            },
            {
              label: 'Capabilities',
              span: true,
              value: (
                <span className="flex flex-wrap gap-1">
                  {model.capabilities.map((capability) => (
                    <Badge key={capability} tone="outline" size="sm">
                      {readable(capability)}
                    </Badge>
                  ))}
                </span>
              ),
            },
          ]}
        />
      </section>
      {family ? (
        <section className="flex flex-col gap-2">
          <SectionLabel>Reasoning profile</SectionLabel>
          <p className="text-xs text-muted-foreground">
            <span className="font-mono">{family.id}</span> · default{' '}
            <span className="text-foreground">
              {family.default ? readable(family.default) : 'model selected'}
            </span>
            {family.default_mode ? ` · mode ${readable(family.default_mode)}` : ''}
            {family.disabled ? ` · disabled ${readable(family.disabled)}` : ''}
            {family.activation_parameter
              ? ` · activation ${readable(family.activation_parameter)}`
              : ''}
          </p>
          {[family.levels ?? [], family.modes ?? []]
            .filter((items) => items.length)
            .map((items, index) => (
              <div key={index} className="flex flex-wrap gap-1">
                {items.map((item) => (
                  <Badge
                    key={item}
                    tone={
                      item === family.default || item === family.default_mode ? 'brand' : 'neutral'
                    }
                    size="sm"
                  >
                    {readable(item)}
                  </Badge>
                ))}
              </div>
            ))}
        </section>
      ) : null}
    </div>
  )
}

function Evidence({
  row,
  catalog,
  ready,
}: {
  row: ModelHubRow
  catalog: BuiltInModelCatalog
  ready: boolean
}) {
  if (!ready)
    return (
      <p role="status" className="text-xs text-muted-foreground">
        Model evaluations have not loaded yet.
      </p>
    )
  const records = modelHubPublicEvaluations(catalog).filter(
    (evaluation) => evaluation.model === row.model.id,
  )
  if (!records.length)
    return (
      <EmptyState
        compact
        icon={LockOpen}
        tone="green"
        title="No published evaluations yet"
        description="The catalog keeps missing evidence explicit rather than substituting a synthetic score."
      />
    )
  return (
    <div className="flex flex-col gap-3">
      <p className="text-xs text-muted-foreground">
        Published results from {row.benchmarkCount}{' '}
        {row.benchmarkCount === 1 ? 'benchmark' : 'benchmarks'}, as their sources report them. They
        are evidence for this model, not a ranking.
      </p>
      {records.map((evaluation) => (
        <EvaluationCard
          key={evaluation.id}
          model={row.model}
          evaluation={evaluation}
          benchmark={catalog.benchmarks.find((benchmark) => benchmark.id === evaluation.benchmark)}
        />
      ))}
    </div>
  )
}

export function ModelHubSheet({
  row: current,
  catalog,
  evidenceReady,
  view,
  onViewChange,
  onOpenChange,
}: {
  row: ModelHubRow | undefined
  catalog: BuiltInModelCatalog
  evidenceReady: boolean
  view: ModelHubSheetView
  onViewChange: (view: ModelHubSheetView) => void
  onOpenChange: (open: boolean) => void
}) {
  const navigate = useNavigate()
  const row = useRetainedValue(current)
  if (!row) return null
  const model = row.model
  const DistributionIcon =
    model.distribution.type === 'open_weights'
      ? LockOpen
      : model.distribution.type === 'router_recipe'
        ? Route
        : Cloud
  const tab = view === 'pool' && model.kind !== 'virtual' ? 'overview' : view
  return (
    <DetailSheet open={Boolean(current)} onOpenChange={onOpenChange} size="lg">
      <DetailSheetHeader
        icon={<HubLogo presentation={model.presentation} size={22} />}
        title={model.display_name}
        badges={
          <>
            <Badge tone={model.distribution.type === 'open_weights' ? 'outline' : 'neutral'}>
              <DistributionIcon />
              {modelHubDistributionLabel[model.distribution.type]}
            </Badge>
            {model.lifecycle !== 'active' ? (
              <Badge tone="outline">{readable(model.lifecycle)}</Badge>
            ) : null}
          </>
        }
        description={
          <>
            <span>{model.publisher}</span>
            {model.released_at ? (
              <>
                <span className="text-subtle-foreground">·</span>
                <span>released {model.released_at}</span>
              </>
            ) : null}
            <span className="text-subtle-foreground">·</span>
            <span className="font-mono">{model.id}</span>
          </>
        }
      />
      <Tabs
        value={tab}
        onValueChange={(value) => onViewChange(value as ModelHubSheetView)}
        className="min-h-0 flex-1 gap-0"
      >
        <TabsList className="px-5">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="providers">
            Providers
            <span className="rounded-full bg-secondary px-1.5 font-mono text-[11px] text-muted-foreground">
              {row.providers.length}
            </span>
          </TabsTrigger>
          <TabsTrigger value="evidence">
            Evaluations
            <span className="rounded-full bg-secondary px-1.5 font-mono text-[11px] text-muted-foreground">
              {evidenceReady ? row.evaluationCount : '…'}
            </span>
          </TabsTrigger>
          {model.kind === 'virtual' ? <TabsTrigger value="pool">Backend pool</TabsTrigger> : null}
        </TabsList>
        <DetailSheetBody>
          <TabsContent value="overview" className="px-5 py-5">
            <Overview row={row} catalog={catalog} />
          </TabsContent>
          <TabsContent value="providers" className="px-5 py-5">
            <ModelAccess row={row} catalog={catalog} />
          </TabsContent>
          <TabsContent value="evidence" className="px-5 py-5">
            <Evidence row={row} catalog={catalog} ready={evidenceReady} />
          </TabsContent>
          {model.kind === 'virtual' ? (
            <TabsContent value="pool" className="px-5 py-5">
              <VirtualPool row={row} catalog={catalog} />
            </TabsContent>
          ) : null}
        </DetailSheetBody>
      </Tabs>
      <DetailSheetFooter>
        <Button variant="ghost" size="sm" asChild>
          <a href={model.distribution.source} target="_blank" rel="noreferrer">
            Official source
            <ArrowUpRight />
          </a>
        </Button>
        <div className="ml-auto flex items-center gap-2">
          <Button variant="ghost" size="sm" onClick={() => onOpenChange(false)}>
            Close
          </Button>
          {model.kind === 'physical' ? (
            <Button size="sm" onClick={() => navigate(modelHubBackendLink(model.id))}>
              <Plus />
              Add as backend
            </Button>
          ) : null}
        </div>
      </DetailSheetFooter>
    </DetailSheet>
  )
}
