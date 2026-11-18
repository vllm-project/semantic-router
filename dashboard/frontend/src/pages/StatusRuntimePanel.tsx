import { useState } from 'react'
import { Boxes, Server, Waypoints } from 'lucide-react'

import {
  Alert,
  Badge,
  Button,
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  CopyButton,
  DataTable,
  EmptyState,
  IconTile,
  KeyValueGrid,
  Progress,
  StatusLabel,
  type Column,
  type StatusTone,
} from '../ui'
import {
  describeRouterRuntime,
  getActiveRouterRuntime,
  getLoadedModelCount,
  getModelStatusSummary,
  getPreviewRouterModels,
  getRouterModelState,
  getRouterModelStateLabel,
  getTotalKnownModelCount,
  type RouterModelInfo,
  type SystemStatus,
} from '../utils/routerRuntime'
import {
  embeddingProviderHealthLabel,
  embeddingProviderTone,
  formatDeploymentType,
  formatEmbeddingProviderBackend,
  formatProviderCheckedAt,
  formatServingMode,
  formatStatusLabel,
} from './statusPageSupport'

/** The status endpoint also reports the public listener URLs. */
export type StatusSnapshot = SystemStatus & { endpoints?: string[] }

const summaryTone: Record<string, StatusTone> = { ok: 'success', warn: 'warning', down: 'danger' }

const modelTone = (state: string): StatusTone =>
  state === 'ready'
    ? 'success'
    : state === 'downloading' || state === 'initializing'
      ? 'info'
      : state === 'pending'
        ? 'warning'
        : 'neutral'

const embeddingTone: Record<string, StatusTone> = {
  healthy: 'success',
  unhealthy: 'danger',
  pending: 'neutral',
}

const MODEL_PREVIEW = 8

function RouterCard({ status }: { status: StatusSnapshot | null }) {
  const summary = getModelStatusSummary(status)
  const runtime = status?.router_runtime
  const starting = getActiveRouterRuntime(status)
  const models = status?.models
  const loaded = models ? getLoadedModelCount(models) : (runtime?.ready_models ?? 0)
  const total = models ? getTotalKnownModelCount(models) : (runtime?.total_models ?? 0)
  const provider = runtime?.embedding_provider
  const system = models?.system
  const message = runtime?.message || models?.summary?.message

  return (
    <Card className="min-w-0">
      <CardHeader className="flex flex-wrap items-start justify-between gap-x-4 gap-y-2">
        <div className="flex min-w-0 items-center gap-3">
          <IconTile icon={Waypoints} tone="slate" size="md" />
          <div className="flex min-w-0 flex-col gap-1">
            <CardTitle>Router</CardTitle>
            <CardDescription>Live state reported by the Router process</CardDescription>
          </div>
        </div>
        <span className="flex flex-col items-end gap-0.5">
          <StatusLabel tone={summaryTone[summary.tone]} pulse={summary.tone === 'warn'}>
            {summary.value}
          </StatusLabel>
          <span className="text-xs text-muted-foreground">{summary.detail}</span>
        </span>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {total > 0 ? (
          <div className="flex flex-col gap-1.5">
            <div className="flex items-baseline justify-between text-xs">
              <span className="text-muted-foreground">Router models ready</span>
              <span className="font-mono tabular-nums text-foreground">
                {loaded} / {total}
              </span>
            </div>
            <Progress
              value={(loaded / total) * 100}
              tone={loaded === total ? 'success' : 'brand'}
              aria-label="Router models ready"
            />
          </div>
        ) : null}
        <KeyValueGrid
          items={[
            { label: 'Phase', value: runtime ? formatStatusLabel(runtime.phase) : 'Not reported' },
            {
              label: 'Serving',
              value: runtime
                ? runtime.ready
                  ? 'Ready for requests'
                  : 'Not ready'
                : 'Not reported',
            },
            ...(starting
              ? [{ label: 'Startup', value: describeRouterRuntime(starting), span: true }]
              : []),
            ...(runtime?.downloading_model
              ? [{ label: 'Downloading', value: runtime.downloading_model, mono: true, span: true }]
              : []),
            ...(runtime?.pending_models?.length
              ? [
                  {
                    label: 'Pending models',
                    span: true,
                    value: (
                      <span className="flex flex-wrap gap-1">
                        {runtime.pending_models.map((model) => (
                          <Badge key={model} tone="outline" size="sm" className="font-mono">
                            {model}
                          </Badge>
                        ))}
                      </span>
                    ),
                  },
                ]
              : []),
            ...(system
              ? [
                  {
                    label: 'Runtime',
                    value: `${system.go_version} · ${system.os}/${system.architecture}`,
                  },
                  {
                    label: 'Memory · accelerator',
                    value: `${system.memory_usage} · ${system.gpu_available ? 'GPU available' : 'CPU only'}`,
                  },
                ]
              : []),
          ]}
        />
        {message && runtime?.phase !== 'error' ? (
          <p className="text-xs text-muted-foreground">{message}</p>
        ) : null}
        {!runtime && !models ? (
          <p className="text-xs leading-relaxed text-muted-foreground">
            The Router has not reported its runtime. Start it with{' '}
            <code className="font-mono text-foreground">vllm-sr serve</code> to see its startup
            phase, signal models and embedding provider here.
          </p>
        ) : null}
        {runtime?.phase === 'error' ? (
          <Alert tone="danger" title="Router startup failed">
            {runtime.message || 'The Router reported a startup error. Check the Router logs.'}
          </Alert>
        ) : null}
        {provider ? (
          <div className="flex flex-col gap-3 rounded-lg border border-border bg-surface-subtle/60 p-3.5">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span className="text-[13px] font-medium">Embedding provider</span>
              <StatusLabel tone={embeddingTone[embeddingProviderTone(provider)]}>
                {embeddingProviderHealthLabel(provider)}
              </StatusLabel>
            </div>
            <KeyValueGrid
              items={[
                { label: 'Mode', value: formatStatusLabel(provider.mode ?? 'not reported') },
                { label: 'Backend', value: formatEmbeddingProviderBackend(provider.backend) },
                { label: 'Model', value: provider.model ?? 'Not reported', mono: true },
                {
                  label: 'Dimension',
                  value: provider.dimension ? provider.dimension.toLocaleString() : 'Not reported',
                },
                ...(provider.api_key_env
                  ? [
                      {
                        label: 'API key',
                        value: (
                          <span className="flex flex-wrap items-center gap-1.5">
                            <span className="font-mono text-[12.5px]">{provider.api_key_env}</span>
                            <Badge
                              size="sm"
                              tone={provider.api_key_env_set ? 'success' : 'warning'}
                            >
                              {provider.api_key_env_set ? 'Set' : 'Missing'}
                            </Badge>
                          </span>
                        ),
                      },
                    ]
                  : []),
                { label: 'Last checked', value: formatProviderCheckedAt(provider.last_checked_at) },
              ]}
            />
            {provider.last_probe_error ? (
              <Alert tone="danger" title="Last probe failed">
                {provider.last_probe_error}
              </Alert>
            ) : null}
          </div>
        ) : null}
      </CardContent>
    </Card>
  )
}

function DeploymentCard({ status }: { status: StatusSnapshot | null }) {
  const history = status?.history
  const endpoints = status?.endpoints ?? []
  return (
    <Card className="min-w-0">
      <CardHeader>
        <div className="flex min-w-0 items-center gap-3">
          <IconTile icon={Server} tone="slate" size="md" />
          <div className="flex min-w-0 flex-col gap-1">
            <CardTitle>Deployment</CardTitle>
            <CardDescription>How this stack is running</CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <KeyValueGrid
          items={[
            {
              label: 'Version',
              value: status?.version ? (
                <span className="flex items-center gap-1">
                  <span className="truncate font-mono text-[12.5px]">{status.version}</span>
                  <CopyButton
                    value={status.version}
                    label="Copy version"
                    className="-my-1 size-6"
                  />
                </span>
              ) : (
                'Not reported'
              ),
            },
            { label: 'Deployment', value: formatDeploymentType(status?.deployment_type) },
            { label: 'Serving mode', value: formatServingMode(status?.serving_mode) },
            {
              label: 'History',
              value: history
                ? `${history.windowHours} hours through ${history.through.slice(11, 16)} UTC`
                : 'Not recorded',
            },
          ]}
        />
        <div className="flex flex-col gap-1.5">
          <span className="text-xs text-subtle-foreground">Public endpoints</span>
          {endpoints.length ? (
            <ul className="flex flex-col divide-y divide-border rounded-lg border border-border">
              {endpoints.map((endpoint) => (
                <li key={endpoint} className="flex min-w-0 items-center gap-2 py-1 pl-3 pr-1">
                  <span className="min-w-0 flex-1 truncate font-mono text-[12.5px]">
                    {endpoint}
                  </span>
                  <CopyButton value={endpoint} label={`Copy ${endpoint}`} />
                </li>
              ))}
            </ul>
          ) : (
            <span className="text-[13px] text-muted-foreground">
              No endpoints reported. They appear once a listener is serving.
            </span>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

function RouterModels({ models }: { models: RouterModelInfo[] }) {
  const [expanded, setExpanded] = useState(false)
  const rows = expanded ? models : models.slice(0, MODEL_PREVIEW)
  const ready = models.filter((model) => getRouterModelState(model) === 'ready').length
  const columns: Column<RouterModelInfo>[] = [
    {
      id: 'model',
      header: 'Model',
      className: 'w-full max-w-0',
      cell: (model) => (
        <div className="flex min-w-0 flex-col gap-0.5">
          <span className="truncate font-mono text-[12.5px] text-foreground">
            {model.registry?.repo_id ?? model.name}
          </span>
          <span className="truncate text-xs text-subtle-foreground">
            {model.recipe && model.recipe !== 'default' ? `${model.recipe} / ` : ''}
            {model.name}
          </span>
        </div>
      ),
    },
    {
      id: 'type',
      header: 'Purpose',
      hideBelow: 'md',
      cell: (model) => (
        <span className="text-xs text-muted-foreground">{formatStatusLabel(model.type)}</span>
      ),
    },
    {
      id: 'state',
      header: 'State',
      width: '140px',
      cell: (model) => {
        const state = getRouterModelState(model)
        return (
          <StatusLabel tone={modelTone(state)} pulse={state === 'downloading'}>
            {getRouterModelStateLabel(model)}
          </StatusLabel>
        )
      },
    },
    {
      id: 'memory',
      header: 'Memory',
      align: 'right',
      hideBelow: 'lg',
      cell: (model) => (
        <span className="font-mono text-xs tabular-nums text-muted-foreground">
          {model.memory_usage ?? '—'}
        </span>
      ),
    },
  ]
  return (
    <section className="flex min-w-0 flex-col gap-3" aria-labelledby="status-router-models">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div className="flex flex-col gap-0.5">
          <h2 id="status-router-models" className="text-[15px] font-semibold tracking-[-0.012em]">
            Router models
          </h2>
          <p className="text-[13px] text-muted-foreground">
            Classifiers and embedding models the Router loads for signals.
          </p>
        </div>
        <Badge tone={ready === models.length ? 'success' : 'warning'}>
          {ready} of {models.length} ready
        </Badge>
      </div>
      <DataTable
        data={rows}
        columns={columns}
        getRowId={(model) => `${model.recipe ?? 'default'}:${model.name}`}
      />
      {models.length > MODEL_PREVIEW ? (
        <Button
          variant="ghost"
          size="sm"
          className="self-start"
          onClick={() => setExpanded((value) => !value)}
        >
          {expanded ? 'Show fewer' : `Show all ${models.length} models`}
        </Button>
      ) : null}
    </section>
  )
}

export default function StatusRuntimePanel({ status }: { status: StatusSnapshot | null }) {
  const models = getPreviewRouterModels(status?.models)
  return (
    <div className="flex min-w-0 flex-col gap-6">
      <div className="grid min-w-0 grid-cols-1 items-start gap-4 xl:grid-cols-2">
        <RouterCard status={status} />
        <DeploymentCard status={status} />
      </div>
      {models.length ? (
        <RouterModels models={models} />
      ) : status?.models ? (
        <Card className="border-dashed shadow-none">
          <EmptyState
            compact
            icon={Boxes}
            tone="slate"
            title="No router models reported"
            description="Classifier and embedding models appear here once the Router loads its recipes."
          />
        </Card>
      ) : null}
    </div>
  )
}
