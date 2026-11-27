import { ArrowRight, BrainCircuit, Server, Waypoints } from 'lucide-react'

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
  IconTile,
  KeyValueGrid,
  Progress,
  Skeleton,
  StatusLabel,
  type StatusTone,
} from '../ui'
import {
  describeRouterRuntime,
  getActiveRouterRuntime,
  getLoadedModelCount,
  getModelStatusSummary,
  getTotalKnownModelCount,
  type SystemStatus,
} from '../utils/routerRuntime'
import type { DecisionModelHealth } from './dashboardRouterIntelligenceSupport'
import {
  embeddingProviderHealthLabel,
  embeddingProviderTone,
  formatDeploymentType,
  formatEmbeddingProviderBackend,
  formatProviderCheckedAt,
  formatServingMode,
  formatStatusLabel,
} from './statusPageSupport'

const summaryTone: Record<string, StatusTone> = { ok: 'success', warn: 'warning', down: 'danger' }

const embeddingTone: Record<string, StatusTone> = {
  healthy: 'success',
  unhealthy: 'danger',
  pending: 'neutral',
}

function RouterCard({ status }: { status: SystemStatus | null }) {
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
    <Card className="min-w-0" data-testid="status-router">
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
              <span className="text-muted-foreground">Required models ready</span>
              <span className="font-mono tabular-nums text-foreground">
                {loaded} / {total}
              </span>
            </div>
            <Progress
              value={(loaded / total) * 100}
              tone={loaded === total ? 'success' : 'brand'}
              aria-label="Required models ready"
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
            phase and readiness here.
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

function DeploymentCard({ status }: { status: SystemStatus | null }) {
  const history = status?.history
  return (
    <Card className="min-w-0" data-testid="status-deployment">
      <CardHeader>
        <div className="flex min-w-0 items-center gap-3">
          <IconTile icon={Server} tone="slate" size="md" />
          <div className="flex min-w-0 flex-col gap-1">
            <CardTitle>Deployment</CardTitle>
            <CardDescription>How this stack is running</CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent>
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
      </CardContent>
    </Card>
  )
}

/** The configured decision model's deployment state, linking to where it is managed. */
function DecisionModelCard({
  health,
  loading,
  onOpen,
}: {
  health: DecisionModelHealth | null
  loading: boolean
  onOpen: () => void
}) {
  return (
    <Card className="min-w-0" data-testid="status-decision-model">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-3 px-5 py-4">
        <IconTile icon={BrainCircuit} tone="violet" size="md" />
        <div className="flex min-w-0 flex-1 flex-col gap-1">
          <CardTitle>Decision model</CardTitle>
          {health ? (
            <span className="truncate font-mono text-[12.5px] text-muted-foreground">
              {health.model}
              {health.deployment !== health.model ? ` · ${health.deployment}` : ''}
            </span>
          ) : loading ? (
            <Skeleton className="h-4 w-40" />
          ) : (
            <span className="text-[13px] text-muted-foreground">
              The configuration did not load, so the decision model is unknown.
            </span>
          )}
        </div>
        {health ? (
          <span className="flex min-w-0 flex-col items-start gap-0.5 sm:items-end">
            <StatusLabel tone={health.tone}>{health.label}</StatusLabel>
            <span className="text-xs text-muted-foreground">{health.detail}</span>
          </span>
        ) : null}
        <Button variant="outline" size="sm" onClick={onOpen}>
          Decision models
          <ArrowRight />
        </Button>
      </div>
    </Card>
  )
}

export default function StatusRuntimePanel({
  status,
  decisionModel,
  decisionModelLoading,
  onOpenDecisionModels,
}: {
  status: SystemStatus | null
  decisionModel: DecisionModelHealth | null
  decisionModelLoading: boolean
  onOpenDecisionModels: () => void
}) {
  return (
    <div className="flex min-w-0 flex-col gap-4">
      <div className="grid min-w-0 grid-cols-1 gap-4 xl:grid-cols-2">
        <RouterCard status={status} />
        <DeploymentCard status={status} />
      </div>
      <DecisionModelCard
        health={decisionModel}
        loading={decisionModelLoading}
        onOpen={onOpenDecisionModels}
      />
    </div>
  )
}
