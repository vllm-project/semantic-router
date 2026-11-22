import type { ReactNode } from 'react'
import { Boxes, Brain, CircleDollarSign, Cpu, Info, Pencil, Radio, Server } from 'lucide-react'

import {
  Alert,
  Badge,
  Button,
  Card,
  CopyField,
  EmptyState,
  KeyValueGrid,
  MiniStat,
  SectionLabel,
  StatusLabel,
  Tooltip,
} from '../ui'
import type { BuiltInModelCatalog } from '../types/modelCatalog'
import type { ModelHealth, ModelHealthEndpoint } from '../utils/modelHealthApi'
import { formatLatency, MODEL_HEALTH_META } from './configPageModelHealthPresentation'
import { CatalogProviderLogo } from './ConfigPageModelLogo'
import {
  formatModelPrice,
  providerDisplayName,
  type NamedDeployment,
} from './configPageModelPresentation'
import type {
  BackendRefEntry,
  ModelReasoningConfig,
  NormalizedModel,
  ProviderReliability,
  ReasoningFamily,
} from './configPageSupport'

/** Deployments every configuration declares; a same-named entry replaces one. */
const BUILT_IN_DEPLOYMENTS = new Set(['primary', 'vela-embedding'])

type GridItem = { label: string; value: ReactNode; mono?: boolean; span?: boolean }

const compact = (items: Array<GridItem | null>): GridItem[] =>
  items.filter((item): item is GridItem => item !== null)

const present = (value: unknown) =>
  value !== undefined && value !== null && value !== '' && !(Array.isArray(value) && !value.length)

function Chips({ values, mono = true }: { values: readonly string[]; mono?: boolean }) {
  return (
    <span className="flex flex-wrap gap-1">
      {values.map((value) => (
        <Badge key={value} size="sm" tone="outline" className={mono ? 'font-mono' : undefined}>
          {value}
        </Badge>
      ))}
    </span>
  )
}

function EditAction({
  readOnly,
  onEdit,
  children,
}: {
  readOnly: boolean
  onEdit: () => void
  children: ReactNode
}) {
  if (readOnly) return null
  return (
    <Button variant="outline" size="sm" onClick={onEdit}>
      <Pencil />
      {children}
    </Button>
  )
}

function backendAuthItems(backend: BackendRefEntry): Array<GridItem | null> {
  const auth = backend.auth
  return [
    auth?.credential
      ? {
          label: 'Credential',
          value:
            auth.credential === 'delegated'
              ? "The signed-in user's credential"
              : 'The provider login the caller sent',
        }
      : null,
    auth?.api_key_env ? { label: 'API key variable', value: auth.api_key_env, mono: true } : null,
    auth?.api_key_file ? { label: 'API key file', value: auth.api_key_file, mono: true } : null,
    backend.api_key ? { label: 'Inline key', value: 'Saved to an owner-only key file' } : null,
    auth?.header ? { label: 'Auth header', value: auth.header, mono: true } : null,
    auth?.prefix !== undefined
      ? { label: 'Auth prefix', value: auth.prefix || '(none — bare key)', mono: true }
      : null,
  ]
}

function EndpointHealth({ health }: { health: ModelHealthEndpoint }) {
  const meta = MODEL_HEALTH_META[health.status]
  const label = (
    <span className="shrink-0">
      <StatusLabel tone={meta.tone}>
        <span className="font-mono text-xs tabular">
          {health.latencyMs !== undefined ? formatLatency(health.latencyMs) : meta.label}
        </span>
      </StatusLabel>
    </span>
  )
  return health.reason ? <Tooltip content={health.reason}>{label}</Tooltip> : label
}

function BackendCard({
  backend,
  index,
  masked,
  catalog,
  health,
}: {
  backend: BackendRefEntry
  index: number
  masked: boolean
  catalog: BuiltInModelCatalog
  health?: ModelHealthEndpoint
}) {
  const endpoint = backend.endpoint?.trim()
  const protocol =
    backend.protocol || (endpoint?.startsWith('https://') ? 'https' : endpoint ? 'http' : '')
  const headers = Object.keys(backend.headers ?? {})
  const tls = backend.tls
  const items = compact([
    backend.provider
      ? { label: 'Provider', value: providerDisplayName(backend.provider, catalog) }
      : null,
    backend.api_version ? { label: 'API version', value: backend.api_version, mono: true } : null,
    backend.chat_path ? { label: 'Chat path', value: backend.chat_path, mono: true } : null,
    ...backendAuthItems(backend),
    headers.length ? { label: 'Headers', value: <Chips values={headers} />, span: true } : null,
    tls?.ca_file ? { label: 'TLS CA bundle', value: tls.ca_file, mono: true } : null,
    tls?.server_name ? { label: 'TLS server name', value: tls.server_name, mono: true } : null,
    tls?.cert_file ? { label: 'Client certificate', value: tls.cert_file, mono: true } : null,
    tls?.key_file ? { label: 'Client key', value: tls.key_file, mono: true } : null,
  ])
  return (
    <Card className="gap-3 p-3.5">
      <div className="flex items-center gap-2.5">
        {backend.provider ? (
          <CatalogProviderLogo provider={backend.provider} size={16} catalog={catalog} />
        ) : (
          <Radio className="size-4 text-chart-6" />
        )}
        <span className="min-w-0 flex-1 truncate text-[13px] font-medium">
          {backend.name?.trim() || backend.provider || `Backend ${index + 1}`}
        </span>
        {health ? <EndpointHealth health={health} /> : null}
        {protocol ? (
          <Badge size="sm" tone={protocol === 'https' ? 'success' : 'outline'}>
            {protocol.toUpperCase()}
          </Badge>
        ) : null}
        <span className="font-mono text-xs tabular text-subtle-foreground">
          weight {backend.weight ?? 1}
        </span>
      </div>
      {endpoint ? (
        masked ? (
          <span className="flex h-9 items-center rounded-md border border-dashed border-border px-3 text-xs text-muted-foreground">
            Endpoint hidden in read-only mode
          </span>
        ) : (
          <CopyField value={endpoint} />
        )
      ) : (
        <Alert tone="warning">This backend has no endpoint. Edit the model to add one.</Alert>
      )}
      {items.length ? <KeyValueGrid items={items} className="gap-y-3" /> : null}
    </Card>
  )
}

const reliabilityLabels: Array<[keyof ProviderReliability, string]> = [
  ['lb_policy', 'Load balancing'],
  ['retry_on', 'Retry conditions'],
  ['health_check_path', 'Health check path'],
  ['health_check_interval', 'Health check interval'],
  ['health_check_timeout', 'Health check timeout'],
  ['consecutive_5xx', 'Errors before ejection'],
  ['base_ejection_time', 'Base ejection time'],
  ['max_ejection_percent', 'Max ejection percent'],
  ['connect_timeout', 'Connect timeout'],
  ['total_timeout', 'Total timeout'],
  ['idle_timeout', 'Stream idle timeout'],
  ['per_try_timeout', 'Per-try timeout'],
  ['first_byte_timeout', 'First-byte timeout'],
  ['retry_back_off_base', 'Retry back-off base'],
  ['retry_back_off_max', 'Retry back-off max'],
  ['retry_after_max', 'Honor Retry-After up to'],
  ['retry_budget_percent', 'Retry budget percent'],
  ['retry_budget_min_concurrency', 'Retry budget minimum'],
  ['retriable_status_codes', 'Retriable status codes'],
]

// The retries the Router and the Envoy the CLI renders apply: the count,
// else two by default unless retry conditions are set without one.
function retrySummary(reliability?: ProviderReliability): string {
  if (reliability?.retry_count !== undefined) return `${reliability.retry_count} retries`
  return reliability?.retry_on?.trim()
    ? '0 retries: retry conditions without a count'
    : '2 retries by default'
}

export function ModelEndpointsPanel({
  model,
  deploymentName,
  masked,
  readOnly,
  catalog,
  health,
  onEdit,
  onShowDeployment,
}: {
  model: NormalizedModel
  deploymentName: string
  masked: boolean
  readOnly: boolean
  catalog: BuiltInModelCatalog
  health?: ModelHealth
  onEdit: () => void
  onShowDeployment: () => void
}) {
  const backends = model.backend_refs ?? []
  const externalIds = Object.entries(model.external_model_ids ?? {})
  const reliability = model.reliability
  const reliabilityItems = reliabilityLabels
    .filter(([key]) => present(reliability?.[key]))
    .map(([key, label]) => {
      const value = reliability?.[key]
      return {
        label,
        value: Array.isArray(value) ? value.join(', ') : String(value),
        mono: true,
      }
    })
  return (
    <div className="flex flex-col gap-5">
      {deploymentName ? (
        <Alert
          tone="info"
          title={`Served by the shared deployment ${deploymentName}`}
          actions={
            <Button variant="outline" size="sm" onClick={onShowDeployment}>
              View deployment
            </Button>
          }
        >
          This alias names a deployment instead of its own endpoint and key.
        </Alert>
      ) : null}
      <section className="flex flex-col gap-2.5">
        <div className="flex items-center justify-between gap-3">
          <SectionLabel>Backend refs · {backends.length}</SectionLabel>
          <EditAction readOnly={readOnly} onEdit={onEdit}>
            Edit backends
          </EditAction>
        </div>
        {backends.length ? (
          backends.map((backend, index) => (
            <BackendCard
              key={`${backend.name ?? 'backend'}-${index}`}
              backend={backend}
              index={index}
              masked={masked}
              catalog={catalog}
              health={health?.endpoints.find((endpoint) => endpoint.name === backend.name?.trim())}
            />
          ))
        ) : deploymentName ? null : (
          <div className="rounded-lg border border-dashed border-border-strong">
            <EmptyState
              compact
              icon={Server}
              tone="green"
              title="No endpoints"
              description="Without a backend this model can be listed in decisions but cannot answer."
            />
          </div>
        )}
      </section>
      {externalIds.length ? (
        <section className="flex flex-col gap-2">
          <SectionLabel>External model IDs</SectionLabel>
          <KeyValueGrid
            items={externalIds.map(([provider, id]) => ({
              label: providerDisplayName(provider, catalog),
              value: id,
              mono: true,
            }))}
          />
        </section>
      ) : null}
      <section className="flex flex-col gap-2">
        <SectionLabel>Delivery</SectionLabel>
        <div className="flex flex-col gap-3 rounded-lg border border-border bg-surface-subtle/50 p-3.5">
          <span className="text-[13px] font-medium">{retrySummary(reliability)}</span>
          {reliabilityItems.length ? (
            <KeyValueGrid items={reliabilityItems} className="gap-y-3" />
          ) : (
            <span className="text-xs text-muted-foreground">
              Platform defaults for load balancing, health checks and timeouts.
            </span>
          )}
        </div>
      </section>
    </div>
  )
}

export function ModelPricingPanel({
  model,
  readOnly,
  onEdit,
}: {
  model: NormalizedModel
  readOnly: boolean
  onEdit: () => void
}) {
  const pricing = model.pricing
  const currency = pricing?.currency || 'USD'
  const rates: Array<[string, number | undefined]> = [
    ['Prompt', pricing?.prompt_per_1m],
    ['Completion', pricing?.completion_per_1m],
    ['Cached input', pricing?.cached_input_per_1m],
    ['Cache write', pricing?.cache_write_per_1m],
    ['Cache write (1h)', pricing?.cache_write_1h_per_1m],
  ]
  const priced = rates.some(([, value]) => typeof value === 'number')
  if (!priced) {
    return (
      <div className="rounded-lg border border-dashed border-border-strong">
        <EmptyState
          compact
          icon={CircleDollarSign}
          tone="gold"
          title="Not priced"
          description="Requests to this model are reported without a cost. Add per-million-token rates to track spend."
          actions={
            <EditAction readOnly={readOnly} onEdit={onEdit}>
              Add pricing
            </EditAction>
          }
        />
      </div>
    )
  }
  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between gap-3">
        <SectionLabel>Per one million tokens · {currency}</SectionLabel>
        <EditAction readOnly={readOnly} onEdit={onEdit}>
          Edit pricing
        </EditAction>
      </div>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        {rates.map(([label, value]) => (
          <MiniStat
            key={label}
            label={label}
            value={
              <span
                className={
                  typeof value === 'number'
                    ? 'font-mono text-base'
                    : 'text-base text-subtle-foreground'
                }
              >
                {formatModelPrice(value, currency) ?? 'Not set'}
              </span>
            }
          />
        ))}
      </div>
      <p className="text-xs text-muted-foreground">
        Cost reporting multiplies these rates by the tokens each request uses. Cache rates apply
        only to providers that report cached tokens.
      </p>
    </div>
  )
}

function reasoningItems(reasoning: Partial<ModelReasoningConfig & ReasoningFamily>) {
  return compact([
    reasoning.type ? { label: 'Type', value: reasoning.type.replace(/_/g, ' ') } : null,
    reasoning.parameter ? { label: 'Parameter', value: reasoning.parameter, mono: true } : null,
    reasoning.activation_parameter
      ? { label: 'Activation parameter', value: reasoning.activation_parameter, mono: true }
      : null,
    reasoning.levels?.length
      ? { label: 'Effort levels', value: <Chips values={reasoning.levels} /> }
      : null,
    reasoning.default ? { label: 'Default effort', value: reasoning.default, mono: true } : null,
    reasoning.modes?.length ? { label: 'Modes', value: <Chips values={reasoning.modes} /> } : null,
    reasoning.default_mode
      ? { label: 'Default mode', value: reasoning.default_mode, mono: true }
      : null,
    reasoning.disabled ? { label: 'Disabled value', value: reasoning.disabled, mono: true } : null,
    Object.keys(reasoning.effort_flags ?? {}).length
      ? {
          label: 'Effort flags',
          value: (
            <Chips
              values={Object.entries(reasoning.effort_flags ?? {}).map(
                ([effort, flag]) => `${effort} → ${flag}`,
              )}
            />
          ),
          span: true,
        }
      : null,
  ])
}

export function ModelReasoningPanel({
  model,
  families,
  catalog,
  readOnly,
  onEdit,
}: {
  model: NormalizedModel
  families: Record<string, ReasoningFamily>
  catalog: BuiltInModelCatalog
  readOnly: boolean
  onEdit: () => void
}) {
  const family = model.reasoning_family ? families[model.reasoning_family] : undefined
  const inline = !model.reasoning_family && model.reasoning?.type ? model.reasoning : undefined
  const card = model.catalog
    ? catalog.models.find((entry) => entry.id === model.catalog)
    : undefined
  const constraints = compact([
    model.reasoning_efforts?.length
      ? { label: 'Efforts the provider accepts', value: <Chips values={model.reasoning_efforts} /> }
      : null,
    model.reasoning_modes?.length
      ? { label: 'Modes the provider accepts', value: <Chips values={model.reasoning_modes} /> }
      : null,
  ])

  if (!family && !inline && !(card && constraints.length)) {
    return (
      <div className="rounded-lg border border-dashed border-border-strong">
        <EmptyState
          compact
          icon={Brain}
          tone="violet"
          title="No reasoning controls"
          description={
            card
              ? `The catalog lists no reasoning controls for ${card.display_name}.`
              : 'Decisions cannot turn reasoning on or set an effort for this model. Choose a reasoning family or describe inline settings.'
          }
          actions={
            card ? undefined : (
              <EditAction readOnly={readOnly} onEdit={onEdit}>
                Set reasoning
              </EditAction>
            )
          }
        />
      </div>
    )
  }
  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-border bg-surface-subtle/50 px-3.5 py-3">
        <span className="flex min-w-0 flex-col gap-0.5">
          <span className="text-xs text-subtle-foreground">
            {card
              ? `Inherited from the catalog model ${card.display_name}`
              : inline
                ? 'Inline settings for a custom model'
                : 'Built-in reasoning family'}
          </span>
          <span className="font-mono text-[13px] font-medium">
            {model.reasoning_family ?? inline?.type ?? 'Provider reasoning controls'}
          </span>
        </span>
        <EditAction readOnly={readOnly} onEdit={onEdit}>
          Edit reasoning
        </EditAction>
      </div>
      {family || inline ? <KeyValueGrid items={reasoningItems(family ?? inline ?? {})} /> : null}
      {constraints.length ? (
        <section className="flex flex-col gap-2">
          <SectionLabel>Provider binding</SectionLabel>
          <KeyValueGrid items={constraints} />
        </section>
      ) : null}
    </div>
  )
}

function deploymentItems(deployment: NamedDeployment['deployment']): GridItem[] {
  if (!deployment) return []
  const replicas = deployment.replicas ?? []
  return compact([
    deployment.provider ? { label: 'API', value: deployment.provider.replace(/_/g, ' ') } : null,
    deployment.artifact ? { label: 'Artifact', value: deployment.artifact, mono: true } : null,
    deployment.revision
      ? { label: 'Revision', value: deployment.revision.slice(0, 12), mono: true }
      : null,
    deployment.device ? { label: 'Device', value: deployment.device, mono: true } : null,
    deployment.endpoint ? { label: 'Endpoint', value: deployment.endpoint, mono: true } : null,
    deployment.model ? { label: 'Served model', value: deployment.model, mono: true } : null,
    deployment.public_name
      ? { label: 'Public name', value: deployment.public_name, mono: true }
      : null,
    replicas.length
      ? {
          label: 'Replicas',
          value: replicas
            .map((replica) => replica.endpoint ?? `local · ${replica.device ?? 'auto'}`)
            .join(', '),
          span: true,
        }
      : null,
    deployment.timeout_ms ? { label: 'Timeout', value: `${deployment.timeout_ms} ms` } : null,
    deployment.max_retries !== undefined
      ? { label: 'Max retries', value: String(deployment.max_retries) }
      : null,
    deployment.auth?.api_key_env
      ? { label: 'API key variable', value: deployment.auth.api_key_env, mono: true }
      : null,
    deployment.capabilities?.question_types?.length
      ? {
          label: 'Question types',
          value: <Chips values={deployment.capabilities.question_types} />,
          span: true,
        }
      : null,
    deployment.pricing
      ? {
          label: 'Pricing',
          value: `${formatModelPrice(deployment.pricing.prompt_per_1m, deployment.pricing.currency) ?? '—'} in · ${formatModelPrice(deployment.pricing.completion_per_1m, deployment.pricing.currency) ?? '—'} out`,
        }
      : null,
  ])
}

function DeploymentCard({ entry, note }: { entry: NamedDeployment; note: string }) {
  const items = deploymentItems(entry.deployment)
  const Icon = entry.deployment?.provider === 'model_runtime' ? Cpu : Boxes
  return (
    <Card className="gap-3 p-3.5">
      <div className="flex items-start gap-2.5">
        <span className="grid size-7 shrink-0 place-items-center rounded-md bg-surface-subtle text-muted-foreground">
          <Icon className="size-4" />
        </span>
        <span className="flex min-w-0 flex-1 flex-col">
          <span className="truncate font-mono text-[13px] font-medium">{entry.name}</span>
          <span className="text-xs text-muted-foreground">{note}</span>
        </span>
        {entry.deployment?.provider ? (
          <Badge size="sm" tone="outline">
            {entry.deployment.provider.replace(/_/g, ' ')}
          </Badge>
        ) : null}
      </div>
      {entry.deployment ? (
        items.length ? (
          <KeyValueGrid items={items} className="gap-y-3" />
        ) : null
      ) : BUILT_IN_DEPLOYMENTS.has(entry.name) ? (
        <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <Info className="size-3.5" />
          Built-in deployment, declared by every configuration.
        </p>
      ) : (
        <Alert tone="warning">
          {entry.name} is not declared under global.model_catalog.deployments. Declare it in the
          config editor before this model can answer.
        </Alert>
      )}
    </Card>
  )
}

export function ModelDeploymentsPanel({
  referenced,
  naming,
}: {
  referenced: NamedDeployment | null
  naming: NamedDeployment[]
}) {
  if (!referenced && naming.length === 0) {
    return (
      <div className="rounded-lg border border-dashed border-border-strong">
        <EmptyState
          compact
          icon={Boxes}
          tone="slate"
          title="No shared deployment"
          description="This model reaches its backends through its own endpoints. A shared deployment is declared once under global.model_catalog.deployments and named by its aliases."
        />
      </div>
    )
  }
  return (
    <div className="flex flex-col gap-5">
      {referenced ? (
        <section className="flex flex-col gap-2">
          <SectionLabel>Deployment this model names</SectionLabel>
          <DeploymentCard
            entry={referenced}
            note="Requests to this alias go to the deployment's runtime."
          />
        </section>
      ) : null}
      {naming.length ? (
        <section className="flex flex-col gap-2">
          <SectionLabel>Deployments that name this model · {naming.length}</SectionLabel>
          {naming.map((entry) => (
            <DeploymentCard
              key={entry.name}
              entry={entry}
              note="Without its own pricing, this deployment bills at this model's rates."
            />
          ))}
        </section>
      ) : null}
    </div>
  )
}
