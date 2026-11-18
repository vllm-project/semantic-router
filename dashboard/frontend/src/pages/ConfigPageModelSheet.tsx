import type { ReactNode } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Activity,
  ChartNoAxesColumn,
  Copy,
  MoreHorizontal,
  Pencil,
  RefreshCw,
  Send,
  Split,
  Star,
  Trash2,
} from 'lucide-react'

import { configCollectionItemYaml } from '../lib/configYaml'
import type { BuiltInModelCatalog } from '../types/modelCatalog'
import {
  Alert,
  Badge,
  Button,
  DetailSheet,
  DetailSheetBody,
  DetailSheetFooter,
  DetailSheetHeader,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
  EmptyState,
  IconTile,
  KeyValueGrid,
  MiniStat,
  SectionLabel,
  Skeleton,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
  toast,
  useCopy,
} from '../ui'
import type { ModelHealth } from '../utils/modelHealthApi'
import ConfigPageEvaluationDetails from './ConfigPageEvaluationDetails'
import { ModelHealthSummary } from './ConfigPageModelHealth'
import { MODEL_HEALTH_META, modelHealthStatus } from './configPageModelHealthPresentation'
import {
  ModelDeploymentsPanel,
  ModelEndpointsPanel,
  ModelPricingPanel,
  ModelReasoningPanel,
} from './ConfigPageModelSheetPanels'
import type { EffectiveEvaluationGroup } from './configPageEffectiveEvaluations'
import { CatalogProviderLogo, ModelLogo } from './ConfigPageModelLogo'
import {
  modalityLabel,
  modelPriceSummary,
  modelProviderIds,
  providerDisplayName,
  type ModelDecisionReference,
  type NamedDeployment,
} from './configPageModelPresentation'
import { EntityYamlPanel } from './configPageRoutingScopeViews'
import type { ConfigData, NormalizedModel, ReasoningFamily } from './configPageSupport'
import type { BackendModelHealth } from './useBackendModelHealth'
import type { ModelLiveVerificationState } from './useModelLiveVerification'

export const MODEL_SHEET_VIEWS = [
  'overview',
  'usage',
  'endpoints',
  'pricing',
  'reasoning',
  'deployments',
  'evidence',
  'yaml',
] as const
export type ModelSheetView = (typeof MODEL_SHEET_VIEWS)[number]

interface ConfigPageModelSheetProps {
  model: NormalizedModel | null
  open: boolean
  view: ModelSheetView
  onViewChange: (view: ModelSheetView) => void
  onOpenChange: (open: boolean) => void
  config: ConfigData | null
  catalog: BuiltInModelCatalog
  defaultModel: string
  isReadonly: boolean
  canVerifyModels: boolean
  decisionReferences: ModelDecisionReference[]
  deployments: { referenced: NamedDeployment | null; naming: NamedDeployment[] }
  evaluationGroup?: EffectiveEvaluationGroup
  evaluationsReady: boolean
  evaluationsLoading: boolean
  onRetryEvaluations: () => void
  /** The server catalog failed and evidence comes from the bundled copy. */
  catalogFallback?: boolean
  reasoningFamilies: Record<string, ReasoningFamily>
  health?: ModelHealth
  healthPhase: BackendModelHealth['phase']
  healthChecking: boolean
  healthIntervalSeconds: number
  onCheckHealth: () => void
  /** Percent of the last 24 h of requests; null without Router Replay data. */
  trafficShare: number | null
  liveState: ModelLiveVerificationState
  onVerify: () => void
  canBeDefault: boolean
  deleteBlocker: string | null
  onEdit: () => void
  onDelete: () => void
  onMakeDefault: () => void
  onManageEvaluations: () => void
}

/** The configuration entry of a model as config.yaml states it: binding and facts together. */
function canonicalModelEntry(config: ConfigData | null, model: NormalizedModel) {
  const entry = config?.providers?.models?.find((candidate) => candidate.name === model.name)
  const card = config?.routing?.modelCards?.find(
    (candidate) => candidate.name === (model.catalog || model.name),
  )
  const merged: Record<string, unknown> = { ...(entry ?? { name: model.name }) }
  for (const [key, value] of Object.entries(card ?? {})) {
    if (key !== 'name' && merged[key] === undefined) merged[key] = value
  }
  return merged
}

const healthBadgeTone = {
  healthy: 'success',
  degraded: 'warning',
  down: 'danger',
  unchecked: 'neutral',
} as const

function LiveEvidence({ state }: { state: ModelLiveVerificationState }) {
  if (state.status === 'idle') return null
  if (state.status === 'pending') {
    return (
      <div className="flex flex-col gap-2 rounded-lg border border-border px-3.5 py-3">
        <Skeleton className="h-3.5 w-40" />
        <Skeleton className="h-3 w-64" />
      </div>
    )
  }
  if (state.status === 'failed') {
    return (
      <div role="alert">
        <Alert tone="danger" title="The test request failed">
          {state.message}
        </Alert>
      </div>
    )
  }
  const { evidence } = state
  return (
    <Alert tone="success" title="Answered a test request">
      <span className="flex flex-col gap-0.5">
        <span>
          <span className="font-mono">{evidence.providerModel}</span> through{' '}
          <span className="font-mono">{evidence.backend}</span>
          {evidence.latencyMs !== undefined ? ` in ${evidence.latencyMs} ms` : ''}.
        </span>
        {evidence.verifiedAt ? (
          <span className="text-xs">Checked {new Date(evidence.verifiedAt).toLocaleString()}</span>
        ) : null}
      </span>
    </Alert>
  )
}

function OverviewPanel({
  model,
  catalog,
  health,
  healthPhase,
  healthChecking,
  trafficShare,
  liveState,
}: Pick<
  ConfigPageModelSheetProps,
  'catalog' | 'health' | 'healthPhase' | 'healthChecking' | 'trafficShare' | 'liveState'
> & { model: NormalizedModel }) {
  const price = modelPriceSummary(model.pricing)
  const providers = modelProviderIds(model)
  const card = model.catalog
    ? catalog.models.find((entry) => entry.id === model.catalog)
    : undefined
  const identity: Array<{ label: string; value: ReactNode; mono?: boolean; span?: boolean }> = [
    {
      label: 'Provider',
      value: providers.length ? (
        <span className="flex flex-wrap items-center gap-x-3 gap-y-1">
          {providers.map((id) => (
            <span key={id} className="inline-flex items-center gap-1.5">
              <CatalogProviderLogo provider={id} size={14} catalog={catalog} />
              {providerDisplayName(id, catalog)}
            </span>
          ))}
        </span>
      ) : (
        <span className="text-muted-foreground">None configured</span>
      ),
    },
    {
      label: 'API format',
      value: (
        <span className="inline-flex items-center gap-1.5">
          <span className="font-mono text-[12.5px]">
            {model.api_format || (model.backend_refs?.length ? 'Unresolved' : 'Not connected')}
          </span>
          {model.api_format_override ? (
            <Badge size="sm" tone="outline">
              override
            </Badge>
          ) : null}
        </span>
      ),
    },
    {
      label: 'Catalog model',
      value: card ? (
        <span className="flex flex-col">
          <span>{card.display_name}</span>
          <span className="font-mono text-[11.5px] text-subtle-foreground">{card.id}</span>
        </span>
      ) : (
        model.catalog || 'Custom model'
      ),
    },
    {
      label: 'Provider model ID',
      value: model.provider_model_id || '—',
      mono: Boolean(model.provider_model_id),
    },
    { label: 'Parameter size', value: model.param_size || '—' },
    { label: 'Modalities', value: modalityLabel(model.modalities) },
  ]
  if (model.description)
    identity.push({ label: 'Description', value: model.description, span: true })
  const metadata = [
    model.capabilities?.length ? { label: 'Capabilities', values: model.capabilities } : null,
    model.tags?.length ? { label: 'Tags', values: model.tags } : null,
    model.loras?.length
      ? { label: 'LoRA adapters', values: model.loras.map((adapter) => adapter.name) }
      : null,
  ].filter((item): item is { label: string; values: string[] } => item !== null)

  return (
    <div className="flex flex-col gap-5">
      <div className="grid grid-cols-3 gap-3">
        <MiniStat
          label="Share of traffic"
          value={
            trafficShare === null ? (
              <span className="text-base text-subtle-foreground">—</span>
            ) : (
              `${trafficShare.toFixed(1)}%`
            )
          }
          hint={trafficShare === null ? 'needs Router Replay' : 'last 24 h'}
        />
        <MiniStat
          label="Price in · out"
          value={
            price.input || price.output ? (
              <span className="font-mono text-base">
                {price.input ?? '—'} · {price.output ?? '—'}
              </span>
            ) : (
              <span className="text-base text-subtle-foreground">Not priced</span>
            )
          }
          hint="per 1M tokens"
        />
        <MiniStat
          label="Context window"
          value={
            model.context_window_size ? (
              model.context_window_size.toLocaleString()
            ) : (
              <span className="text-base text-subtle-foreground">—</span>
            )
          }
          hint={model.context_window_size ? 'tokens' : 'not declared'}
        />
      </div>
      <ModelHealthSummary health={health} phase={healthPhase} checking={healthChecking} />
      <LiveEvidence state={liveState} />
      <KeyValueGrid items={identity} />
      {metadata.length ? (
        <section className="flex flex-col gap-3">
          {metadata.map((item) => (
            <div key={item.label} className="flex flex-col gap-1.5">
              <SectionLabel>{item.label}</SectionLabel>
              <span className="flex flex-wrap gap-1.5">
                {item.values.map((value) => (
                  <Badge key={value} tone="outline" className="font-mono">
                    {value}
                  </Badge>
                ))}
              </span>
            </div>
          ))}
        </section>
      ) : null}
    </div>
  )
}

function UsagePanel({
  references,
  isReadonly,
  onOpenDecision,
}: {
  references: ModelDecisionReference[]
  isReadonly: boolean
  onOpenDecision: (reference: ModelDecisionReference) => void
}) {
  if (references.length === 0) {
    return (
      <div className="rounded-lg border border-dashed border-border-strong">
        <EmptyState
          compact
          icon={Split}
          tone="violet"
          title="Not used yet"
          description={
            isReadonly
              ? 'No decision lists this model as a candidate.'
              : 'No decision lists this model as a candidate. Add it to a decision to start sending it traffic.'
          }
        />
      </div>
    )
  }
  return (
    <div className="flex flex-col gap-2">
      <SectionLabel className="mb-1">Decisions that can select this model</SectionLabel>
      {references.map((reference) => (
        <button
          key={`${reference.scope}/${reference.decision.name}`}
          type="button"
          onClick={() => onOpenDecision(reference)}
          className="flex items-center gap-3 rounded-lg border border-border bg-surface px-3 py-2.5 text-left transition-colors hover:border-border-strong hover:bg-surface-subtle"
        >
          <IconTile icon={Split} tone="violet" size="sm" />
          <span className="flex min-w-0 flex-1 flex-col">
            <span className="truncate font-mono text-[12.5px] font-medium">
              {reference.decision.name}
            </span>
            <span className="truncate text-xs text-subtle-foreground">{reference.scopeLabel}</span>
          </span>
          <span className="font-mono text-xs tabular text-subtle-foreground">
            P{reference.decision.priority}
          </span>
        </button>
      ))}
    </div>
  )
}

function EvidencePanel({
  group,
  ready,
  loading,
  fallback,
  isReadonly,
  onRetry,
  onManage,
}: {
  group?: EffectiveEvaluationGroup
  ready: boolean
  loading: boolean
  fallback: boolean
  isReadonly: boolean
  onRetry: () => void
  onManage: () => void
}) {
  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <p className="max-w-md text-xs leading-relaxed text-muted-foreground">
          Published results from the model catalog and the records you configure, as their sources
          report them. Configured records add to built-in results; they never replace them.
        </p>
        {isReadonly ? null : (
          <Button variant="outline" size="sm" onClick={onManage}>
            <Pencil />
            Manage records
          </Button>
        )}
      </div>
      {ready && fallback ? (
        <Alert
          tone="neutral"
          title="Using the bundled catalog"
          actions={
            <Button variant="outline" size="sm" onClick={onRetry}>
              <RefreshCw />
              Retry
            </Button>
          }
        >
          The live catalog API is unavailable, so built-in results come from the catalog bundled
          with the Dashboard.
        </Alert>
      ) : null}
      {!ready ? (
        loading ? (
          <div className="flex flex-col gap-3" aria-label="Loading model evaluations">
            <Skeleton className="h-28 w-full" />
            <Skeleton className="h-28 w-full" />
          </div>
        ) : (
          <Alert
            tone="warning"
            title="Model evaluations could not be loaded"
            actions={
              <Button variant="outline" size="sm" onClick={onRetry}>
                <RefreshCw />
                Retry
              </Button>
            }
          >
            Configured records still apply; built-in catalog results appear once the catalog loads.
          </Alert>
        )
      ) : group ? (
        <ConfigPageEvaluationDetails key={group.modelName} group={group} />
      ) : null}
    </div>
  )
}

export default function ConfigPageModelSheet(props: ConfigPageModelSheetProps) {
  const {
    model,
    open,
    view,
    onViewChange,
    onOpenChange,
    config,
    catalog,
    defaultModel,
    isReadonly,
    canVerifyModels,
    decisionReferences,
    deployments,
    evaluationGroup,
    health,
    healthPhase,
    healthChecking,
    onCheckHealth,
    liveState,
    onVerify,
    canBeDefault,
    deleteBlocker,
    onEdit,
    onDelete,
    onMakeDefault,
  } = props
  const navigate = useNavigate()
  const { copy } = useCopy()
  if (!model) return null
  const isDefault = model.name === defaultModel
  const backends = model.backend_refs?.length ?? 0
  const hasEndpoint = model.endpoints.some((endpoint) => endpoint.endpoint.trim())
  const deploymentCount = (deployments.referenced ? 1 : 0) + deployments.naming.length
  const evidenceCount = evaluationGroup?.records.length ?? 0
  const yaml = configCollectionItemYaml(['providers', 'models'], canonicalModelEntry(config, model))

  return (
    <DetailSheet open={open} onOpenChange={onOpenChange} size="xl">
      <DetailSheetHeader
        icon={<ModelLogo model={model} size={24} framed catalog={catalog} />}
        title={<span className="font-mono">{model.name}</span>}
        badges={
          <>
            {isDefault ? (
              <Badge tone="brand">
                <Star />
                Default
              </Badge>
            ) : null}
            {healthPhase === 'unavailable' && !health ? null : (
              <Badge tone={healthBadgeTone[modelHealthStatus(health)]} dot>
                {MODEL_HEALTH_META[modelHealthStatus(health)].label}
              </Badge>
            )}
          </>
        }
        description={
          <>
            {model.provider_model_id ? (
              <span className="font-mono">{model.provider_model_id}</span>
            ) : null}
            {model.provider_model_id ? <span className="text-subtle-foreground">·</span> : null}
            <span>{model.api_format || 'API not resolved'}</span>
            <span className="text-subtle-foreground">·</span>
            <span>
              {deployments.referenced
                ? `deployment ${deployments.referenced.name}`
                : `${backends} ${backends === 1 ? 'backend' : 'backends'}`}
            </span>
          </>
        }
        actions={
          <>
            {healthPhase === 'unavailable' ? null : (
              <Button
                variant="outline"
                size="sm"
                onClick={onCheckHealth}
                disabled={healthChecking}
                loading={healthChecking}
              >
                {healthChecking ? null : <Activity />}
                {healthChecking ? 'Checking…' : 'Check health'}
              </Button>
            )}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="icon-sm" aria-label="Model actions">
                  <MoreHorizontal />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-52">
                <DropdownMenuItem
                  onSelect={() => {
                    copy(model.name)
                    toast.success('Model name copied')
                  }}
                >
                  <Copy />
                  Copy name
                </DropdownMenuItem>
                {canVerifyModels ? (
                  <DropdownMenuItem
                    disabled={!hasEndpoint || liveState.status === 'pending'}
                    onSelect={() => {
                      onViewChange('overview')
                      onVerify()
                    }}
                  >
                    <Send />
                    Send a test request
                  </DropdownMenuItem>
                ) : null}
                <DropdownMenuItem
                  onSelect={() =>
                    navigate(`/observe/evaluations?model=${encodeURIComponent(model.name)}`)
                  }
                >
                  <ChartNoAxesColumn />
                  Evaluate
                </DropdownMenuItem>
                {isReadonly ? null : (
                  <>
                    <DropdownMenuItem disabled={!canBeDefault} onSelect={onMakeDefault}>
                      <Star />
                      Make default
                    </DropdownMenuItem>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem variant="destructive" onSelect={onDelete}>
                      <Trash2 />
                      Delete
                    </DropdownMenuItem>
                  </>
                )}
              </DropdownMenuContent>
            </DropdownMenu>
          </>
        }
      />
      <Tabs
        value={view}
        onValueChange={(next) => onViewChange(next as ModelSheetView)}
        className="min-h-0 flex-1 gap-0"
      >
        <TabsList className="overflow-x-auto px-5 [scrollbar-width:none]">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="usage">
            Used by
            <Badge size="sm" tone="neutral" className="tabular">
              {decisionReferences.length}
            </Badge>
          </TabsTrigger>
          <TabsTrigger value="endpoints">
            Endpoints
            <Badge size="sm" tone="neutral" className="tabular">
              {backends}
            </Badge>
          </TabsTrigger>
          <TabsTrigger value="pricing">Pricing</TabsTrigger>
          <TabsTrigger value="reasoning">Reasoning</TabsTrigger>
          {deploymentCount || view === 'deployments' ? (
            <TabsTrigger value="deployments">
              Deployments
              <Badge size="sm" tone="neutral" className="tabular">
                {deploymentCount}
              </Badge>
            </TabsTrigger>
          ) : null}
          <TabsTrigger value="evidence">
            Evidence
            {props.evaluationsReady ? (
              <Badge size="sm" tone="neutral" className="tabular">
                {evidenceCount}
              </Badge>
            ) : null}
          </TabsTrigger>
          <TabsTrigger value="yaml">YAML</TabsTrigger>
        </TabsList>
        <DetailSheetBody>
          <TabsContent value="overview" className="px-5 py-5">
            <OverviewPanel
              model={model}
              catalog={catalog}
              health={health}
              healthPhase={healthPhase}
              healthChecking={healthChecking}
              trafficShare={props.trafficShare}
              liveState={liveState}
            />
          </TabsContent>
          <TabsContent value="usage" className="px-5 py-5">
            <UsagePanel
              references={decisionReferences}
              isReadonly={isReadonly}
              onOpenDecision={(reference) =>
                navigate(
                  `/routing/decisions?recipe=${encodeURIComponent(reference.scope)}&id=${encodeURIComponent(reference.decision.name)}`,
                )
              }
            />
          </TabsContent>
          <TabsContent value="endpoints" className="px-5 py-5">
            <ModelEndpointsPanel
              model={model}
              deploymentName={deployments.referenced?.name ?? ''}
              masked={isReadonly}
              readOnly={isReadonly}
              catalog={catalog}
              health={health}
              onEdit={onEdit}
              onShowDeployment={() => onViewChange('deployments')}
            />
          </TabsContent>
          <TabsContent value="pricing" className="px-5 py-5">
            <ModelPricingPanel model={model} readOnly={isReadonly} onEdit={onEdit} />
          </TabsContent>
          <TabsContent value="reasoning" className="px-5 py-5">
            <ModelReasoningPanel
              model={model}
              families={props.reasoningFamilies}
              catalog={catalog}
              readOnly={isReadonly}
              onEdit={onEdit}
            />
          </TabsContent>
          <TabsContent value="deployments" className="px-5 py-5">
            <ModelDeploymentsPanel
              referenced={deployments.referenced}
              naming={deployments.naming}
            />
          </TabsContent>
          <TabsContent value="evidence" className="px-5 py-5">
            <EvidencePanel
              group={evaluationGroup}
              ready={props.evaluationsReady}
              loading={props.evaluationsLoading}
              fallback={Boolean(props.catalogFallback)}
              isReadonly={isReadonly}
              onRetry={props.onRetryEvaluations}
              onManage={props.onManageEvaluations}
            />
          </TabsContent>
          <TabsContent value="yaml" className="px-5 py-5">
            <EntityYamlPanel
              path={`providers.models[${model.name}]`}
              yaml={yaml}
              note="The model's binding and facts as config.yaml states them."
            />
          </TabsContent>
        </DetailSheetBody>
      </Tabs>
      <DetailSheetFooter>
        <span className="min-w-0 truncate text-xs text-muted-foreground">
          {isReadonly
            ? 'Read-only: your account can view this configuration.'
            : healthPhase !== 'unavailable'
              ? `Health checks run every ${props.healthIntervalSeconds} s`
              : deleteBlocker || 'The Router validates every change when you save.'}
        </span>
        <div className="ml-auto flex items-center gap-2">
          <Button variant="ghost" size="sm" onClick={() => onOpenChange(false)}>
            Close
          </Button>
          {isReadonly ? null : (
            <Button size="sm" onClick={onEdit}>
              <Pencil />
              Edit model
            </Button>
          )}
        </div>
      </DetailSheetFooter>
    </DetailSheet>
  )
}
