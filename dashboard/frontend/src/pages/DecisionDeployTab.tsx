import { useMemo } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import {
  Activity,
  Cpu,
  Crown,
  Link2,
  MessagesSquare,
  PanelRightOpen,
  RefreshCw,
  X,
} from 'lucide-react'
import ConfigSaveWarnings from '../components/ConfigSaveWarnings'
import { useAuth } from '../contexts/AuthContext'
import { useReadonly } from '../contexts/ReadonlyContext'
import {
  Alert,
  Button,
  DataTable,
  DropdownMenuItem,
  DropdownMenuSeparator,
  EmptyState,
  ProviderLogo,
  StatusLabel,
  cn,
  toast,
  useRetainedValue,
  type Column,
} from '../ui'
import { canWriteConfig } from '../utils/accessControl'
import type { CanonicalGlobalConfig } from './configPageSupport'
import DecisionDefaultDeployment from './DecisionDefaultDeployment'
import DecisionDeployDialog, {
  type DecisionBinding,
  type DecisionDeployTarget,
} from './DecisionDeployDialog'
import DecisionDeploymentSheet from './DecisionDeploymentSheet'
import {
  configuredDecisionDevices,
  decisionConsumerSummary,
  decisionDeploymentRows,
  observedDecisionDevices,
  type DecisionDeploymentRow,
} from './decisionDeploymentRows'
import DecisionModelCatalog from './DecisionModelCatalog'
import { cpuRecommendedDecisionModel } from './decisionModelRecommendation'
import {
  DECISION_CATALOG_MODELS,
  DECISION_MODEL_OPTIONS,
  configuredDecisionDeployment,
} from './decisionModelSupport'
import {
  decisionRuntimeConsumers,
  decisionRuntimeDeclarations,
  withDecisionRuntimeDeployment,
  type DecisionRuntimeDeploymentRequest,
} from './decisionRuntimeDeployment'
import { decisionStageStatus } from './decisionModelManagement'
import DecisionStatusTrack from './DecisionStatusTrack'
import DecisionTaskBindings, { withDecisionTaskBinding } from './DecisionTaskBindings'
import { useDecisionModelManagement } from './useDecisionModelManagement'
import { useDecisionTasks } from './useDecisionTasks'
import { useInstanceDeployment } from './useInstanceDeployment'

const BINDINGS: readonly DecisionBinding[] = ['default', 'signals', 'decisions', 'none']

export default function DecisionDeployTab() {
  const navigate = useNavigate()
  const { user } = useAuth()
  const { isReadonly, isLoading: accessLoading } = useReadonly()
  const instance = useInstanceDeployment()
  const instanceBusy = !instance.stale && instance.busy
  const model = useDecisionModelManagement()
  const tasks = useDecisionTasks()
  const [params, setParams] = useSearchParams()
  const writable = !isReadonly && !accessLoading && canWriteConfig(user)
  const canWrite = writable && !instanceBusy && !model.observationState.config.stale
  const canDeploy = canWrite && Boolean(model.config)

  const effectiveDeployments = (model.global as CanonicalGlobalConfig | undefined)?.model_catalog
    ?.deployments
  const rows = useMemo(
    () =>
      decisionDeploymentRows({
        config: model.config,
        effectiveDeployments,
        inventory: model.inventory,
        activation: model.activation,
        tasks: tasks.data,
      }),
    [model.config, effectiveDeployments, model.inventory, model.activation, tasks.data],
  )
  const defaultRow =
    model.config || tasks.data || model.inventory ? rows.find((row) => row.isDefault) : undefined
  const otherRows = rows.filter((row) => !row.isDefault)
  const consumers = useMemo(() => decisionRuntimeConsumers(model.config), [model.config])
  const takenNames = useMemo(
    () =>
      new Set([
        ...Object.keys(decisionRuntimeDeclarations(model.config)),
        ...rows.map((row) => row.name),
      ]),
    [model.config, rows],
  )
  const recommended = useMemo(
    () =>
      cpuRecommendedDecisionModel(DECISION_CATALOG_MODELS, {
        gpuAvailable: model.status?.models?.system?.gpu_available,
        observedDevices: [
          ...observedDecisionDevices(rows),
          ...(model.status?.models?.models ?? []).map((item) => item.metadata?.device),
        ],
        configuredDevices: configuredDecisionDevices(rows),
      }),
    [model.status, rows],
  )

  const update = (entries: Record<string, string | null>) =>
    setParams(
      (current) => {
        const next = new URLSearchParams(current)
        for (const [key, value] of Object.entries(entries)) {
          if (value === null) next.delete(key)
          else next.set(key, value)
        }
        return next
      },
      { replace: true },
    )
  const bindName = params.get('bind')
  const deployName = params.get('deploy')
  const bindingParam = params.get('binding') as DecisionBinding | null
  const bindRow = bindName
    ? rows.find((row) => row.name === bindName && row.declaration)
    : undefined
  const deployModel = deployName
    ? DECISION_MODEL_OPTIONS.find((option) => option.name === deployName)
    : undefined
  const target: DecisionDeployTarget | null = bindRow
    ? { kind: 'bind', row: bindRow }
    : deployModel
      ? { kind: 'new', model: deployModel }
      : null
  const retainedTarget = useRetainedValue(target)
  const sheetRow = rows.find((row) => row.name === params.get('deployment'))

  const openBind = (row: DecisionDeploymentRow, binding?: DecisionBinding) =>
    update({ bind: row.name, binding: binding ?? null, deploy: null, deployment: null })
  const submitDeployment = async (
    request: DecisionRuntimeDeploymentRequest,
    binding: DecisionBinding,
  ) => {
    await model.updateConfig((current) => withDecisionRuntimeDeployment(current, request))
    toast.success(
      binding === 'none' ? `${request.name} saved` : `${request.name} saved and activated`,
      {
        description:
          binding === 'none'
            ? 'Nothing starts until a consumer uses it.'
            : binding === 'default'
              ? 'Questions without a binding of their own move to it.'
              : 'Only the consumers you picked move to it.',
      },
    )
  }

  const operation = instance.status?.operation
  const observedMode =
    instance.status?.observed_mode === 'unknown'
      ? model.status?.serving_mode
      : (instance.status?.observed_mode ?? model.status?.serving_mode)
  const activeDeployment = instance.status?.active_deployment
  const activeObserved = model.inventory?.deployments.find((item) => item.name === activeDeployment)
  const activeLabel =
    activeObserved?.repo?.split('/').slice(-1)[0] ||
    instance.status?.model ||
    activeObserved?.served_name
  const savedLabel =
    DECISION_MODEL_OPTIONS.find((option) => option.name === model.savedModel)?.label ??
    model.savedModel

  const columns: Column<DecisionDeploymentRow>[] = [
    {
      id: 'deployment',
      header: 'Deployment',
      sortValue: (row) => row.name,
      cell: (row) => (
        <div className="flex min-w-0 items-center gap-3">
          <ProviderLogo provider="vllm" size={16} framed />
          <div className="flex min-w-0 flex-col">
            <span className="truncate font-mono text-[12.5px] font-medium">{row.name}</span>
            <span className="truncate text-xs text-subtle-foreground">
              {row.modelName}
              {row.device ? ` · ${row.device}` : ''}
            </span>
          </div>
        </div>
      ),
    },
    {
      id: 'state',
      header: 'State',
      cell: (row) => {
        const status = decisionStageStatus(row.progress)
        return (
          <div className="flex items-center gap-3">
            <DecisionStatusTrack size="sm" progress={row.progress} />
            <div className="flex min-w-0 flex-col">
              <StatusLabel tone={status.tone} pulse={row.progress.stage === 'activated'}>
                {status.label}
              </StatusLabel>
              {row.progress.pending ? (
                <span className="truncate text-[11.5px] text-subtle-foreground">
                  {row.progress.pending}
                </span>
              ) : null}
            </div>
          </div>
        )
      },
    },
    {
      id: 'serves',
      header: 'Serves',
      hideBelow: 'lg',
      cell: (row) => {
        const summary = decisionConsumerSummary(row)
        return (
          <span className={cn('text-[13px]', summary === 'Not bound' && 'text-subtle-foreground')}>
            {summary}
          </span>
        )
      },
    },
    {
      id: 'replicas',
      header: 'Replicas',
      align: 'right',
      hideBelow: 'md',
      cell: (row) => (
        <span className="font-mono text-xs tabular">
          {row.replicas.ready ?? '–'}/{row.replicas.desired}
        </span>
      ),
    },
    {
      id: 'provider',
      header: 'Runs on',
      hideBelow: 'xl',
      cell: (row) => (
        <span className="text-xs text-muted-foreground">
          {row.kind === 'systemone'
            ? 'System One API'
            : row.kind === 'observed'
              ? 'Observed only'
              : 'Model runtime'}
        </span>
      ),
    },
  ]

  return (
    <div className="flex flex-col gap-8">
      <div className="-mb-4 flex flex-wrap items-center gap-3">
        <span className="text-xs text-muted-foreground" role="status">
          {model.updatedAt
            ? `Observed ${model.updatedAt.toLocaleTimeString()} · runtime state refreshes automatically`
            : 'Loading deployment observations…'}
        </span>
        <Button
          variant="outline"
          size="sm"
          className="ml-auto"
          disabled={model.refreshing || instanceBusy}
          onClick={() => {
            void model.refresh()
            void instance.refresh()
            tasks.refresh()
          }}
        >
          <RefreshCw className={cn(model.refreshing && 'animate-spin')} />
          Refresh
        </Button>
      </div>

      {model.applyResult ? (
        <Alert
          tone={model.applyResult.status === 'success' ? 'success' : 'warning'}
          title={
            model.applyResult.status === 'success'
              ? 'Saved and applied'
              : model.applyResult.status === 'restart_required'
                ? 'Saved · restart required to activate'
                : 'Saved · rollout required to activate'
          }
          actions={
            <Button
              variant="ghost"
              size="icon-xs"
              aria-label="Dismiss"
              onClick={model.clearApplyResult}
            >
              <X />
            </Button>
          }
        >
          {model.applyResult.message}
          <ConfigSaveWarnings warnings={model.applyResult.warnings} />
        </Alert>
      ) : null}
      {model.errors.length > 0 ? (
        <Alert tone="warning" title="Some deployment observations are unavailable">
          Readiness and activation show as unknown until the Router reports them; the saved
          configuration stays editable. Details appear next to each affected state.
          {model.errors.some((error) => /^(status|config):/.test(error)) ? (
            <ul className="mt-1.5 flex flex-col gap-0.5 font-mono text-[11.5px]">
              {model.errors
                .filter((error) => /^(status|config):/.test(error))
                .map((error) => (
                  <li key={error} className="truncate" title={error}>
                    {error}
                  </li>
                ))}
            </ul>
          ) : null}
        </Alert>
      ) : null}

      <DecisionDefaultDeployment
        row={defaultRow}
        loading={model.observationState.inventory.loading}
        mode={`${instance.stale ? 'Last observed: ' : ''}${
          observedMode === 'router'
            ? 'Router mode'
            : observedMode === 'engine'
              ? 'Engine mode'
              : 'Mode unavailable'
        }`}
        phase={
          instance.loading
            ? 'Checking instance…'
            : `${instance.stale ? 'Last observed: ' : ''}${(
                operation?.phase ??
                model.status?.router_runtime?.phase ??
                'Not reported'
              ).replace(/_/g, ' ')}`
        }
        servingNote={
          activeDeployment && defaultRow && activeDeployment !== defaultRow.name
            ? `${instance.stale ? 'Last observed serving' : 'Still serving'} ${activeLabel || activeDeployment} on ${activeDeployment} until the saved default is active.`
            : undefined
        }
        busy={instanceBusy}
        operationError={
          operation?.error
            ? `${operation.error}${operation.rolled_back ? ' The previous deployment was restored.' : ''}`
            : undefined
        }
        observations={[
          { label: 'Runtime', observation: model.observationState.inventory },
          { label: 'Instance', observation: instance },
          { label: 'Activation', observation: model.observationState.activation },
          { label: 'Model configuration', observation: model.observationState.global },
        ]}
        diagnostics={{
          savedModel: savedLabel || 'Not reported',
          defaultDeployment: model.config
            ? configuredDecisionDeployment(model.config) || 'Not reported'
            : 'Loading',
          ownership: instance.status?.ownership,
          activation: model.activation ?? null,
          messages: [
            instance.status?.unavailable_reason,
            instance.error,
            model.status?.router_runtime?.message,
            ...(model.activation?.activation?.reasons ?? []).map(
              (reason) => reason.message || reason.code,
            ),
          ].filter((value): value is string => Boolean(value)),
        }}
        writable={canWrite && Boolean(defaultRow?.declaration)}
        onBind={() => defaultRow && openBind(defaultRow)}
        onDetails={() => defaultRow && update({ deployment: defaultRow.name })}
      />

      <section className="flex flex-col gap-3" aria-labelledby="decision-deployments-title">
        <div className="flex flex-col gap-0.5">
          <h2
            id="decision-deployments-title"
            className="text-[15px] font-semibold tracking-[-0.012em]"
          >
            Other deployments
          </h2>
          <p className="text-[13px] text-muted-foreground">
            Bound to specific signals, decisions or tasks. Click one to see its state, replicas and
            what it serves.
          </p>
        </div>
        <DataTable
          data={otherRows}
          columns={columns}
          loading={!model.config && model.observationState.config.loading}
          getRowId={(row) => row.name}
          activeRowId={sheetRow?.name}
          onRowClick={(row) => update({ deployment: row.name })}
          rowActions={(row) => (
            <>
              <DropdownMenuItem onSelect={() => update({ deployment: row.name })}>
                <PanelRightOpen />
                Details
              </DropdownMenuItem>
              <DropdownMenuItem
                disabled={!canWrite || !row.declaration}
                onSelect={() => openBind(row)}
              >
                <Link2 />
                {row.consumers.length ? 'Bind more consumers' : 'Bind consumers'}
              </DropdownMenuItem>
              <DropdownMenuItem
                disabled={!canWrite || !row.declaration}
                onSelect={() => openBind(row, 'default')}
              >
                <Crown />
                Make default
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem
                disabled={row.progress.stage !== 'ready'}
                onSelect={() =>
                  navigate(
                    `/models/decision?tab=playground&target=${encodeURIComponent(`deployment:${row.name}`)}`,
                  )
                }
              >
                <MessagesSquare />
                Try in playground
              </DropdownMenuItem>
              <DropdownMenuItem
                onSelect={() =>
                  navigate(
                    `/models/decision?tab=monitoring&deployment=${encodeURIComponent(row.name)}`,
                  )
                }
              >
                <Activity />
                Monitoring
              </DropdownMenuItem>
            </>
          )}
          empty={
            <EmptyState
              compact
              icon={Cpu}
              tone="violet"
              title="Only the default deployment"
              description="Deploy a model from the catalog to give some signals or decisions their own decision model."
            />
          }
        />
      </section>

      <DecisionTaskBindings
        data={tasks.data}
        error={tasks.error}
        observation={tasks}
        writable={canWrite && !model.deploying && !tasks.stale}
        save={(binding, deployment) =>
          model.updateConfig((current) => withDecisionTaskBinding(current, binding, deployment))
        }
      />

      <DecisionModelCatalog
        rows={rows}
        recommended={recommended}
        defaultArtifact={defaultRow?.artifact}
        writable={canDeploy}
        onDeploy={(option) => update({ deploy: option.name, bind: null, binding: null })}
        onOpenDeployment={(name) => update({ deployment: name })}
      />

      <DecisionDeployDialog
        open={Boolean(target) && (Boolean(model.config) || !model.observationState.config.loading)}
        onOpenChange={(open) =>
          open ? undefined : update({ deploy: null, bind: null, binding: null })
        }
        target={retainedTarget}
        initialBinding={bindingParam && BINDINGS.includes(bindingParam) ? bindingParam : undefined}
        consumers={consumers}
        takenNames={takenNames}
        defaultDeployment={defaultRow?.name ?? 'primary'}
        writable={canDeploy}
        onSubmit={submitDeployment}
      />
      <DecisionDeploymentSheet
        row={sheetRow}
        model={model}
        writable={canWrite}
        onOpenChange={(open) => (open ? undefined : update({ deployment: null }))}
        onBind={(row) => openBind(row)}
        onMakeDefault={(row) => openBind(row, 'default')}
      />
    </div>
  )
}
