import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  Activity,
  Boxes,
  Brain,
  ChartNoAxesColumn,
  CircleDollarSign,
  Gauge,
  LibraryBig,
  MoreHorizontal,
  Plus,
  SlidersHorizontal,
  Trash2,
} from 'lucide-react'

import ConfirmDialog from '../components/ConfirmDialog'
import {
  Alert,
  BulkBar,
  Button,
  Card,
  DataToolbar,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
  FilterPill,
  IconTile,
  Skeleton,
  Tooltip,
  toast,
  useRetainedValue,
  type Tone,
} from '../ui'
import { formatInsightsCost } from '../utils/insightsCost'
import ConfigPageConnectModelsDialog from './ConfigPageConnectModelsDialog'
import ConfigPageManagerLayout from './ConfigPageManagerLayout'
import ConfigPageModelSheet, {
  MODEL_SHEET_VIEWS,
  type ModelSheetView,
} from './ConfigPageModelSheet'
import ConfigPageModelsTable from './ConfigPageModelsTable'
import ConfigPageReasoningFamilies from './ConfigPageReasoningFamilies'
import { collectDecisionTargetModels } from './configPageEntrypointsRecipesSupport'
import {
  checkedAgo,
  formatLatency,
  MODEL_HEALTH_META,
  median,
  modelHealthHint,
  modelHealthTotals,
  visibleP95,
} from './configPageModelHealthPresentation'
import { modelBackendKind } from './configPageModelInventory'
import { CatalogProviderLogo } from './ConfigPageModelLogo'
import {
  modelDecisionReferences,
  modelDeployments,
  modelProviderIds,
} from './configPageModelPresentation'
import type { ConfigPageModelsSectionProps } from './configPageModelsSectionTypes'
import type { NormalizedModel } from './configPageSupport'
import ModelDeleteDialog from './ModelDeleteDialog'
import type { BackendTrafficUnavailable } from './useBackendModelTraffic'
import { useConfigPageModelsSectionController } from './useConfigPageModelsSectionController'
import { modelLiveVerificationState } from './useModelLiveVerification'

const ease = [0.16, 1, 0.3, 1] as const

const TRAFFIC_UNAVAILABLE: Record<BackendTrafficUnavailable, string> = {
  replay_off: 'Router Replay is off',
  forbidden: 'No access to Router Replay',
  router_unreachable: 'Router not reachable',
}

function InventoryStat({
  icon,
  tone,
  label,
  value,
  hint,
  index,
  loading = false,
}: {
  icon: typeof Boxes
  tone: Tone
  label: string
  /** null shows "—": the stat has no data source right now. */
  value: ReactNode | null
  hint: ReactNode
  index: number
  loading?: boolean
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.04, duration: 0.4, ease }}
    >
      <Card className="h-full flex-row items-center gap-3 px-4 py-3.5">
        <IconTile icon={icon} tone={tone} size="md" />
        <div className="flex min-w-0 flex-col">
          <span className="text-xs text-muted-foreground">{label}</span>
          {loading ? (
            <Skeleton className="my-1 h-5 w-16" />
          ) : (
            <span className="truncate text-lg font-semibold tabular tracking-[-0.02em]">
              {value ?? <span className="text-subtle-foreground">—</span>}
            </span>
          )}
          <span className="truncate text-[11px] text-subtle-foreground">{hint}</span>
        </div>
      </Card>
    </motion.div>
  )
}

export default function ConfigPageModelsSection(props: ConfigPageModelsSectionProps) {
  const controller = useConfigPageModelsSectionController(props)
  const { filters, deletion, defaults, evaluations, liveVerification, catalog, health, traffic } =
    controller
  const navigate = useNavigate()
  const [params, setParams] = useSearchParams()
  const update = useCallback(
    (patch: Record<string, string | null>, options?: { replace?: boolean }) =>
      setParams((current) => {
        const next = new URLSearchParams(current)
        for (const [key, value] of Object.entries(patch)) {
          if (value === null) next.delete(key)
          else next.set(key, value)
        }
        return next
      }, options),
    [setParams],
  )

  const activeName = params.get('m')
  const active = props.models.find((model) => model.name === activeName) ?? null
  const shown = useRetainedValue(active)
  const viewParam = params.get('view') as ModelSheetView | null
  const view: ModelSheetView =
    viewParam && MODEL_SHEET_VIEWS.includes(viewParam) ? viewParam : 'overview'
  const familiesOpen =
    params.get('families') === '1' || params.has('family') || params.get('tab') === 'families'

  // A link's ?new=1 or ?add=<catalog model> opens the connect dialog and is consumed: the dialog
  // lives in state, so a reload after saving does not open it again.
  const [connect, setConnect] = useState<{ catalogModel?: string } | null>(null)
  const linkedConnect = !props.isReadonly && (params.get('new') === '1' || params.has('add'))
  const linkedModel = params.get('add') ?? undefined
  useEffect(() => {
    if (!linkedConnect) return
    setConnect({ catalogModel: linkedModel })
    update({ new: null, add: null }, { replace: true })
  }, [linkedConnect, linkedModel, update])
  const modelsPendingDelete = deletion.modelNames

  const openModel = useCallback(
    (model: NormalizedModel, nextView: ModelSheetView = 'overview') =>
      update({
        m: model.name,
        view: nextView === 'overview' ? null : nextView,
        tab: null,
        families: null,
        family: null,
      }),
    [update],
  )
  const showUsage = useCallback((model: NormalizedModel) => openModel(model, 'usage'), [openModel])
  const closeModel = () => update({ m: null, view: null })
  const openConnect = () => setConnect({})
  const openFamilies = () => update({ families: '1', family: null, tab: null, m: null, view: null })
  const closeFamilies = () => update({ families: null, family: null, tab: null })

  const checkModelHealth = useCallback(
    async (model: NormalizedModel) => {
      try {
        const result = await health.checkModel(model.name)
        if (!result) return
        const meta = MODEL_HEALTH_META[result.status]
        const p95 = visibleP95(result)
        const description =
          result.reason ?? (p95 !== undefined ? `p95 ${formatLatency(p95)}` : undefined)
        if (result.status === 'healthy') toast.success(`${model.name} is healthy`, { description })
        else if (result.status === 'unchecked')
          toast(`${model.name} was not checked`, { description })
        else toast.error(`${model.name} is ${meta.label.toLowerCase()}`, { description })
      } catch (error) {
        toast.error(`${model.name} could not be checked`, {
          description: error instanceof Error ? error.message : undefined,
        })
      }
    },
    [health],
  )
  const checkAllHealth = async () => {
    try {
      const snapshot = await health.checkAll()
      const totals = modelHealthTotals(
        snapshot.models.map((entry) => entry.model),
        new Map(snapshot.models.map((entry) => [entry.model, entry])),
      )
      const checked = snapshot.models.length
      toast.success(`Checked ${checked} ${checked === 1 ? 'model' : 'models'}`, {
        description:
          totals.healthy === checked
            ? modelHealthHint(totals)
            : `${totals.healthy} healthy · ${modelHealthHint(totals)}`,
      })
    } catch (error) {
      toast.error('Health could not be checked', {
        description: error instanceof Error ? error.message : undefined,
      })
    }
  }

  const connected = useMemo(
    () => props.models.filter((model) => modelBackendKind(model, filters.deployments) !== 'none'),
    [filters.deployments, props.models],
  )
  const providers = useMemo(
    () => new Set(props.models.flatMap(modelProviderIds)).size,
    [props.models],
  )
  const totals = modelHealthTotals(
    connected.map((model) => model.name),
    health.byModel,
  )
  const medianP95 = median(
    connected
      .map((model) => health.byModel.get(model.name))
      .filter((entry) => entry?.status === 'healthy')
      .map((entry) => visibleP95(entry))
      .filter((value): value is number => value !== undefined),
  )
  const unconnected = props.models.length - connected.length
  const healthAvailable = health.phase !== 'unavailable'
  const lastChecked = checkedAgo(health.checkedAt)

  const decisionReferences = useMemo(
    () =>
      shown ? modelDecisionReferences(props.config, shown.name, collectDecisionTargetModels) : [],
    [props.config, shown],
  )
  const deployments = useMemo(
    () => (shown ? modelDeployments(props.config, shown.name) : { referenced: null, naming: [] }),
    [props.config, shown],
  )
  const evaluationGroup = shown
    ? evaluations.groups.find((group) => group.modelName === shown.name)
    : undefined
  const trafficShare =
    shown && traffic.phase === 'ready' && traffic.requests > 0
      ? (traffic.shares.get(shown.name) ?? (traffic.complete ? 0 : null))
      : null

  return (
    <>
      <ConfigPageManagerLayout
        title="Backend models"
        description={
          healthAvailable
            ? `Model endpoints the router can send traffic to. Health is checked every ${health.intervalSeconds} seconds.`
            : 'Model endpoints the router can send traffic to.'
        }
        icon={<IconTile icon={Boxes} tone="green" size="lg" />}
        actions={
          <>
            <Button variant="outline" onClick={() => navigate('/models/hub')}>
              <LibraryBig />
              Browse Model Hub
            </Button>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" size="icon" aria-label="More model actions">
                  <MoreHorizontal />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-60">
                <DropdownMenuItem onSelect={openFamilies}>
                  <Brain />
                  Reasoning families
                </DropdownMenuItem>
                {props.isReadonly ? null : (
                  <>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem onSelect={() => controller.forms.add()}>
                      <SlidersHorizontal />
                      Add a model manually
                    </DropdownMenuItem>
                    <DropdownMenuItem onSelect={() => evaluations.manage()}>
                      <ChartNoAxesColumn />
                      Manage evaluation records
                    </DropdownMenuItem>
                  </>
                )}
              </DropdownMenuContent>
            </DropdownMenu>
            {props.isReadonly ? null : (
              <Button onClick={openConnect}>
                <Plus />
                Connect model
              </Button>
            )}
          </>
        }
      >
        <div className="mb-5 grid grid-cols-2 gap-3 xl:grid-cols-4">
          <InventoryStat
            index={0}
            icon={Boxes}
            tone="green"
            label="Connected"
            value={connected.length}
            hint={[
              `${providers} ${providers === 1 ? 'provider' : 'providers'}`,
              unconnected ? `${unconnected} without a backend` : '',
            ]
              .filter(Boolean)
              .join(' · ')}
          />
          <InventoryStat
            index={1}
            icon={Activity}
            tone="green"
            label="Healthy"
            loading={health.phase === 'loading'}
            value={
              healthAvailable && connected.length ? `${totals.healthy}/${connected.length}` : null
            }
            hint={
              !healthAvailable
                ? 'Health checks unavailable'
                : connected.length
                  ? modelHealthHint(totals)
                  : 'No backend to check'
            }
          />
          <InventoryStat
            index={2}
            icon={Gauge}
            tone="violet"
            label="Median p95"
            loading={health.phase === 'loading'}
            value={medianP95 !== undefined ? formatLatency(medianP95) : null}
            hint={
              !healthAvailable
                ? 'Health checks unavailable'
                : medianP95 !== undefined
                  ? 'Across healthy endpoints'
                  : 'No healthy endpoint yet'
            }
          />
          <InventoryStat
            index={3}
            icon={CircleDollarSign}
            tone="gold"
            label="Spend · 24h"
            loading={traffic.phase === 'loading'}
            value={
              traffic.phase === 'ready' && traffic.requests > 0 && traffic.spend
                ? formatInsightsCost(traffic.spend.amount, traffic.spend.currency, { total: true })
                : null
            }
            hint={
              traffic.phase === 'unavailable'
                ? TRAFFIC_UNAVAILABLE[traffic.reason ?? 'router_unreachable']
                : traffic.requests === 0
                  ? 'No requests in the last 24 h'
                  : traffic.savedPercent !== null
                    ? `${Math.round(traffic.savedPercent)}% saved by routing`
                    : `${traffic.requests.toLocaleString()} requests${
                        traffic.spend && traffic.spend.currencies > 1
                          ? ` · ${traffic.spend.currencies} currencies`
                          : ''
                      }`
            }
          />
        </div>

        {deletion.error ? (
          <div role="alert" className="mb-3">
            <Alert
              tone="danger"
              title="The model was not deleted"
              actions={
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => deletion.setError(null)}
                  aria-label="Dismiss model operation error"
                >
                  Dismiss
                </Button>
              }
            >
              {deletion.error}
            </Alert>
          </div>
        ) : null}

        <DataToolbar
          search={props.modelsSearch}
          onSearchChange={props.onModelsSearchChange}
          placeholder="Search models, IDs or tags…"
          filters={
            <>
              {filters.providerOptions.length ? (
                <FilterPill
                  label="Provider"
                  selected={filters.providers}
                  onChange={filters.setProviders}
                  options={filters.providerOptions.map((option) => ({
                    ...option,
                    icon: (
                      <CatalogProviderLogo provider={option.value} size={14} catalog={catalog} />
                    ),
                  }))}
                />
              ) : null}
              <FilterPill
                label="Kind"
                selected={filters.kinds}
                onChange={filters.setKinds}
                options={filters.kindOptions}
              />
              {healthAvailable ? (
                <FilterPill
                  label="Health"
                  selected={filters.health}
                  onChange={filters.setHealth}
                  options={filters.healthOptions}
                />
              ) : null}
              {filters.active ? (
                <Button variant="ghost" size="sm" onClick={filters.clear}>
                  Clear filters
                </Button>
              ) : null}
            </>
          }
          actions={
            <>
              <span className="text-xs text-muted-foreground" aria-live="polite">
                <span className="tabular">{filters.models.length}</span> of{' '}
                <span className="tabular">{props.models.length}</span> models
              </span>
              {healthAvailable ? (
                <Tooltip
                  content={
                    lastChecked
                      ? `Checked ${lastChecked} · checks run every ${health.intervalSeconds} s`
                      : `Checks run every ${health.intervalSeconds} s`
                  }
                >
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => void checkAllHealth()}
                    disabled={health.checking || health.phase === 'loading'}
                    loading={health.checking}
                  >
                    {health.checking ? null : <Activity />}
                    {health.checking ? 'Checking…' : 'Check health'}
                  </Button>
                </Tooltip>
              ) : null}
            </>
          }
        />
        <ConfigPageModelsTable
          models={filters.models}
          totalModels={props.models.length}
          defaultModel={props.defaultModel}
          deployments={filters.deployments}
          referenceCounts={filters.referenceCounts}
          kinds={filters.kindByModel}
          catalog={catalog}
          health={health}
          traffic={traffic}
          isReadonly={props.isReadonly}
          activeName={active?.name ?? null}
          resetKey={filters.resetKey}
          filtersActive={filters.active}
          selectedKeys={deletion.selectedKeys}
          onSelectedKeysChange={deletion.setSelectedKeys}
          getDeleteBlocker={deletion.blocker}
          canBeDefault={defaults.canBeDefault}
          onOpen={openModel}
          onShowUsage={showUsage}
          onEdit={controller.forms.edit}
          onDelete={deletion.requestOne}
          onMakeDefault={defaults.request}
          onCheckHealth={(model) => void checkModelHealth(model)}
          onConnect={openConnect}
          onClearFilters={filters.clear}
        />
      </ConfigPageManagerLayout>

      <BulkBar
        count={props.isReadonly ? 0 : deletion.selectedKeys.size}
        onClear={() => deletion.setSelectedKeys(new Set())}
      >
        <Button
          variant="destructive-ghost"
          size="sm"
          onClick={() => {
            deletion.setError(null)
            deletion.setModelNames([...deletion.selectedKeys])
          }}
        >
          <Trash2 />
          Delete selected
        </Button>
      </BulkBar>

      <ConfigPageModelSheet
        model={shown}
        open={Boolean(active)}
        view={view}
        onViewChange={(next) =>
          update({ view: next === 'overview' ? null : next }, { replace: true })
        }
        onOpenChange={(open) => (open ? null : closeModel())}
        config={props.config}
        catalog={catalog}
        defaultModel={props.defaultModel}
        isReadonly={props.isReadonly}
        canVerifyModels={props.canVerifyModels}
        decisionReferences={decisionReferences}
        deployments={deployments}
        evaluationGroup={evaluationGroup}
        evaluationsReady={evaluations.ready}
        evaluationsLoading={evaluations.loading}
        onRetryEvaluations={evaluations.retry}
        catalogFallback={Boolean(controller.catalogError)}
        reasoningFamilies={props.reasoningFamilies}
        health={shown ? health.byModel.get(shown.name) : undefined}
        healthPhase={health.phase}
        healthChecking={Boolean(
          shown && (health.checking || health.checkingModels.has(shown.name)),
        )}
        healthIntervalSeconds={health.intervalSeconds}
        onCheckHealth={() => shown && void checkModelHealth(shown)}
        trafficShare={trafficShare}
        liveState={
          shown
            ? modelLiveVerificationState(liveVerification.states, shown.name)
            : { status: 'idle' }
        }
        onVerify={() => shown && void liveVerification.verify(shown.name)}
        canBeDefault={shown ? defaults.canBeDefault(shown) : false}
        deleteBlocker={shown ? deletion.blocker(shown.name) : null}
        onEdit={() => shown && controller.forms.edit(shown)}
        onDelete={() => shown && deletion.requestOne(shown)}
        onMakeDefault={() => shown && defaults.request(shown)}
        onManageEvaluations={evaluations.manage}
      />

      <ConfigPageReasoningFamilies
        open={familiesOpen}
        families={props.reasoningFamilies}
        models={props.models}
        activeName={params.get('family')}
        onOpen={(name) => update({ families: '1', family: name, tab: null })}
        onBack={() => update({ family: null, families: '1', tab: null })}
        onClose={closeFamilies}
        onOpenModel={(name) =>
          update({ families: null, family: null, tab: null, m: name, view: 'reasoning' })
        }
      />

      {props.isReadonly ? null : (
        <ConfigPageConnectModelsDialog
          isOpen={connect !== null}
          existingModelNames={[
            ...props.models.map((model) => model.name),
            ...(props.config?.entrypoints ?? []).flatMap((entrypoint) => entrypoint.model_names),
          ]}
          reasoningFamilies={Object.keys(props.reasoningFamilies)}
          catalog={catalog}
          catalogModel={connect?.catalogModel}
          onClose={() => setConnect(null)}
          onImport={(input) => controller.forms.connect(input)}
          onManualSetup={controller.forms.add}
        />
      )}

      <ModelDeleteDialog
        modelNames={modelsPendingDelete}
        pending={deletion.pending}
        onCancel={() => deletion.setModelNames([])}
        onConfirm={() =>
          void deletion.remove(modelsPendingDelete, (names) => {
            if (activeName && names.includes(activeName)) closeModel()
          })
        }
      />

      <ConfirmDialog
        isOpen={defaults.target !== null}
        tone="neutral"
        title={`Make ${defaults.target ?? ''} the default model?`}
        description="Requests that no decision matches go to the default model. The change applies when you confirm."
        confirmLabel="Make default"
        pendingLabel="Saving…"
        pending={defaults.pending}
        errorMessage={defaults.error}
        onCancel={defaults.cancel}
        onConfirm={defaults.confirm}
      />
    </>
  )
}
