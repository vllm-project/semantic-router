import * as React from 'react'
import { useNavigate } from 'react-router-dom'
import { CopyPlus, Eye, FileDown, MessagesSquare, Pencil, Plus, Split, Trash2 } from 'lucide-react'

import ConfirmDialog from '../components/ConfirmDialog'
import RoutingScopeSelector from '../components/RoutingScopeSelector'
import { configCollectionItemYaml } from '../lib/configYaml'
import {
  BulkBar,
  Button,
  DataTable,
  DataToolbar,
  DropdownMenuItem,
  DropdownMenuSeparator,
  EmptyState,
  FilterPill,
  IconTile,
  ProviderLogo,
  Tabs,
  TabsList,
  TabsTrigger,
  Tooltip,
  toast,
  useCopy,
} from '../ui'
import ConfigPageManagerLayout from './ConfigPageManagerLayout'
import ConfigPageDecisionSheet, {
  type DecisionSheetTarget,
  type DecisionSheetView,
} from './ConfigPageDecisionsSectionSheet'
import { SchemaFormHintsProvider } from './ConfigPageSchemaFieldsEditorSupport'
import { cloneConfigData } from './configPageCanonicalization'
import { cloneDecision, newDecisionDraft, upsertDecision } from './configPageDecisionFormSupport'
import {
  DECISION_KIND_HINT,
  algorithmMeta,
  decisionKind,
  modelDisplayName,
  modelLogo,
  pluginLabel,
  type DecisionKind,
} from './configPageDecisionPresentation'
import { decisionColumns } from './configPageDecisionTable'
import { useDecisionTraffic } from './configPageDecisionTraffic'
import { ruleExpression } from './configPageDecisionRuleSupport'
import type { OpenEditModal, OpenViewModal } from './configPageRouterSectionSupport'
import { routingScopeFormHints, routingScopeInventory } from './configPageRoutingScopeHints'
import {
  routingScopeRecipeName,
  useRoutingPageParams,
  useRoutingScopeManager,
} from './configPageRoutingScopeSupport'
import { copyName } from './configPageSignalForm'
import {
  getReasoningFamiliesMap,
  type ConfigData,
  type DecisionConfig,
  type NormalizedModel,
} from './configPageSupport'

interface ConfigPageDecisionsSectionProps {
  config: ConfigData | null
  isReadonly: boolean
  decisionsSearch: string
  onDecisionsSearchChange: (value: string) => void
  saveConfig: (config: ConfigData) => Promise<void>
  /** Kept for ConfigPage; decisions edit in their own drawer. */
  openEditModal?: OpenEditModal
  openViewModal?: OpenViewModal
  removeDecisionByName?: (cfg: ConfigData, targetName: string) => void
  models: NormalizedModel[]
}

type KindFilter = 'all' | DecisionKind
const KIND_TABS: Array<[KindFilter, string]> = [
  ['all', 'All'],
  ['routing', 'Routing'],
  ['guardrail', 'Guardrails'],
  ['fallback', 'Fallback'],
]
const VIEWS: DecisionSheetView[] = [
  'overview',
  'rules',
  'models',
  'algorithm',
  'plugins',
  'advanced',
  'try',
  'yaml',
]

function parseWhen(value: string | null): DecisionConfig['rules'] {
  if (!value) return {}
  const index = value.indexOf(':')
  if (index <= 0) return {}
  return {
    operator: 'AND',
    conditions: [{ type: value.slice(0, index), name: value.slice(index + 1) }],
  }
}

export default function ConfigPageDecisionsSection({
  config,
  isReadonly,
  decisionsSearch,
  onDecisionsSearchChange,
  saveConfig,
  models,
}: ConfigPageDecisionsSectionProps) {
  const navigate = useNavigate()
  const { copy } = useCopy()
  const { params, update } = useRoutingPageParams()
  const { applyScopedConfig, routingScopes, scopedConfig, selectedScope, selectedScopeId } =
    useRoutingScopeManager(config, params.get('recipe') ?? '')
  const reasoningFamilies = React.useMemo(() => getReasoningFamiliesMap(), [])
  const [modelFilter, setModelFilter] = React.useState<string[]>([])
  const [algorithmFilter, setAlgorithmFilter] = React.useState<string[]>([])
  const [pluginFilter, setPluginFilter] = React.useState<string[]>([])
  const [selected, setSelected] = React.useState<Set<string>>(new Set())
  const [pendingDelete, setPendingDelete] = React.useState<string[]>([])
  const [deleting, setDeleting] = React.useState(false)
  const [deleteError, setDeleteError] = React.useState<string | null>(null)
  const [duplicateDraft, setDuplicateDraft] = React.useState<DecisionConfig | null>(null)

  React.useEffect(() => {
    setSelected(new Set())
    setModelFilter([])
    setAlgorithmFilter([])
    setPluginFilter([])
  }, [selectedScopeId])

  const scopeLabel = routingScopeRecipeName(
    config,
    selectedScopeId,
    selectedScope?.isDefault ?? true,
  )
  const strategy = (selectedScope?.routing.strategy as string | undefined) ?? 'priority'
  const traffic = useDecisionTraffic(scopeLabel)
  const decisions = React.useMemo(() => scopedConfig?.decisions ?? [], [scopedConfig])
  const inventory = React.useMemo(
    () => routingScopeInventory(scopedConfig, models),
    [scopedConfig, models],
  )
  const hints = React.useMemo(() => routingScopeFormHints(inventory, 'violet'), [inventory])

  const kindParam = (params.get('kind') ?? 'all') as KindFilter
  const kind: KindFilter = KIND_TABS.some(([value]) => value === kindParam) ? kindParam : 'all'
  const counts = React.useMemo(() => {
    const next: Record<KindFilter, number> = {
      all: decisions.length,
      routing: 0,
      guardrail: 0,
      fallback: 0,
    }
    decisions.forEach((decision) => {
      next[decisionKind(decision)] += 1
    })
    return next
  }, [decisions])
  const search = decisionsSearch.trim().toLowerCase()
  const rows = decisions
    .filter((decision) => kind === 'all' || decisionKind(decision) === kind)
    .filter(
      (decision) =>
        !search ||
        `${decision.name} ${decision.description ?? ''} ${ruleExpression(decision.rules)} ${(decision.modelRefs ?? []).map((ref) => ref.model).join(' ')}`
          .toLowerCase()
          .includes(search),
    )
    .filter(
      (decision) =>
        modelFilter.length === 0 ||
        (decision.modelRefs ?? []).some((ref) => modelFilter.includes(ref.model)),
    )
    .filter(
      (decision) =>
        algorithmFilter.length === 0 ||
        algorithmFilter.includes(String(decision.algorithm?.type ?? 'none')),
    )
    .filter(
      (decision) =>
        pluginFilter.length === 0 ||
        (decision.plugins ?? []).some((plugin) => pluginFilter.includes(plugin.type)),
    )
    .sort((a, b) => (b.priority ?? 0) - (a.priority ?? 0))

  const usedModels = [
    ...new Set(decisions.flatMap((decision) => (decision.modelRefs ?? []).map((ref) => ref.model))),
  ].sort()
  const usedAlgorithms = [
    ...new Set(decisions.map((decision) => String(decision.algorithm?.type ?? 'none'))),
  ].sort()
  const usedPlugins = [
    ...new Set(
      decisions.flatMap((decision) => (decision.plugins ?? []).map((plugin) => plugin.type)),
    ),
  ].sort()

  // Drawer state lives in the URL: ?id=<name>, or ?new=1 (optionally &when=<type>:<name>).
  const activeName = params.get('id')
  const active = activeName ? decisions.find((decision) => decision.name === activeName) : undefined
  const creating = params.get('new') === '1' && !isReadonly
  const when = params.get('when')
  const viewParam = params.get('view') as DecisionSheetView | null
  const view: DecisionSheetView = viewParam && VIEWS.includes(viewParam) ? viewParam : 'overview'
  const sheetTarget = React.useMemo<DecisionSheetTarget | null>(() => {
    if (active) return { mode: 'edit', original: active, draft: cloneDecision(active) }
    if (creating) {
      if (duplicateDraft) return { mode: 'create', draft: duplicateDraft }
      return { mode: 'create', draft: { ...newDecisionDraft(), rules: parseWhen(when) } }
    }
    return null
  }, [active, creating, duplicateDraft, when])
  const shownTarget = React.useRef<DecisionSheetTarget | null>(null)
  if (sheetTarget) shownTarget.current = sheetTarget

  const openDecision = (decision: DecisionConfig, nextView?: DecisionSheetView) =>
    update({ id: decision.name, new: null, when: null, view: nextView ?? null })
  const closeSheet = () => {
    setDuplicateDraft(null)
    update({ id: null, new: null, when: null, view: null })
  }
  const startCreate = (draft?: DecisionConfig) => {
    setDuplicateDraft(draft ?? null)
    update({ new: '1', id: null, view: null, when: null })
  }
  const duplicate = (decision: DecisionConfig) => {
    const draft = cloneDecision(decision)
    draft.name = copyName(
      decisions.map((entry) => entry.name),
      decision.name,
    )
    startCreate(draft)
  }
  const copyYaml = (targets: DecisionConfig[]) => {
    copy(
      targets
        .map((decision) => configCollectionItemYaml(['decisions'], decision, '#/$defs/Decision'))
        .join('\n'),
    )
    toast.success(
      targets.length === 1
        ? `${targets[0].name} YAML copied`
        : `${targets.length} decisions copied as YAML`,
    )
  }
  const saveDecision = async (decision: DecisionConfig, originalName?: string) => {
    if (!scopedConfig) throw new Error('Routing profile not loaded yet.')
    const next = cloneConfigData(scopedConfig)
    next.decisions = upsertDecision(next.decisions ?? [], decision, originalName)
    await saveConfig(applyScopedConfig(next))
    setDuplicateDraft(null)
    update({ id: decision.name, new: null, when: null }, { replace: true })
  }
  const confirmDelete = async () => {
    if (!scopedConfig || pendingDelete.length === 0) return
    setDeleting(true)
    setDeleteError(null)
    try {
      const next = cloneConfigData(scopedConfig)
      next.decisions = (next.decisions ?? []).filter(
        (decision) => !pendingDelete.includes(decision.name),
      )
      await saveConfig(applyScopedConfig(next))
      toast.success(
        pendingDelete.length === 1
          ? `${pendingDelete[0]} deleted`
          : `${pendingDelete.length} decisions deleted`,
      )
      if (activeName && pendingDelete.includes(activeName)) closeSheet()
      setSelected(new Set())
      setPendingDelete([])
    } catch (failure) {
      setDeleteError(
        failure instanceof Error ? failure.message : 'The decisions could not be deleted.',
      )
    } finally {
      setDeleting(false)
    }
  }

  const columns = decisionColumns({ models, traffic: traffic ?? undefined })
  const modelOptions = usedModels.map((name) => {
    const model = models.find((candidate) => candidate.name === name)
    return {
      value: name,
      label: modelDisplayName(model, name),
      icon: <ProviderLogo provider={modelLogo(model)} size={14} />,
      count: decisions.filter((decision) =>
        (decision.modelRefs ?? []).some((ref) => ref.model === name),
      ).length,
    }
  })

  return (
    <SchemaFormHintsProvider value={hints}>
      <ConfigPageManagerLayout
        title="Decisions"
        description={`Rules that map request signals to candidate models, a selection algorithm and plugins. ${
          strategy === 'confidence'
            ? 'The most confident match wins.'
            : 'The highest-priority match wins.'
        }`}
        icon={<IconTile icon={Split} tone="violet" size="lg" />}
        actions={
          <>
            <RoutingScopeSelector
              label="Recipe"
              scopes={routingScopes}
              value={selectedScopeId}
              onChange={(scopeId) =>
                update({ recipe: scopeId, id: null, new: null, when: null, view: null, kind: null })
              }
            />
            {isReadonly ? null : (
              <Button onClick={() => startCreate()}>
                <Plus />
                New decision
              </Button>
            )}
          </>
        }
        tabs={
          <Tabs
            value={kind}
            onValueChange={(value) =>
              update({ kind: value === 'all' ? null : value }, { replace: true })
            }
            variant="line"
            className="gap-0"
          >
            <TabsList>
              {KIND_TABS.map(([value, label]) => (
                <Tooltip
                  key={value}
                  content={
                    value === 'all' ? `Every decision in ${scopeLabel}` : DECISION_KIND_HINT[value]
                  }
                >
                  <TabsTrigger value={value}>
                    {label}
                    <span className="rounded-full bg-secondary px-1.5 font-mono text-[11px] tabular text-muted-foreground">
                      {counts[value]}
                    </span>
                  </TabsTrigger>
                </Tooltip>
              ))}
            </TabsList>
          </Tabs>
        }
      >
        <DataToolbar
          search={decisionsSearch}
          onSearchChange={onDecisionsSearchChange}
          placeholder="Search decisions or signals…"
          filters={
            <>
              <FilterPill
                label="Model"
                selected={modelFilter}
                onChange={setModelFilter}
                options={modelOptions}
              />
              <FilterPill
                label="Algorithm"
                selected={algorithmFilter}
                onChange={setAlgorithmFilter}
                options={usedAlgorithms.map((type) => ({
                  value: type,
                  label: type === 'none' ? 'Not set' : (algorithmMeta(type)?.label ?? type),
                  count: decisions.filter(
                    (decision) => String(decision.algorithm?.type ?? 'none') === type,
                  ).length,
                }))}
              />
              <FilterPill
                label="Plugin"
                selected={pluginFilter}
                onChange={setPluginFilter}
                options={usedPlugins.map((type) => ({
                  value: type,
                  label: pluginLabel(type),
                  count: decisions.filter((decision) =>
                    (decision.plugins ?? []).some((plugin) => plugin.type === type),
                  ).length,
                }))}
              />
            </>
          }
          actions={
            <span className="text-xs text-muted-foreground">
              <span className="tabular">{rows.length}</span> of{' '}
              <span className="tabular">{decisions.length}</span> · sorted by priority
              {strategy === 'confidence' ? ' · selected by confidence' : ''}
            </span>
          }
        />
        <DataTable
          data={rows}
          columns={columns}
          getRowId={(row) => row.name}
          selectable={!isReadonly}
          selected={selected}
          onSelectedChange={setSelected}
          activeRowId={active?.name}
          onRowClick={(row) => openDecision(row)}
          rowActions={(row) => (
            <>
              <DropdownMenuItem onSelect={() => openDecision(row)}>
                {isReadonly ? <Eye /> : <Pencil />}
                {isReadonly ? 'View' : 'Edit'}
              </DropdownMenuItem>
              <DropdownMenuItem onSelect={() => openDecision(row, 'try')}>
                <MessagesSquare />
                Try a prompt
              </DropdownMenuItem>
              {isReadonly ? null : (
                <DropdownMenuItem onSelect={() => duplicate(row)}>
                  <CopyPlus />
                  Duplicate
                </DropdownMenuItem>
              )}
              <DropdownMenuItem onSelect={() => copyYaml([row])}>
                <FileDown />
                Copy YAML
              </DropdownMenuItem>
              {isReadonly ? null : (
                <>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem
                    variant="destructive"
                    onSelect={() => {
                      setDeleteError(null)
                      setPendingDelete([row.name])
                    }}
                  >
                    <Trash2 />
                    Delete
                  </DropdownMenuItem>
                </>
              )}
            </>
          )}
          empty={
            decisions.length === 0 ? (
              <EmptyState
                compact
                icon={Split}
                tone="violet"
                title={`${scopeLabel} has no decisions yet`}
                description="A decision matches request signals and sends the request to its candidate models."
                actions={
                  isReadonly ? undefined : (
                    <Button size="sm" onClick={() => startCreate()}>
                      <Plus />
                      New decision
                    </Button>
                  )
                }
              />
            ) : (
              <EmptyState
                compact
                icon={Split}
                tone="violet"
                title="No decisions match"
                description="Try another search or clear the filters."
                actions={
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      onDecisionsSearchChange('')
                      setModelFilter([])
                      setAlgorithmFilter([])
                      setPluginFilter([])
                      update({ kind: null }, { replace: true })
                    }}
                  >
                    Clear filters
                  </Button>
                }
              />
            )
          }
        />
      </ConfigPageManagerLayout>

      <BulkBar count={selected.size} onClear={() => setSelected(new Set())}>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => copyYaml(decisions.filter((decision) => selected.has(decision.name)))}
        >
          <FileDown />
          Copy YAML
        </Button>
        <Button
          variant="destructive-ghost"
          size="sm"
          onClick={() => {
            setDeleteError(null)
            setPendingDelete(
              decisions
                .filter((decision) => selected.has(decision.name))
                .map((decision) => decision.name),
            )
          }}
        >
          <Trash2 />
          Delete
        </Button>
      </BulkBar>

      <ConfigPageDecisionSheet
        target={sheetTarget ?? shownTarget.current}
        open={Boolean(sheetTarget)}
        view={view}
        onViewChange={(next) =>
          update({ view: next === 'overview' ? null : next }, { replace: true })
        }
        onOpenChange={(open) => (open ? null : closeSheet())}
        scopeLabel={scopeLabel}
        readOnly={isReadonly}
        models={models}
        reasoningFamilies={reasoningFamilies}
        entrypointModel={selectedScope?.entrypointModelNames[0]}
        recordedRequests={active && traffic ? (traffic.get(active.name) ?? 0) : undefined}
        onSave={saveDecision}
        onDelete={(decision) => {
          setDeleteError(null)
          setPendingDelete([decision.name])
        }}
        onDuplicate={duplicate}
        onTestInPlayground={() => navigate('/playground')}
      />

      <ConfirmDialog
        isOpen={pendingDelete.length > 0}
        title={
          pendingDelete.length === 1
            ? `Delete ${pendingDelete[0]}?`
            : `Delete ${pendingDelete.length} decisions?`
        }
        description={`Requests that matched ${pendingDelete.length === 1 ? 'this decision' : 'these decisions'} fall through to the next decision in ${scopeLabel}. This removes ${
          pendingDelete.length === 1 ? 'it' : 'them'
        } from config.yaml.`}
        confirmLabel={pendingDelete.length === 1 ? 'Delete decision' : 'Delete decisions'}
        pending={deleting}
        errorMessage={deleteError ?? undefined}
        onCancel={() => {
          if (!deleting) {
            setPendingDelete([])
            setDeleteError(null)
          }
        }}
        onConfirm={confirmDelete}
      />
    </SchemaFormHintsProvider>
  )
}
