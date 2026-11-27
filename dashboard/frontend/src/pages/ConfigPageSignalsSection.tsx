import * as React from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { ChevronDown, CopyPlus, Eye, FileDown, Pencil, Plus, Radar, Trash2 } from 'lucide-react'

import ConfirmDialog from '../components/ConfirmDialog'
import RoutingScopeSelector from '../components/RoutingScopeSelector'
import { configCollectionItemYaml } from '../lib/configYaml'
import {
  Badge,
  BulkBar,
  Button,
  DataTable,
  DataToolbar,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
  EmptyState,
  FilterPill,
  IconTile,
  Tooltip,
  cn,
  toast,
  useCopy,
  type Column,
} from '../ui'
import ConfigPageManagerLayout from './ConfigPageManagerLayout'
import ConfigPageSignalSheet, {
  type SignalSheetTarget,
  type SignalSheetView,
} from './ConfigPageSignalsSectionSheet'
import { SchemaFormHintsProvider } from './ConfigPageSchemaFieldsEditorSupport'
import type { OpenEditModal, OpenViewModal } from './configPageRouterSectionSupport'
import { routingScopeFormHints, routingScopeInventory } from './configPageRoutingScopeHints'
import {
  routingScopeRecipeName,
  useRoutingPageParams,
  useRoutingScopeManager,
} from './configPageRoutingScopeSupport'
import {
  SIGNAL_CATALOG,
  findSignalCatalogEntry,
  signalTypeIcon,
  type SignalCatalogEntry,
} from './configPageSignalCatalog'
import {
  signalEvaluator,
  signalEvaluatorLabel,
  type SignalEvaluator,
} from './configPageSignalEvaluator'
import {
  applySignalDraft,
  copyName,
  removeSignalsFromConfig,
  signalDraftFromRecord,
  signalRecords,
  signalSummary,
  type SignalDraft,
  type SignalRef,
} from './configPageSignalForm'
import { signalUsageInRoutingProfile, type SignalUsage } from './configPageSignalReferences'
import { signalBindings, type SignalCapabilityScope } from './signalCapabilities'
import type { ConfigData } from './configPageSupport'
import { getNormalizedModels } from './configPageModelNormalization'
import { useDecisionTasks } from './useDecisionTasks'

interface ConfigPageSignalsSectionProps {
  config: ConfigData | null
  isReadonly: boolean
  signalsSearch: string
  onSignalsSearchChange: (value: string) => void
  saveConfig: (config: ConfigData) => Promise<void>
  /** Kept for ConfigPage; signals edit in their own drawer. */
  openEditModal?: OpenEditModal
  openViewModal?: OpenViewModal
}

interface SignalRow {
  key: string
  type: string
  name: string
  entry: SignalCatalogEntry
  record: Record<string, unknown>
  summary: string
  usage: SignalUsage
  evaluator: SignalEvaluator
}

type UsageKind = 'decisions' | 'projections' | 'guards' | 'unused'

function usageKind(usage: SignalUsage): UsageKind {
  if (usage.decisions.length) return 'decisions'
  if (usage.projections.length) return 'projections'
  if (usage.guards.length || usage.masking.length) return 'guards'
  return 'unused'
}

const signalKey = (ref: SignalRef) => `${ref.type}:${ref.name}`

function parseSignalKey(key: string | null): SignalRef | null {
  if (!key) return null
  const index = key.indexOf(':')
  return index > 0 ? { type: key.slice(0, index), name: key.slice(index + 1) } : null
}

function TypeNav({
  value,
  onChange,
  counts,
  total,
}: {
  value: string
  onChange: (value: string) => void
  counts: Map<string, number>
  total: number
}) {
  const scroller = React.useRef<HTMLDivElement>(null)
  const [edges, setEdges] = React.useState({ left: false, right: false })
  const measure = React.useCallback(() => {
    const node = scroller.current
    if (node)
      setEdges({
        left: node.scrollLeft > 4,
        right: node.scrollLeft + node.clientWidth < node.scrollWidth - 4,
      })
  }, [])
  React.useEffect(() => {
    measure()
    window.addEventListener('resize', measure)
    return () => window.removeEventListener('resize', measure)
  }, [measure, counts])
  const items = [
    { type: 'all', label: 'All', count: total },
    ...SIGNAL_CATALOG.filter((entry) => counts.get(entry.type)).map((entry) => ({
      type: entry.type,
      label: entry.label,
      count: counts.get(entry.type) ?? 0,
    })),
  ]
  return (
    <div className="relative">
      <div
        ref={scroller}
        role="tablist"
        aria-label="Signal types"
        onScroll={measure}
        className="flex gap-1.5 overflow-x-auto pb-0.5 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
      >
        {items.map((item) => {
          const active = value === item.type
          const Icon = item.type === 'all' ? Radar : signalTypeIcon(item.type)
          return (
            <button
              key={item.type}
              type="button"
              role="tab"
              aria-selected={active}
              onClick={() => onChange(item.type)}
              className={cn(
                'relative inline-flex h-8 shrink-0 items-center gap-1.5 rounded-full border px-3 text-[13px] font-medium outline-none transition-colors focus-visible:ring-[3px] focus-visible:ring-ring/40',
                active
                  ? 'border-transparent text-foreground'
                  : 'border-border bg-surface text-muted-foreground hover:border-border-strong hover:text-foreground',
              )}
            >
              {active ? (
                <motion.span
                  layoutId="signal-type-nav"
                  className="absolute inset-0 rounded-full bg-chart-2/14 ring-1 ring-inset ring-chart-2/45"
                  transition={{ type: 'spring', bounce: 0.18, duration: 0.4 }}
                />
              ) : null}
              <Icon
                className={cn(
                  'relative size-3.5',
                  active ? 'text-chart-2' : 'text-subtle-foreground',
                )}
              />
              <span className="relative">{item.label}</span>
              <span className="relative font-mono text-[11px] tabular text-subtle-foreground">
                {item.count}
              </span>
            </button>
          )
        })}
      </div>
      <span
        aria-hidden
        className={cn(
          'pointer-events-none absolute inset-y-0 left-0 w-10 bg-gradient-to-r from-background transition-opacity',
          edges.left ? 'opacity-100' : 'opacity-0',
        )}
      />
      <span
        aria-hidden
        className={cn(
          'pointer-events-none absolute inset-y-0 right-0 w-14 bg-gradient-to-l from-background transition-opacity',
          edges.right ? 'opacity-100' : 'opacity-0',
        )}
      />
    </div>
  )
}

function UsageCell({ usage }: { usage: SignalUsage }) {
  if (usage.total === 0) {
    return (
      <Tooltip content="No decision, projection or response guard reads this rule, so the Router skips it">
        <span className="text-xs text-subtle-foreground">Nothing yet</span>
      </Tooltip>
    )
  }
  const names = [...usage.decisions, ...usage.guards, ...usage.masking].map(
    (decision) => decision.name,
  )
  return (
    <Tooltip
      content={[...new Set(names), ...usage.projections.map((projection) => projection.name)].join(
        ', ',
      )}
    >
      <span className="flex items-center gap-1.5 text-xs">
        {usage.decisions.length ? (
          <span>
            <span className="font-medium tabular text-foreground">{usage.decisions.length}</span>{' '}
            <span className="text-muted-foreground">
              {usage.decisions.length === 1 ? 'decision' : 'decisions'}
            </span>
          </span>
        ) : null}
        {usage.projections.length ? (
          <Badge size="sm" tone="teal">
            {usage.projections.length}{' '}
            {usage.projections.length === 1 ? 'projection' : 'projections'}
          </Badge>
        ) : null}
        {usage.guards.length ? (
          <Badge size="sm" tone="rose">
            response guard
          </Badge>
        ) : null}
        {usage.masking.length ? (
          <Badge size="sm" tone="rose">
            masking
          </Badge>
        ) : null}
      </span>
    </Tooltip>
  )
}

function EvaluatorCell({ evaluator }: { evaluator: SignalEvaluator }) {
  if (evaluator.kind === 'rule') {
    return (
      <Badge tone="outline" className="font-mono">
        {signalEvaluatorLabel(evaluator)}
      </Badge>
    )
  }
  return (
    <Tooltip
      content={
        evaluator.source
          ? `Answered by this deployment through its ${evaluator.source}`
          : 'Answered by a model deployment'
      }
    >
      <span className="block max-w-[150px] truncate font-mono text-[12px] text-foreground/85">
        {signalEvaluatorLabel(evaluator)}
      </span>
    </Tooltip>
  )
}

export default function ConfigPageSignalsSection({
  config,
  isReadonly,
  signalsSearch,
  onSignalsSearchChange,
  saveConfig,
}: ConfigPageSignalsSectionProps) {
  const navigate = useNavigate()
  const { copy } = useCopy()
  const { params, update } = useRoutingPageParams()
  const { applyScopedConfig, routingScopes, scopedConfig, selectedScope, selectedScopeId } =
    useRoutingScopeManager(config, params.get('recipe') ?? '')
  const tasks = useDecisionTasks()
  const [usageFilter, setUsageFilter] = React.useState<string[]>([])
  const [evaluatorFilter, setEvaluatorFilter] = React.useState<string[]>([])
  const [selected, setSelected] = React.useState<Set<string>>(new Set())
  const [pendingDelete, setPendingDelete] = React.useState<SignalRef[]>([])
  const [deleting, setDeleting] = React.useState(false)
  const [deleteError, setDeleteError] = React.useState<string | null>(null)
  const [duplicateDraft, setDuplicateDraft] = React.useState<SignalDraft | null>(null)

  React.useEffect(() => {
    setSelected(new Set())
    setUsageFilter([])
    setEvaluatorFilter([])
  }, [selectedScopeId])

  const scopeLabel = routingScopeRecipeName(
    config,
    selectedScopeId,
    selectedScope?.isDefault ?? true,
  )
  const routing = selectedScope?.routing as ConfigData['routing']
  const models = React.useMemo(() => getNormalizedModels(config), [config])
  const inventory = React.useMemo(
    () =>
      routingScopeInventory(
        scopedConfig,
        models,
        tasks.data?.deployments.map((deployment) => deployment.deployment),
      ),
    [scopedConfig, models, tasks.data],
  )
  const hints = React.useMemo(() => routingScopeFormHints(inventory, 'gold'), [inventory])
  const evaluatorContext = React.useMemo(
    () => ({
      tasks: tasks.data,
      recipeBindings: selectedScope?.routing.model_bindings as Record<string, unknown> | undefined,
      globalBindings: config?.global?.model_catalog?.bindings as
        | Record<string, unknown>
        | undefined,
      decisionModel: config?.global?.model_catalog?.system?.decision_model?.deployment,
    }),
    [tasks.data, selectedScope, config],
  )

  const rows = React.useMemo<SignalRow[]>(
    () =>
      SIGNAL_CATALOG.flatMap((entry) =>
        signalRecords(scopedConfig?.signals, entry.collection).map((record) => ({
          key: signalKey({ type: entry.type, name: record.name }),
          type: entry.type,
          name: record.name,
          entry,
          record,
          summary: signalSummary(record),
          usage: signalUsageInRoutingProfile(routing, entry.type, record.name),
          evaluator: signalEvaluator(entry.type, record, evaluatorContext),
        })),
      ),
    [scopedConfig, routing, evaluatorContext],
  )

  const typeParam = params.get('type') ?? 'all'
  const typeFilter = typeParam === 'all' || findSignalCatalogEntry(typeParam) ? typeParam : 'all'
  const counts = React.useMemo(() => {
    const next = new Map<string, number>()
    rows.forEach((row) => next.set(row.type, (next.get(row.type) ?? 0) + 1))
    return next
  }, [rows])
  const search = signalsSearch.trim().toLowerCase()
  const visibleRows = rows
    .filter((row) => typeFilter === 'all' || row.type === typeFilter)
    .filter(
      (row) =>
        !search ||
        `${row.name} ${row.type} ${row.entry.label} ${row.summary} ${signalEvaluatorLabel(row.evaluator)}`
          .toLowerCase()
          .includes(search),
    )
    .filter((row) => usageFilter.length === 0 || usageFilter.includes(usageKind(row.usage)))
    .filter(
      (row) =>
        evaluatorFilter.length === 0 ||
        evaluatorFilter.includes(signalEvaluatorLabel(row.evaluator)),
    )
  const unusedCount = rows.filter((row) => row.usage.total === 0).length
  const evaluatorOptions = [
    ...new Set(rows.map((row) => signalEvaluatorLabel(row.evaluator))),
  ].sort()

  // Drawer state lives in the URL: ?id=<type>:<name>, or ?new=<type>.
  const activeRef = parseSignalKey(params.get('id'))
  const activeRow = activeRef ? rows.find((row) => row.key === signalKey(activeRef)) : undefined
  const newType = params.get('new')
  const view = (
    ['overview', 'try', 'used', 'yaml'].includes(params.get('view') ?? '')
      ? params.get('view')
      : 'overview'
  ) as SignalSheetView
  const sheetTarget = React.useMemo<SignalSheetTarget | null>(() => {
    if (activeRow) {
      return {
        mode: 'edit',
        original: { type: activeRow.type, name: activeRow.name, record: activeRow.record },
        draft: signalDraftFromRecord(activeRow.type, activeRow.record),
      }
    }
    if (newType && findSignalCatalogEntry(newType) && !isReadonly) {
      return {
        mode: 'create',
        draft: duplicateDraft?.type === newType ? duplicateDraft : signalDraftFromRecord(newType),
      }
    }
    return null
  }, [activeRow, newType, duplicateDraft, isReadonly])
  const shownTarget = React.useRef<SignalSheetTarget | null>(null)
  if (sheetTarget) shownTarget.current = sheetTarget

  const capabilityScope: SignalCapabilityScope = {
    recipe: selectedScope?.isDefault ? 'default' : selectedScope?.id || 'default',
    bindings: signalBindings(selectedScope?.routing.model_bindings),
    globalBindings: config?.global?.model_catalog?.bindings,
    defaultDeployment: config?.global?.model_catalog?.system?.decision_model?.deployment,
  }

  const openSignal = (row: SignalRef, nextView?: SignalSheetView) =>
    update({ id: signalKey(row), new: null, view: nextView ?? null })
  const closeSheet = () => {
    setDuplicateDraft(null)
    update({ id: null, new: null, view: null })
  }
  const startCreate = (type: string, draft?: SignalDraft) => {
    setDuplicateDraft(draft ?? null)
    update({ new: type, id: null, view: null })
  }
  const duplicate = (signal: SignalRef & { record: Record<string, unknown> }) => {
    const entry = findSignalCatalogEntry(signal.type)
    const existing = entry
      ? signalRecords(scopedConfig?.signals, entry.collection).map((record) => record.name)
      : []
    const draft = signalDraftFromRecord(signal.type, signal.record)
    startCreate(signal.type, { ...draft, name: copyName(existing, signal.name) })
  }
  const copyYaml = (targets: SignalRow[]) => {
    copy(
      targets
        .map((row) =>
          configCollectionItemYaml(
            ['signals', row.entry.collection],
            row.record,
            row.entry.schemaRef,
          ),
        )
        .join('\n'),
    )
    toast.success(
      targets.length === 1
        ? `${targets[0].name} YAML copied`
        : `${targets.length} signals copied as YAML`,
    )
  }
  const saveSignal = async (draft: SignalDraft, original?: SignalRef) => {
    if (!scopedConfig) throw new Error('Routing profile not loaded yet.')
    await saveConfig(applyScopedConfig(applySignalDraft(scopedConfig, draft, original)))
    setDuplicateDraft(null)
    update(
      { id: signalKey({ type: draft.type, name: draft.name.trim() }), new: null },
      { replace: true },
    )
  }
  const requestDelete = (targets: SignalRef[]) => {
    setDeleteError(null)
    setPendingDelete(targets)
  }
  const confirmDelete = async () => {
    if (!scopedConfig || pendingDelete.length === 0) return
    setDeleting(true)
    setDeleteError(null)
    try {
      await saveConfig(applyScopedConfig(removeSignalsFromConfig(scopedConfig, pendingDelete)))
      toast.success(
        pendingDelete.length === 1
          ? `${pendingDelete[0].name} deleted`
          : `${pendingDelete.length} signals deleted`,
      )
      if (activeRef && pendingDelete.some((target) => signalKey(target) === signalKey(activeRef)))
        closeSheet()
      setSelected(new Set())
      setPendingDelete([])
    } catch (failure) {
      setDeleteError(
        failure instanceof Error ? failure.message : 'The signals could not be deleted.',
      )
    } finally {
      setDeleting(false)
    }
  }

  const columns: Column<SignalRow>[] = [
    {
      id: 'name',
      header: 'Signal',
      sortValue: (row) => row.name,
      cell: (row) => (
        <span className="flex min-w-0 items-center gap-1.5">
          <span className="max-w-[220px] truncate font-mono text-[13px] font-medium">
            {row.name}
          </span>
          {row.record.stage === 'response' || !row.entry.decisionReferenceable ? (
            <Tooltip content="Reads the model's answer">
              <span>
                <Badge size="sm" tone="outline">
                  response
                </Badge>
              </span>
            </Tooltip>
          ) : null}
        </span>
      ),
    },
    {
      id: 'type',
      header: 'Type',
      sortValue: (row) => SIGNAL_CATALOG.indexOf(row.entry),
      cell: (row) => {
        const Icon = signalTypeIcon(row.type)
        return (
          <span className="flex items-center gap-2 whitespace-nowrap text-[13px]">
            <Icon className="size-3.5 shrink-0 text-chart-2" />
            {row.entry.label}
          </span>
        )
      },
    },
    {
      id: 'evaluator',
      header: 'Evaluated by',
      hideBelow: 'xl',
      sortValue: (row) => signalEvaluatorLabel(row.evaluator),
      cell: (row) => <EvaluatorCell evaluator={row.evaluator} />,
    },
    {
      id: 'summary',
      header: 'Summary',
      hideBelow: '2xl',
      cell: (row) => (
        <span className="block max-w-[300px] truncate text-muted-foreground">{row.summary}</span>
      ),
    },
    {
      id: 'used',
      header: 'Used by',
      sortValue: (row) =>
        row.usage.decisions.length * 100 +
        row.usage.projections.length * 10 +
        row.usage.guards.length +
        row.usage.masking.length,
      cell: (row) => <UsageCell usage={row.usage} />,
    },
  ]

  const newSignalMenu = (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button>
          <Plus />
          New signal
          <ChevronDown className="-mr-1 opacity-70" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        align="end"
        className="max-h-[min(520px,var(--radix-dropdown-menu-content-available-height))] w-64 overflow-y-auto"
      >
        <DropdownMenuLabel>Signal type</DropdownMenuLabel>
        {SIGNAL_CATALOG.map((entry) => {
          const Icon = signalTypeIcon(entry.type)
          return (
            <DropdownMenuItem key={entry.type} onSelect={() => startCreate(entry.type)}>
              <Icon className="text-chart-2" />
              <span className="flex-1">{entry.label}</span>
              <span className="font-mono text-[11px] text-subtle-foreground">{entry.type}</span>
            </DropdownMenuItem>
          )
        })}
      </DropdownMenuContent>
    </DropdownMenu>
  )

  return (
    <SchemaFormHintsProvider value={hints}>
      <ConfigPageManagerLayout
        title="Signals"
        description="Evidence extracted from each request that decisions combine to choose a model."
        icon={<IconTile icon={Radar} tone="gold" size="lg" />}
        actions={
          <>
            <RoutingScopeSelector
              label="Recipe"
              scopes={routingScopes}
              value={selectedScopeId}
              onChange={(scopeId) =>
                update({ recipe: scopeId, id: null, new: null, view: null, type: null })
              }
            />
            {isReadonly ? null : newSignalMenu}
          </>
        }
        tabs={
          rows.length ? (
            <TypeNav
              value={typeFilter}
              onChange={(type) => update({ type: type === 'all' ? null : type }, { replace: true })}
              counts={counts}
              total={rows.length}
            />
          ) : undefined
        }
      >
        <DataToolbar
          search={signalsSearch}
          onSearchChange={onSignalsSearchChange}
          placeholder="Search signals…"
          filters={
            <>
              <FilterPill
                label="Usage"
                selected={usageFilter}
                onChange={setUsageFilter}
                options={[
                  {
                    value: 'decisions',
                    label: 'Read by decisions',
                    count: rows.filter((row) => usageKind(row.usage) === 'decisions').length,
                  },
                  {
                    value: 'projections',
                    label: 'Only by projections',
                    count: rows.filter((row) => usageKind(row.usage) === 'projections').length,
                  },
                  {
                    value: 'guards',
                    label: 'Response guards or masking',
                    count: rows.filter((row) => usageKind(row.usage) === 'guards').length,
                  },
                  { value: 'unused', label: 'Skipped, nothing reads it', count: unusedCount },
                ]}
              />
              <FilterPill
                label="Evaluated by"
                selected={evaluatorFilter}
                onChange={setEvaluatorFilter}
                options={evaluatorOptions.map((option) => ({
                  value: option,
                  label: option,
                  count: rows.filter((row) => signalEvaluatorLabel(row.evaluator) === option)
                    .length,
                }))}
              />
            </>
          }
          actions={
            <span className="text-xs text-muted-foreground">
              <span className="tabular">{visibleRows.length}</span> of{' '}
              <span className="tabular">{rows.length}</span> ·{' '}
              <span className="tabular">{rows.length - unusedCount}</span> evaluated per request
            </span>
          }
        />
        <DataTable
          data={visibleRows}
          columns={columns}
          getRowId={(row) => row.key}
          selectable={!isReadonly}
          selected={selected}
          onSelectedChange={setSelected}
          activeRowId={activeRow?.key}
          onRowClick={(row) => openSignal(row)}
          rowActions={(row) => (
            <>
              <DropdownMenuItem onSelect={() => openSignal(row)}>
                {isReadonly ? <Eye /> : <Pencil />}
                {isReadonly ? 'View' : 'Edit'}
              </DropdownMenuItem>
              <DropdownMenuItem onSelect={() => openSignal(row, 'try')}>
                <Radar />
                Try it
              </DropdownMenuItem>
              {isReadonly ? null : (
                <DropdownMenuItem
                  onSelect={() => duplicate({ type: row.type, name: row.name, record: row.record })}
                >
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
                    disabled={row.usage.total > 0}
                    onSelect={() => requestDelete([row])}
                  >
                    <Trash2 />
                    {row.usage.total > 0 ? 'In use' : 'Delete'}
                  </DropdownMenuItem>
                </>
              )}
            </>
          )}
          empty={
            rows.length === 0 ? (
              <EmptyState
                compact
                icon={Radar}
                tone="gold"
                title={`${scopeLabel} has no signals yet`}
                description="Signals are the evidence decisions match on: keywords, domains, safety checks and questions for the decision model."
                actions={isReadonly ? undefined : newSignalMenu}
              />
            ) : (
              <EmptyState
                compact
                icon={Radar}
                tone="gold"
                title="No signals match"
                description="Try another search, clear the filters, or pick All above."
                actions={
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      onSignalsSearchChange('')
                      setUsageFilter([])
                      setEvaluatorFilter([])
                      update({ type: null }, { replace: true })
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
          onClick={() => copyYaml(rows.filter((row) => selected.has(row.key)))}
        >
          <FileDown />
          Copy YAML
        </Button>
        <Button
          variant="destructive-ghost"
          size="sm"
          onClick={() => {
            const chosen = rows.filter((row) => selected.has(row.key))
            const removable = chosen.filter((row) => row.usage.total === 0)
            if (removable.length === 0) {
              toast('Every selected signal is in use', {
                description: 'Remove their references from decisions and projections first.',
              })
              return
            }
            if (removable.length < chosen.length) {
              toast(`${chosen.length - removable.length} selected signals are in use and stay`, {
                description: 'Only signals nothing reads can be deleted.',
              })
            }
            requestDelete(removable)
          }}
        >
          <Trash2 />
          Delete unused
        </Button>
      </BulkBar>

      <ConfigPageSignalSheet
        target={sheetTarget ?? shownTarget.current}
        open={Boolean(sheetTarget)}
        view={view}
        onViewChange={(next) =>
          update({ view: next === 'overview' ? null : next }, { replace: true })
        }
        onOpenChange={(open) => (open ? null : closeSheet())}
        scopeLabel={scopeLabel}
        scopedConfig={scopedConfig}
        readOnly={isReadonly}
        usage={activeRow?.usage}
        evaluator={
          activeRow?.evaluator ??
          (sheetTarget
            ? signalEvaluator(sheetTarget.draft.type, sheetTarget.draft.fields, evaluatorContext)
            : undefined)
        }
        tasks={tasks}
        capabilityScope={capabilityScope}
        entrypointModel={selectedScope?.entrypointModelNames[0]}
        onSave={saveSignal}
        onDelete={(signal) => requestDelete([signal])}
        onDuplicate={duplicate}
        onOpenDecision={(name) =>
          navigate(
            `/routing/decisions?${new URLSearchParams({ recipe: selectedScopeId, id: name })}`,
          )
        }
        onOpenProjection={(kind, name) =>
          navigate(
            `/routing/projections?${new URLSearchParams({ recipe: selectedScopeId, tab: kind === 'score' ? 'scores' : 'partitions', id: `${kind}:${name}` })}`,
          )
        }
        onUseInDecision={(signal) =>
          navigate(
            `/routing/decisions?${new URLSearchParams({ recipe: selectedScopeId, new: '1', when: signalKey(signal) })}`,
          )
        }
      />

      <ConfirmDialog
        isOpen={pendingDelete.length > 0}
        title={
          pendingDelete.length === 1
            ? `Delete ${pendingDelete[0]?.name}?`
            : `Delete ${pendingDelete.length} signals?`
        }
        description={`Nothing in ${scopeLabel} reads ${pendingDelete.length === 1 ? 'this signal' : 'these signals'}. Deleting removes ${pendingDelete.length === 1 ? 'it' : 'them'} from config.yaml.`}
        confirmLabel={pendingDelete.length === 1 ? 'Delete signal' : 'Delete signals'}
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
