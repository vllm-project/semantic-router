import * as React from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  Blend,
  ChevronDown,
  CopyPlus,
  FileDown,
  MoreHorizontal,
  Pencil,
  Plus,
  Radar,
  Split,
  Trash2,
} from 'lucide-react'

import ConfirmDialog from '../components/ConfirmDialog'
import RoutingScopeSelector from '../components/RoutingScopeSelector'
import { configCollectionItemYaml } from '../lib/configYaml'
import {
  Badge,
  Button,
  Card,
  DataToolbar,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
  EmptyState,
  IconTile,
  SectionLabel,
  Tabs,
  TabsList,
  TabsTrigger,
  Tooltip,
  toast,
  useCopy,
} from '../ui'
import ConfigPageManagerLayout from './ConfigPageManagerLayout'
import ConfigPageProjectionSheet, {
  type ProjectionSheetTarget,
  type ProjectionSheetView,
} from './configPageProjectionSheet'
import { SchemaFormHintsProvider } from './ConfigPageSchemaFieldsEditorSupport'
import { getNormalizedModels } from './configPageModelNormalization'
import {
  newProjectionDraft,
  removeProjection,
  upsertProjection,
  type ProjectionEntity,
} from './configPageProjectionDraft'
import { BandStrip, WeightBars } from './configPageProjectionStructuredEditors'
import {
  PROJECTION_COLLECTION,
  PROJECTION_KIND_LABEL,
  PROJECTION_SCHEMA_REF,
  projectionDecisionConsumers,
  scoreMappings,
  type ProjectionKind,
} from './configPageProjectionTableSupport'
import type { OpenEditModal, OpenViewModal } from './configPageRouterSectionSupport'
import { routingScopeFormHints, routingScopeInventory } from './configPageRoutingScopeHints'
import {
  routingScopeRecipeName,
  useRoutingPageParams,
  useRoutingScopeManager,
} from './configPageRoutingScopeSupport'
import { SignalChip } from './configPageRoutingScopeViews'
import { copyName } from './configPageSignalForm'
import type {
  ConfigData,
  ProjectionMapping,
  ProjectionPartition,
  ProjectionScore,
} from './configPageSupport'

interface ConfigPageProjectionsSectionProps {
  config: ConfigData | null
  isReadonly: boolean
  saveConfig: (config: ConfigData) => Promise<void>
  /** Kept for ConfigPage; projections edit in their own drawer. */
  openEditModal?: OpenEditModal
  openViewModal?: OpenViewModal
}

type ProjectionTab = 'partitions' | 'scores' | 'mappings'
const TAB_KIND: Record<ProjectionTab, ProjectionKind> = {
  partitions: 'partition',
  scores: 'score',
  mappings: 'mapping',
}
const KIND_TAB: Record<ProjectionKind, ProjectionTab> = {
  partition: 'partitions',
  score: 'scores',
  mapping: 'mappings',
}
const KIND_HINT: Record<ProjectionKind, string> = {
  partition: 'Group domain rules so only the strongest one matches a request.',
  score: 'Weigh several signals into one number that mappings turn into bands.',
  mapping: 'Cut a score into named bands that decisions match.',
}
const ease = [0.16, 1, 0.3, 1] as const

function parseKey(value: string | null): { kind: ProjectionKind; name: string } | null {
  if (!value) return null
  const index = value.indexOf(':')
  const kind = value.slice(0, index) as ProjectionKind
  return index > 0 && kind in PROJECTION_COLLECTION ? { kind, name: value.slice(index + 1) } : null
}

function CardMenu({
  kind,
  entity,
  readOnly,
  onOpen,
  onCopy,
  onDuplicate,
  onDelete,
}: {
  kind: ProjectionKind
  entity: ProjectionEntity
  readOnly: boolean
  onOpen: (view?: ProjectionSheetView) => void
  onCopy: () => void
  onDuplicate: () => void
  onDelete: () => void
}) {
  return (
    <span
      onClick={(event) => event.stopPropagation()}
      onKeyDown={(event) => event.stopPropagation()}
      role="presentation"
    >
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            variant="ghost"
            size="icon-xs"
            aria-label={`Actions for ${entity.name}`}
            className="-mr-1.5"
          >
            <MoreHorizontal />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-44">
          <DropdownMenuItem onSelect={() => onOpen()}>
            <Pencil />
            {readOnly ? 'View' : 'Edit'}
          </DropdownMenuItem>
          <DropdownMenuItem onSelect={() => onOpen('try')}>
            <Radar />
            Try it
          </DropdownMenuItem>
          {readOnly ? null : (
            <DropdownMenuItem onSelect={onDuplicate}>
              <CopyPlus />
              Duplicate
            </DropdownMenuItem>
          )}
          <DropdownMenuItem onSelect={onCopy}>
            <FileDown />
            Copy YAML
          </DropdownMenuItem>
          {readOnly ? null : (
            <>
              <DropdownMenuSeparator />
              <DropdownMenuItem variant="destructive" onSelect={onDelete}>
                <Trash2 />
                Delete {PROJECTION_KIND_LABEL[kind].toLowerCase()}
              </DropdownMenuItem>
            </>
          )}
        </DropdownMenuContent>
      </DropdownMenu>
    </span>
  )
}

function ReadByFooter({ names }: { names: string[] }) {
  return (
    <div className="mt-auto flex items-center gap-2 border-t border-border px-4 py-2.5">
      <Split className="size-3.5 shrink-0 text-chart-3" />
      <Tooltip content={names.length ? names.join(', ') : 'No decision reads it yet'}>
        <span className="min-w-0 truncate text-xs text-muted-foreground">
          {names.length ? (
            <>
              Read by <span className="font-medium tabular text-foreground">{names.length}</span>{' '}
              {names.length === 1 ? 'decision' : 'decisions'}:{' '}
              <span className="font-mono">{names.join(', ')}</span>
            </>
          ) : (
            'No decision reads it yet'
          )}
        </span>
      </Tooltip>
    </div>
  )
}

function ProjectionCard({
  kind,
  entity,
  index,
  config,
  readOnly,
  onOpen,
  onCopy,
  onDuplicate,
  onDelete,
}: {
  kind: ProjectionKind
  entity: ProjectionEntity
  index: number
  config: ConfigData | null
  readOnly: boolean
  onOpen: (view?: ProjectionSheetView) => void
  onCopy: () => void
  onDuplicate: () => void
  onDelete: () => void
}) {
  const consumers = projectionDecisionConsumers(config, kind, entity).map(
    (consumer) => consumer.decision.name,
  )
  let badges: React.ReactNode
  let body: React.ReactNode
  if (kind === 'partition') {
    const partition = entity as ProjectionPartition
    badges = (
      <>
        <Badge size="sm" tone="teal">
          Partition · {partition.semantics?.replace(/_/g, ' ') || 'exclusive'}
        </Badge>
        {partition.temperature !== undefined ? (
          <Badge size="sm" tone="outline" className="font-mono">
            temperature {partition.temperature}
          </Badge>
        ) : null}
      </>
    )
    body = (
      <>
        <SectionLabel>Members</SectionLabel>
        <div className="flex flex-wrap items-center gap-1.5">
          {(partition.members ?? []).map((member) => (
            <SignalChip key={member} type="domain" name={member} />
          ))}
          <span className="ml-1 text-xs text-muted-foreground">
            falls back to{' '}
            <span className="font-mono text-foreground/85">{partition.default || '—'}</span>
          </span>
        </div>
      </>
    )
  } else if (kind === 'score') {
    const score = entity as ProjectionScore
    const mappings = scoreMappings(config, score.name)
    badges = (
      <>
        <Badge size="sm" tone="teal">
          Score · {score.method?.replace(/_/g, ' ') || 'weighted sum'}
        </Badge>
        <Badge size="sm" tone="outline">
          {score.inputs?.length ?? 0} inputs
        </Badge>
      </>
    )
    body = (
      <>
        <SectionLabel>Weights</SectionLabel>
        <WeightBars inputs={score.inputs ?? []} compact />
        {mappings.map((mapping) => (
          <div key={mapping.name} className="mt-2 flex flex-col gap-1.5">
            <SectionLabel>{mapping.name} · bands</SectionLabel>
            <BandStrip outputs={mapping.outputs ?? []} method={mapping.method} />
          </div>
        ))}
        {mappings.length === 0 ? (
          <p className="mt-1 text-xs text-muted-foreground">
            No mapping turns this score into bands yet.
          </p>
        ) : null}
      </>
    )
  } else {
    const mapping = entity as ProjectionMapping
    badges = (
      <>
        <Badge size="sm" tone="teal">
          Mapping · {mapping.method?.replace(/_/g, ' ') || 'threshold bands'}
        </Badge>
        {mapping.calibration?.method ? (
          <Badge size="sm" tone="outline" className="font-mono">
            {mapping.calibration.method}
          </Badge>
        ) : null}
      </>
    )
    body = (
      <>
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          from <SignalChip type="projection" name={mapping.source} />
        </div>
        <SectionLabel className="mt-1">Bands</SectionLabel>
        <BandStrip outputs={mapping.outputs ?? []} method={mapping.method} />
      </>
    )
  }
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: Math.min(index, 8) * 0.04, duration: 0.4, ease }}
      className="h-full"
    >
      <Card
        interactive
        role="button"
        tabIndex={0}
        aria-label={`Open ${entity.name}`}
        onClick={() => onOpen()}
        onKeyDown={(event) => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault()
            onOpen()
          }
        }}
        className="flex h-full flex-col overflow-hidden"
      >
        <div className="flex items-start gap-3 px-4 pt-4">
          <IconTile icon={Blend} tone="teal" />
          <div className="flex min-w-0 flex-1 flex-col gap-1.5">
            <span className="truncate font-mono text-[13px] font-semibold">{entity.name}</span>
            <div className="flex flex-wrap items-center gap-1.5">{badges}</div>
          </div>
          <CardMenu
            kind={kind}
            entity={entity}
            readOnly={readOnly}
            onOpen={onOpen}
            onCopy={onCopy}
            onDuplicate={onDuplicate}
            onDelete={onDelete}
          />
        </div>
        <div className="flex flex-col gap-2 px-4 pb-4 pt-4">{body}</div>
        <ReadByFooter names={consumers} />
      </Card>
    </motion.div>
  )
}

function NewCard({
  kind,
  index,
  onCreate,
}: {
  kind: ProjectionKind
  index: number
  onCreate: () => void
}) {
  return (
    <motion.button
      type="button"
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: Math.min(index, 8) * 0.04, duration: 0.4, ease }}
      onClick={onCreate}
      className="group flex min-h-48 flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-border-strong p-6 text-center transition-colors hover:border-solid hover:bg-surface-subtle focus-visible:ring-[3px] focus-visible:ring-ring/40 focus-visible:outline-none"
    >
      <span className="grid size-10 place-items-center rounded-full border border-border bg-surface text-muted-foreground shadow-xs transition-colors group-hover:text-foreground">
        <Plus className="size-4" />
      </span>
      <span className="flex flex-col gap-1">
        <span className="text-[13px] font-medium">
          New {PROJECTION_KIND_LABEL[kind].toLowerCase()}
        </span>
        <span className="max-w-xs text-xs text-muted-foreground">{KIND_HINT[kind]}</span>
      </span>
    </motion.button>
  )
}

export default function ConfigPageProjectionsSection({
  config,
  isReadonly,
  saveConfig,
}: ConfigPageProjectionsSectionProps) {
  const navigate = useNavigate()
  const { copy } = useCopy()
  const { params, update } = useRoutingPageParams()
  const { applyScopedConfig, routingScopes, scopedConfig, selectedScope, selectedScopeId } =
    useRoutingScopeManager(config, params.get('recipe') ?? '')
  const [search, setSearch] = React.useState('')
  const [pendingDelete, setPendingDelete] = React.useState<{
    kind: ProjectionKind
    name: string
  } | null>(null)
  const [deleting, setDeleting] = React.useState(false)
  const [deleteError, setDeleteError] = React.useState<string | null>(null)
  const [draftOverride, setDraftOverride] = React.useState<{
    kind: ProjectionKind
    draft: ProjectionEntity
  } | null>(null)

  const scopeLabel = routingScopeRecipeName(
    config,
    selectedScopeId,
    selectedScope?.isDefault ?? true,
  )
  const models = React.useMemo(() => getNormalizedModels(config), [config])
  const inventory = React.useMemo(
    () => routingScopeInventory(scopedConfig, models),
    [scopedConfig, models],
  )
  const hints = React.useMemo(() => routingScopeFormHints(inventory, 'teal'), [inventory])
  const lists = React.useMemo<Record<ProjectionKind, ProjectionEntity[]>>(
    () => ({
      partition: scopedConfig?.projections?.partitions ?? [],
      score: scopedConfig?.projections?.scores ?? [],
      mapping: scopedConfig?.projections?.mappings ?? [],
    }),
    [scopedConfig],
  )
  const total = lists.partition.length + lists.score.length + lists.mapping.length

  const activeKey = parseKey(params.get('id'))
  const active = activeKey
    ? lists[activeKey.kind].find((entity) => entity.name === activeKey.name)
    : undefined
  const newKind = params.get('new') as ProjectionKind | null
  const creating = newKind && newKind in PROJECTION_COLLECTION && !isReadonly ? newKind : null
  const tabParam = params.get('tab') as ProjectionTab | null
  const firstFilled =
    (['partitions', 'scores', 'mappings'] as const).find(
      (candidate) => lists[TAB_KIND[candidate]].length > 0,
    ) ?? 'partitions'
  const tab: ProjectionTab =
    tabParam && tabParam in TAB_KIND
      ? tabParam
      : activeKey
        ? KIND_TAB[activeKey.kind]
        : creating
          ? KIND_TAB[creating]
          : firstFilled
  const tabKind = TAB_KIND[tab]
  const viewParam = params.get('view') as ProjectionSheetView | null
  const view: ProjectionSheetView =
    viewParam === 'try' || viewParam === 'yaml' ? viewParam : 'overview'
  const sheetTarget = React.useMemo<ProjectionSheetTarget | null>(() => {
    if (active && activeKey)
      return {
        kind: activeKey.kind,
        mode: 'edit',
        original: active,
        draft: JSON.parse(JSON.stringify(active)),
      }
    if (creating) {
      const draft =
        draftOverride?.kind === creating
          ? draftOverride.draft
          : newProjectionDraft(
              creating,
              lists.score.map((score) => score.name),
            )
      return { kind: creating, mode: 'create', draft }
    }
    return null
  }, [active, activeKey, creating, draftOverride, lists])
  const shownTarget = React.useRef<ProjectionSheetTarget | null>(null)
  if (sheetTarget) shownTarget.current = sheetTarget

  const query = search.trim().toLowerCase()
  const visible = lists[tabKind].filter(
    (entity) => !query || JSON.stringify(entity).toLowerCase().includes(query),
  )

  const open = (kind: ProjectionKind, name: string, nextView?: ProjectionSheetView) =>
    update({ id: `${kind}:${name}`, new: null, tab: KIND_TAB[kind], view: nextView ?? null })
  const closeSheet = () => {
    setDraftOverride(null)
    update({ id: null, new: null, view: null })
  }
  const startCreate = (kind: ProjectionKind, draft?: ProjectionEntity) => {
    setDraftOverride(draft ? { kind, draft } : null)
    update({ new: kind, id: null, view: null, tab: KIND_TAB[kind] })
  }
  const copyYaml = (kind: ProjectionKind, entity: ProjectionEntity) => {
    copy(
      configCollectionItemYaml(
        ['projections', PROJECTION_COLLECTION[kind]],
        entity,
        PROJECTION_SCHEMA_REF[kind],
      ),
    )
    toast.success(`${entity.name} YAML copied`)
  }
  const duplicate = (kind: ProjectionKind, entity: ProjectionEntity) => {
    const draft = JSON.parse(JSON.stringify(entity)) as ProjectionEntity
    draft.name = copyName(
      lists[kind].map((entry) => entry.name),
      entity.name,
    )
    startCreate(kind, draft)
  }
  const save = async (kind: ProjectionKind, entity: ProjectionEntity, originalName?: string) => {
    if (!scopedConfig) throw new Error('Routing profile not loaded yet.')
    await saveConfig(applyScopedConfig(upsertProjection(scopedConfig, kind, entity, originalName)))
    setDraftOverride(null)
    update({ id: `${kind}:${entity.name}`, new: null }, { replace: true })
  }
  const confirmDelete = async () => {
    if (!scopedConfig || !pendingDelete) return
    setDeleting(true)
    setDeleteError(null)
    try {
      await saveConfig(
        applyScopedConfig(removeProjection(scopedConfig, pendingDelete.kind, pendingDelete.name)),
      )
      toast.success(`${pendingDelete.name} deleted`)
      if (
        activeKey &&
        activeKey.kind === pendingDelete.kind &&
        activeKey.name === pendingDelete.name
      )
        closeSheet()
      setPendingDelete(null)
    } catch (failure) {
      setDeleteError(
        failure instanceof Error ? failure.message : 'The projection could not be deleted.',
      )
    } finally {
      setDeleting(false)
    }
  }
  const deleteConsumers = pendingDelete
    ? (() => {
        const entity = lists[pendingDelete.kind].find((entry) => entry.name === pendingDelete.name)
        return entity
          ? projectionDecisionConsumers(scopedConfig, pendingDelete.kind, entity).map(
              (consumer) => consumer.decision.name,
            )
          : []
      })()
    : []
  const dependentMappings =
    pendingDelete?.kind === 'score'
      ? scoreMappings(scopedConfig, pendingDelete.name).map((mapping) => mapping.name)
      : []

  const newMenu = (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button>
          <Plus />
          New projection
          <ChevronDown className="-mr-1 opacity-70" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-72">
        <DropdownMenuLabel>Projection type</DropdownMenuLabel>
        {(['partition', 'score', 'mapping'] as const).map((kind) => (
          <DropdownMenuItem key={kind} className="items-start" onSelect={() => startCreate(kind)}>
            <Blend className="mt-0.5 text-chart-4" />
            <span className="flex flex-col gap-0.5">
              <span className="font-medium">{PROJECTION_KIND_LABEL[kind]}</span>
              <span className="text-xs leading-snug text-muted-foreground">{KIND_HINT[kind]}</span>
            </span>
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )

  return (
    <SchemaFormHintsProvider value={hints}>
      <ConfigPageManagerLayout
        title="Projections"
        description="Combine several signals into one derived value, such as the winning domain or a difficulty band, that decisions can match."
        icon={<IconTile icon={Blend} tone="teal" size="lg" />}
        actions={
          <>
            <RoutingScopeSelector
              label="Recipe"
              scopes={routingScopes}
              value={selectedScopeId}
              onChange={(scopeId) => update({ recipe: scopeId, id: null, new: null, view: null })}
            />
            {isReadonly ? null : newMenu}
          </>
        }
        tabs={
          <Tabs
            value={tab}
            onValueChange={(value) => update({ tab: value }, { replace: true })}
            variant="line"
            className="gap-0"
          >
            <TabsList>
              {(['partitions', 'scores', 'mappings'] as const).map((value) => (
                <TabsTrigger key={value} value={value}>
                  {PROJECTION_KIND_LABEL[TAB_KIND[value]]}s
                  <span className="rounded-full bg-secondary px-1.5 font-mono text-[11px] tabular text-muted-foreground">
                    {lists[TAB_KIND[value]].length}
                  </span>
                </TabsTrigger>
              ))}
            </TabsList>
          </Tabs>
        }
      >
        {total === 0 ? (
          <div className="rounded-xl border border-dashed border-border-strong">
            <EmptyState
              icon={Blend}
              tone="teal"
              title={`${scopeLabel} has no projections`}
              description="Projections are optional. Add one when several signals should become one value a decision can match, like a difficulty band."
              actions={isReadonly ? undefined : newMenu}
            />
          </div>
        ) : (
          <>
            <DataToolbar
              search={search}
              onSearchChange={setSearch}
              placeholder={`Search ${tab}…`}
              actions={
                <span className="text-xs text-muted-foreground">
                  <span className="tabular">{visible.length}</span> of{' '}
                  <span className="tabular">{lists[tabKind].length}</span> {tab}
                </span>
              }
            />
            {visible.length === 0 && lists[tabKind].length > 0 ? (
              <EmptyState
                compact
                icon={Blend}
                tone="teal"
                title={`No ${tab} match`}
                description="Try another search."
                actions={
                  <Button variant="outline" size="sm" onClick={() => setSearch('')}>
                    Clear search
                  </Button>
                }
              />
            ) : (
              <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
                {visible.map((entity, index) => (
                  <ProjectionCard
                    key={entity.name}
                    kind={tabKind}
                    entity={entity}
                    index={index}
                    config={scopedConfig}
                    readOnly={isReadonly}
                    onOpen={(nextView) => open(tabKind, entity.name, nextView)}
                    onCopy={() => copyYaml(tabKind, entity)}
                    onDuplicate={() => duplicate(tabKind, entity)}
                    onDelete={() => {
                      setDeleteError(null)
                      setPendingDelete({ kind: tabKind, name: entity.name })
                    }}
                  />
                ))}
                {isReadonly || query ? null : (
                  <NewCard
                    kind={tabKind}
                    index={visible.length}
                    onCreate={() => startCreate(tabKind)}
                  />
                )}
              </div>
            )}
          </>
        )}
      </ConfigPageManagerLayout>

      <ConfigPageProjectionSheet
        target={sheetTarget ?? shownTarget.current}
        open={Boolean(sheetTarget)}
        view={view}
        onViewChange={(next) =>
          update({ view: next === 'overview' ? null : next }, { replace: true })
        }
        onOpenChange={(isOpen) => (isOpen ? null : closeSheet())}
        scopeLabel={scopeLabel}
        scopedConfig={scopedConfig}
        readOnly={isReadonly}
        entrypointModel={selectedScope?.entrypointModelNames[0]}
        onSave={save}
        onDelete={(kind, name) => {
          setDeleteError(null)
          setPendingDelete({ kind, name })
        }}
        onDuplicate={duplicate}
        onOpenDecision={(name) =>
          navigate(
            `/routing/decisions?${new URLSearchParams({ recipe: selectedScopeId, id: name })}`,
          )
        }
        onOpenProjection={(kind, name) => open(kind, name)}
        onCreateMapping={(source) =>
          startCreate('mapping', { name: '', source, method: 'threshold_bands', outputs: [] })
        }
      />

      <ConfirmDialog
        isOpen={pendingDelete !== null}
        title={`Delete ${pendingDelete?.name ?? ''}?`}
        description={
          deleteConsumers.length || dependentMappings.length
            ? `${[
                deleteConsumers.length
                  ? `${deleteConsumers.length} ${deleteConsumers.length === 1 ? 'decision reads' : 'decisions read'} it (${deleteConsumers.join(', ')})`
                  : '',
                dependentMappings.length
                  ? `${dependentMappings.length} ${dependentMappings.length === 1 ? 'mapping uses' : 'mappings use'} it as a source (${dependentMappings.join(', ')})`
                  : '',
              ]
                .filter(Boolean)
                .join(' and ')}. Update them, or the Router rejects the configuration.`
            : `Nothing in ${scopeLabel} reads this ${pendingDelete ? PROJECTION_KIND_LABEL[pendingDelete.kind].toLowerCase() : 'projection'}. Deleting removes it from config.yaml.`
        }
        confirmLabel={`Delete ${pendingDelete ? PROJECTION_KIND_LABEL[pendingDelete.kind].toLowerCase() : 'projection'}`}
        pending={deleting}
        errorMessage={deleteError ?? undefined}
        onCancel={() => {
          if (!deleting) {
            setPendingDelete(null)
            setDeleteError(null)
          }
        }}
        onConfirm={confirmDelete}
      />
    </SchemaFormHintsProvider>
  )
}
