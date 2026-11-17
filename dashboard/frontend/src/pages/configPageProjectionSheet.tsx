import * as React from 'react'
import {
  ArrowUpRight,
  Blend,
  CopyPlus,
  FileDown,
  MoreHorizontal,
  Plus,
  Save,
  Split,
  Trash2,
} from 'lucide-react'

import ConfirmDialog from '../components/ConfirmDialog'
import { configCollectionItemYaml } from '../lib/configYaml'
import { getProjectionFieldSchema } from '../lib/projectionSchemas'
import type { FieldSchema } from '../lib/dslSchemas'
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
  Field,
  IconTile,
  Input,
  MiniStat,
  SectionLabel,
  Slider,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
  toast,
  useCopy,
} from '../ui'
import ConfigPageSchemaFieldsEditor from './ConfigPageSchemaFieldsEditor'
import { NumberInput, OptionSelect } from './ConfigPageSchemaFieldsEditorControls'
import { EntityYamlPanel, SignalChip, TryPromptPanel } from './configPageRoutingScopeViews'
import { projectionFromDraft, type ProjectionEntity } from './configPageProjectionDraft'
import {
  BandStrip,
  ProjectionInputsEditor,
  ProjectionMembersEditor,
  ProjectionOutputsEditor,
  WeightBars,
} from './configPageProjectionStructuredEditors'
import {
  PROJECTION_COLLECTION,
  PROJECTION_KIND_LABEL,
  PROJECTION_SCHEMA_REF,
  projectionDecisionConsumers,
  scoreMappings,
  scoresReadingProjection,
  type ProjectionKind,
} from './configPageProjectionTableSupport'
import type {
  ConfigData,
  ProjectionMapping,
  ProjectionPartition,
  ProjectionScore,
} from './configPageSupport'

export type ProjectionSheetView = 'overview' | 'try' | 'yaml'

export interface ProjectionSheetTarget {
  kind: ProjectionKind
  mode: 'create' | 'edit'
  original?: ProjectionEntity
  draft: ProjectionEntity
}

function schemaField(kind: ProjectionKind, key: string): FieldSchema | undefined {
  return getProjectionFieldSchema(PROJECTION_COLLECTION[kind]).find((field) => field.key === key)
}

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T
}

function KindBadge({ kind, entity }: { kind: ProjectionKind; entity: ProjectionEntity }) {
  const detail =
    kind === 'partition'
      ? (entity as ProjectionPartition).semantics
      : kind === 'score'
        ? (entity as ProjectionScore).method
        : (entity as ProjectionMapping).method
  return (
    <Badge tone="teal">
      {PROJECTION_KIND_LABEL[kind]}
      {detail ? ` · ${detail.replace(/_/g, ' ')}` : ''}
    </Badge>
  )
}

function ReadBy({
  rows,
  onOpenDecision,
}: {
  rows: Array<{ decision: { name: string; priority?: number }; via: string }>
  onOpenDecision: (name: string) => void
}) {
  return (
    <section className="flex flex-col gap-2">
      <SectionLabel>Read by</SectionLabel>
      {rows.length === 0 ? (
        <p className="rounded-lg border border-dashed border-border-strong px-3 py-3 text-xs text-muted-foreground">
          No decision reads this projection yet. Reference its bands in a decision&apos;s rules with
          the projection signal type.
        </p>
      ) : (
        rows.map(({ decision, via }) => (
          <button
            key={decision.name}
            type="button"
            onClick={() => onOpenDecision(decision.name)}
            className="group flex items-center gap-3 rounded-lg border border-border bg-surface px-3 py-2.5 text-left transition-colors hover:border-border-strong hover:bg-surface-subtle focus-visible:ring-[3px] focus-visible:ring-ring/40 focus-visible:outline-none"
          >
            <IconTile icon={Split} tone="violet" size="sm" />
            <span className="flex min-w-0 flex-1 flex-col">
              <span className="truncate font-mono text-[13px] font-medium">{decision.name}</span>
              <span className="truncate font-mono text-[11.5px] text-subtle-foreground">{via}</span>
            </span>
            <span className="rounded-md bg-secondary px-1.5 py-0.5 font-mono text-xs tabular">
              P{decision.priority ?? 0}
            </span>
            <ArrowUpRight className="size-4 text-subtle-foreground transition-colors group-hover:text-foreground" />
          </button>
        ))
      )}
    </section>
  )
}

function Formula({
  kind,
  entity,
  mappings,
}: {
  kind: ProjectionKind
  entity: ProjectionEntity
  mappings: ProjectionMapping[]
}) {
  const box =
    'flex flex-col gap-2 rounded-lg border border-border bg-surface-sunken/60 p-3.5 font-mono text-[12.5px] leading-relaxed'
  if (kind === 'partition') {
    const partition = entity as ProjectionPartition
    return (
      <div className={box}>
        <div>
          <span className="text-chart-4">{partition.name || 'partition'}</span>{' '}
          <span className="text-subtle-foreground">=</span> argmax{' '}
          <span className="text-subtle-foreground">{'{'}</span>
        </div>
        <div className="flex flex-wrap gap-1.5 pl-5">
          {(partition.members ?? []).map((member) => (
            <SignalChip key={member} type="domain" name={member} />
          ))}
        </div>
        <div className="text-subtle-foreground">{'}'}</div>
        {partition.semantics === 'softmax_exclusive' ? (
          <div className="text-muted-foreground">
            confidence <span className="text-subtle-foreground">=</span> softmax(score{' '}
            <span className="text-subtle-foreground">/</span>{' '}
            <span className="text-foreground">{partition.temperature ?? '?'}</span>)
          </div>
        ) : null}
        <div className="text-muted-foreground">
          when no member matches <span className="text-subtle-foreground">→</span>{' '}
          <span className="text-foreground">{partition.default || '?'}</span>
        </div>
      </div>
    )
  }
  if (kind === 'score') {
    const score = entity as ProjectionScore
    return (
      <div className={box}>
        <div>
          <span className="text-chart-4">{score.name || 'score'}</span>{' '}
          <span className="text-subtle-foreground">= Σ weight ×</span>{' '}
          <span className="text-muted-foreground">value</span>
        </div>
        {mappings.map((mapping) => (
          <div key={mapping.name} className="border-t border-border pt-2 text-muted-foreground">
            <span className="text-foreground">{mapping.name}</span>{' '}
            <span className="text-subtle-foreground">←</span>{' '}
            {(mapping.outputs ?? []).map((output) => output.name).join(' | ')}
          </div>
        ))}
      </div>
    )
  }
  const mapping = entity as ProjectionMapping
  return (
    <div className={box}>
      <div>
        <span className="text-chart-4">{mapping.name || 'mapping'}</span>{' '}
        <span className="text-subtle-foreground">←</span>{' '}
        <span className="text-foreground">{mapping.source || '?'}</span>{' '}
        <span className="text-subtle-foreground">({mapping.method?.replace(/_/g, ' ')})</span>
      </div>
      {mapping.calibration?.method ? (
        <div className="text-muted-foreground">
          confidence <span className="text-subtle-foreground">=</span> {mapping.calibration.method}
          (slope {mapping.calibration.slope ?? 'default'})
        </div>
      ) : null}
    </div>
  )
}

interface ConfigPageProjectionSheetProps {
  target: ProjectionSheetTarget | null
  open: boolean
  view: ProjectionSheetView
  onViewChange: (view: ProjectionSheetView) => void
  onOpenChange: (open: boolean) => void
  scopeLabel: string
  scopedConfig: ConfigData | null
  readOnly: boolean
  entrypointModel?: string
  onSave: (kind: ProjectionKind, entity: ProjectionEntity, originalName?: string) => Promise<void>
  onDelete: (kind: ProjectionKind, name: string) => void
  onDuplicate: (kind: ProjectionKind, entity: ProjectionEntity) => void
  onOpenDecision: (name: string) => void
  onOpenProjection: (kind: ProjectionKind, name: string) => void
  onCreateMapping: (source: string) => void
}

export default function ConfigPageProjectionSheet({
  target,
  open,
  view,
  onViewChange,
  onOpenChange,
  scopeLabel,
  scopedConfig,
  readOnly,
  entrypointModel,
  onSave,
  onDelete,
  onDuplicate,
  onOpenDecision,
  onOpenProjection,
  onCreateMapping,
}: ConfigPageProjectionSheetProps) {
  const { copy } = useCopy()
  const [draft, setDraft] = React.useState<ProjectionEntity | null>(
    target ? clone(target.draft) : null,
  )
  const [saving, setSaving] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)
  const [confirmDiscard, setConfirmDiscard] = React.useState(false)
  const targetKey = target
    ? `${target.kind}:${target.mode}:${target.original?.name ?? ''}:${target.mode === 'create' ? JSON.stringify(target.draft) : ''}`
    : ''
  React.useEffect(() => {
    setDraft(target ? clone(target.draft) : null)
    setError(null)
    setSaving(false)
    // Reset only when a different projection opens, not when the saved config refreshes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [targetKey])

  if (!target || !draft) return null
  const { kind, original } = target
  const dirty = JSON.stringify(draft) !== JSON.stringify(target.draft)
  const collection = PROJECTION_COLLECTION[kind]
  const preview = (() => {
    try {
      return projectionFromDraft(kind, draft)
    } catch {
      return draft
    }
  })()
  const yaml = configCollectionItemYaml(
    ['projections', collection],
    preview,
    PROJECTION_SCHEMA_REF[kind],
  )
  const savedYaml = original
    ? configCollectionItemYaml(['projections', collection], original, PROJECTION_SCHEMA_REF[kind])
    : undefined
  const consumers = projectionDecisionConsumers(scopedConfig, kind, draft)
  const mappings = kind === 'score' && original ? scoreMappings(scopedConfig, original.name) : []
  const readers =
    kind !== 'partition' && original ? scoresReadingProjection(scopedConfig, original.name) : []
  const scoreNames = (scopedConfig?.projections?.scores ?? []).map((score) => score.name)
  const title = draft.name.trim() || `New ${PROJECTION_KIND_LABEL[kind].toLowerCase()}`
  const set = (patch: Record<string, unknown>) => {
    const next = { ...draft } as Record<string, unknown>
    for (const [key, value] of Object.entries(patch)) {
      if (value === undefined || value === '') delete next[key]
      else next[key] = value
    }
    setDraft(next as unknown as ProjectionEntity)
  }
  const trySignals =
    kind === 'partition'
      ? ((draft as ProjectionPartition).members ?? []).map((name) => ({ type: 'domain', name }))
      : (kind === 'mapping' ? [draft as ProjectionMapping] : mappings).flatMap((mapping) =>
          (mapping.outputs ?? []).map((output) => ({ type: 'projection', name: output.name })),
        )

  const requestClose = () => {
    if (dirty && !readOnly) setConfirmDiscard(true)
    else onOpenChange(false)
  }
  const save = async () => {
    let entity: ProjectionEntity
    try {
      entity = projectionFromDraft(kind, draft)
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure))
      onViewChange('overview')
      return
    }
    setSaving(true)
    setError(null)
    try {
      await onSave(kind, entity, original?.name)
      toast.success(`${entity.name} saved`, { description: `Saved in recipe ${scopeLabel}.` })
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : 'The projection could not be saved.')
      onViewChange('overview')
    } finally {
      setSaving(false)
    }
  }

  const nameField = (
    <Field
      label="Name"
      htmlFor="projection-name"
      hint={kind === 'mapping' ? 'Decisions match the bands below, not this name.' : undefined}
    >
      <Input
        id="projection-name"
        value={draft.name}
        disabled={readOnly}
        placeholder={
          kind === 'partition'
            ? 'practice_areas'
            : kind === 'score'
              ? 'request_difficulty'
              : 'request_band'
        }
        className="font-mono text-[12.5px]"
        aria-invalid={!draft.name.trim() || undefined}
        onChange={(event) => setDraft({ ...draft, name: event.target.value })}
      />
    </Field>
  )

  let editor: React.ReactNode
  if (kind === 'partition') {
    const partition = draft as ProjectionPartition
    const semantics = schemaField('partition', 'semantics')
    const temperature = partition.temperature
    editor = (
      <>
        <section className="flex flex-col gap-2">
          <SectionLabel>Formula</SectionLabel>
          <Formula kind={kind} entity={draft} mappings={[]} />
        </section>
        <section className="flex flex-col gap-4">
          <SectionLabel>Definition</SectionLabel>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            {nameField}
            <Field label={semantics?.label ?? 'Semantics'} hint={semantics?.description}>
              <OptionSelect
                ariaLabel="Semantics"
                value={partition.semantics}
                options={semantics?.options ?? []}
                required
                disabled={readOnly}
                onChange={(next) => set({ semantics: next })}
              />
            </Field>
          </div>
          {partition.semantics === 'softmax_exclusive' || temperature !== undefined ? (
            <div className="flex flex-col gap-2.5 rounded-lg border border-border bg-surface-subtle/60 px-3 py-3">
              <div className="flex items-center justify-between gap-3">
                <span className="text-[13px] font-medium">
                  {schemaField('partition', 'temperature')?.label ?? 'Temperature'}
                  <span className="ml-2 font-mono text-[11px] font-normal text-subtle-foreground">
                    temperature
                  </span>
                </span>
                <NumberInput
                  ariaLabel="Temperature value"
                  value={temperature}
                  min={0}
                  disabled={readOnly}
                  placeholder="unset"
                  className="h-8 w-20 text-center"
                  onChange={(next) => set({ temperature: next })}
                />
              </div>
              <Slider
                aria-label="Temperature"
                value={temperature ?? 0}
                min={0}
                max={Math.max(2, temperature ?? 0)}
                step={0.05}
                tone="teal"
                disabled={readOnly}
                onValueChange={(next) => set({ temperature: Math.round(next * 100) / 100 })}
              />
              <span className="text-xs text-muted-foreground">
                {schemaField('partition', 'temperature')?.description}
              </span>
            </div>
          ) : null}
          <Field
            label={schemaField('partition', 'members')?.label ?? 'Members'}
            hint={schemaField('partition', 'members')?.description}
          >
            <ProjectionMembersEditor
              value={partition.members}
              readOnly={readOnly}
              onChange={(members) => set({ members })}
            />
          </Field>
          <Field
            label={schemaField('partition', 'default')?.label ?? 'Default'}
            hint={schemaField('partition', 'default')?.description}
          >
            <OptionSelect
              ariaLabel="Default member"
              value={partition.default}
              options={partition.members ?? []}
              required
              disabled={readOnly}
              onChange={(next) => set({ default: next })}
            />
          </Field>
        </section>
      </>
    )
  } else if (kind === 'score') {
    const score = draft as ProjectionScore
    const method = schemaField('score', 'method')
    editor = (
      <>
        <section className="flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <SectionLabel>Weights</SectionLabel>
            <span className="text-xs text-subtle-foreground">updates as you edit the inputs</span>
          </div>
          <WeightBars inputs={score.inputs ?? []} />
        </section>
        <section className="flex flex-col gap-4">
          <SectionLabel>Definition</SectionLabel>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            {nameField}
            <Field label={method?.label ?? 'Method'} hint={method?.description}>
              <OptionSelect
                ariaLabel="Method"
                value={score.method}
                options={method?.options ?? []}
                required
                disabled={readOnly}
                onChange={(next) => set({ method: next })}
              />
            </Field>
          </div>
          <Field
            label={schemaField('score', 'inputs')?.label ?? 'Inputs'}
            hint={schemaField('score', 'inputs')?.description}
          >
            <ProjectionInputsEditor
              value={score.inputs}
              readOnly={readOnly}
              projectionNames={scoreNames.filter((name) => name !== score.name)}
              onChange={(inputs) => set({ inputs })}
            />
          </Field>
        </section>
        <section className="flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <SectionLabel>Bands</SectionLabel>
            {!readOnly && original ? (
              <Button variant="ghost" size="xs" onClick={() => onCreateMapping(original.name)}>
                <Plus />
                Map into bands
              </Button>
            ) : null}
          </div>
          {mappings.length === 0 ? (
            <p className="rounded-lg border border-dashed border-border-strong px-3 py-3 text-xs text-muted-foreground">
              No mapping turns this score into bands yet, so decisions cannot match it.
            </p>
          ) : (
            mappings.map((mapping) => (
              <button
                key={mapping.name}
                type="button"
                onClick={() => onOpenProjection('mapping', mapping.name)}
                className="flex flex-col gap-2 rounded-lg border border-border bg-surface px-3 py-2.5 text-left transition-colors hover:border-border-strong focus-visible:ring-[3px] focus-visible:ring-ring/40 focus-visible:outline-none"
              >
                <span className="flex items-center justify-between gap-2">
                  <span className="font-mono text-[12.5px] font-medium">{mapping.name}</span>
                  <Badge size="sm" tone="outline">
                    {mapping.method?.replace(/_/g, ' ')}
                  </Badge>
                </span>
                <BandStrip outputs={mapping.outputs ?? []} method={mapping.method} />
              </button>
            ))
          )}
        </section>
      </>
    )
  } else {
    const mapping = draft as ProjectionMapping
    const method = schemaField('mapping', 'method')
    const calibration = schemaField('mapping', 'calibration')
    editor = (
      <>
        <section className="flex flex-col gap-2">
          <SectionLabel>Bands</SectionLabel>
          <BandStrip outputs={mapping.outputs ?? []} method={mapping.method} />
        </section>
        <section className="flex flex-col gap-4">
          <SectionLabel>Definition</SectionLabel>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            {nameField}
            <Field
              label={schemaField('mapping', 'source')?.label ?? 'Source'}
              hint={schemaField('mapping', 'source')?.description}
            >
              <OptionSelect
                ariaLabel="Source score"
                value={mapping.source}
                options={scoreNames}
                required
                disabled={readOnly}
                onChange={(next) => set({ source: next })}
              />
            </Field>
            <Field
              label={method?.label ?? 'Method'}
              hint={method?.description}
              className="sm:col-span-2"
            >
              <OptionSelect
                ariaLabel="Method"
                value={mapping.method}
                options={method?.options ?? []}
                required
                disabled={readOnly}
                onChange={(next) => set({ method: next })}
              />
            </Field>
          </div>
          {calibration ? (
            <ConfigPageSchemaFieldsEditor
              schema={[calibration]}
              value={mapping as unknown as Record<string, unknown>}
              readOnly={readOnly}
              hideAdditional
              showAllByDefault
              path={['projections', 'mappings']}
              onChange={(next) => setDraft(next as unknown as ProjectionEntity)}
            />
          ) : null}
          <Field
            label={schemaField('mapping', 'outputs')?.label ?? 'Outputs'}
            hint={schemaField('mapping', 'outputs')?.description}
          >
            <ProjectionOutputsEditor
              value={mapping.outputs}
              readOnly={readOnly}
              onChange={(outputs) => set({ outputs })}
            />
          </Field>
        </section>
      </>
    )
  }

  const statItems: Array<{ label: string; value: React.ReactNode; hint?: string }> =
    kind === 'partition'
      ? [
          {
            label: 'Members',
            value: (draft as ProjectionPartition).members?.length ?? 0,
            hint: (draft as ProjectionPartition).default
              ? `default ${(draft as ProjectionPartition).default}`
              : undefined,
          },
          {
            label: 'Semantics',
            value: (
              <span className="text-[15px]">
                {(draft as ProjectionPartition).semantics?.replace(/_/g, ' ') || '—'}
              </span>
            ),
          },
          {
            label: 'Read by',
            value: consumers.length,
            hint: consumers.length === 1 ? 'decision' : 'decisions',
          },
        ]
      : kind === 'score'
        ? [
            {
              label: 'Inputs',
              value: (draft as ProjectionScore).inputs?.length ?? 0,
              hint: `${((draft as ProjectionScore).inputs ?? []).filter((input) => input.weight < 0).length} negative`,
            },
            {
              label: 'Mappings',
              value: mappings.length,
              hint: readers.length ? `${readers.length} scores read it` : undefined,
            },
            {
              label: 'Read by',
              value: consumers.length,
              hint: consumers.length === 1 ? 'decision' : 'decisions',
            },
          ]
        : [
            { label: 'Bands', value: (draft as ProjectionMapping).outputs?.length ?? 0 },
            {
              label: 'Source',
              value: (
                <span className="block truncate font-mono text-[14px]">
                  {(draft as ProjectionMapping).source || '—'}
                </span>
              ),
            },
            {
              label: 'Read by',
              value: consumers.length,
              hint: consumers.length === 1 ? 'decision' : 'decisions',
            },
          ]

  return (
    <>
      <DetailSheet
        open={open}
        onOpenChange={(next) => (next ? onOpenChange(true) : requestClose())}
      >
        <DetailSheetHeader
          icon={<IconTile icon={Blend} tone="teal" size="lg" />}
          title={<span className="font-mono">{title}</span>}
          badges={
            <>
              <KindBadge kind={kind} entity={draft} />
              {target.mode === 'create' ? <Badge tone="info">New</Badge> : null}
            </>
          }
          description={
            <span>
              recipe <span className="font-mono">{scopeLabel}</span> ·{' '}
              <span className="font-mono">projections.{collection}</span>
            </span>
          }
          actions={
            original ? (
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="ghost" size="icon-sm" aria-label="Projection actions">
                    <MoreHorizontal />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="w-44">
                  {readOnly ? null : (
                    <DropdownMenuItem onSelect={() => onDuplicate(kind, original)}>
                      <CopyPlus />
                      Duplicate
                    </DropdownMenuItem>
                  )}
                  <DropdownMenuItem
                    onSelect={() => {
                      copy(savedYaml ?? yaml)
                      toast.success('Projection YAML copied')
                    }}
                  >
                    <FileDown />
                    Copy YAML
                  </DropdownMenuItem>
                  {readOnly ? null : (
                    <>
                      <DropdownMenuSeparator />
                      <DropdownMenuItem
                        variant="destructive"
                        onSelect={() => onDelete(kind, original.name)}
                      >
                        <Trash2 />
                        Delete
                      </DropdownMenuItem>
                    </>
                  )}
                </DropdownMenuContent>
              </DropdownMenu>
            ) : null
          }
        />
        <Tabs
          value={view}
          onValueChange={(next) => onViewChange(next as ProjectionSheetView)}
          className="min-h-0 flex-1 gap-0"
        >
          <TabsList className="px-5">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="try">Try it</TabsTrigger>
            <TabsTrigger value="yaml">YAML</TabsTrigger>
          </TabsList>
          <DetailSheetBody>
            <TabsContent value="overview" className="flex flex-col gap-5 px-5 py-5">
              {error ? (
                <Alert tone="danger" title="The projection was not saved">
                  {error}
                </Alert>
              ) : null}
              <div className="grid grid-cols-3 gap-3">
                {statItems.map((item) => (
                  <MiniStat
                    key={item.label}
                    label={item.label}
                    value={item.value}
                    hint={item.hint}
                  />
                ))}
              </div>
              {editor}
              {readers.length ? (
                <section className="flex flex-col gap-2">
                  <SectionLabel>Scores that read it</SectionLabel>
                  <div className="flex flex-wrap gap-1.5">
                    {readers.map((reader) => (
                      <button
                        key={reader.name}
                        type="button"
                        onClick={() => onOpenProjection('score', reader.name)}
                      >
                        <SignalChip type="projection" name={reader.name} />
                      </button>
                    ))}
                  </div>
                </section>
              ) : null}
              <ReadBy rows={consumers} onOpenDecision={onOpenDecision} />
            </TabsContent>
            <TabsContent value="try" className="px-5 py-5">
              {original && entrypointModel ? (
                <TryPromptPanel
                  entrypointModel={entrypointModel}
                  dirty={dirty}
                  target={{ signals: trySignals }}
                />
              ) : (
                <Alert
                  tone="info"
                  title={
                    original
                      ? 'No chat model name reaches this recipe'
                      : 'Save the projection to try it'
                  }
                >
                  {original
                    ? 'Add an entrypoint that routes a model name to this recipe on the Recipes page, then try prompts here.'
                    : 'The Router evaluates saved projections; save this one, then send it a prompt.'}
                </Alert>
              )}
            </TabsContent>
            <TabsContent value="yaml" className="px-5 py-5">
              <EntityYamlPanel
                path={`recipes[${scopeLabel}].projections.${collection}[${draft.name.trim() || 'new'}]`}
                yaml={yaml}
                savedYaml={savedYaml}
                note="Fields follow the Router schema order; saving writes them to config.yaml."
              />
            </TabsContent>
          </DetailSheetBody>
        </Tabs>
        <DetailSheetFooter highlight={dirty}>
          <span className="min-w-0 truncate text-xs text-muted-foreground">
            {readOnly
              ? 'Read-only: your account can view this configuration.'
              : dirty
                ? 'Unsaved changes'
                : 'The Router validates every change when you save.'}
          </span>
          <div className="ml-auto flex items-center gap-2">
            {readOnly ? (
              <Button variant="ghost" size="sm" onClick={() => onOpenChange(false)}>
                Close
              </Button>
            ) : (
              <>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => (dirty ? setDraft(clone(target.draft)) : onOpenChange(false))}
                >
                  {dirty ? 'Discard' : 'Close'}
                </Button>
                <Button
                  size="sm"
                  onClick={() => void save()}
                  loading={saving}
                  disabled={!dirty && target.mode !== 'create'}
                >
                  {saving ? null : <Save />}
                  {target.mode === 'create'
                    ? `Create ${PROJECTION_KIND_LABEL[kind].toLowerCase()}`
                    : 'Save and apply'}
                </Button>
              </>
            )}
          </div>
        </DetailSheetFooter>
      </DetailSheet>
      <ConfirmDialog
        isOpen={confirmDiscard}
        tone="warning"
        title="Discard unsaved changes?"
        description={`Your edits to ${title} have not been saved.`}
        confirmLabel="Discard changes"
        cancelLabel="Keep editing"
        onCancel={() => setConfirmDiscard(false)}
        onConfirm={() => {
          setConfirmDiscard(false)
          onOpenChange(false)
        }}
      />
    </>
  )
}
