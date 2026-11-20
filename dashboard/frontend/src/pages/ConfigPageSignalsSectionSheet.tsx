import * as React from 'react'
import {
  ArrowUpRight,
  Blend,
  CopyPlus,
  FileDown,
  MoreHorizontal,
  Plus,
  RefreshCw,
  Save,
  Split,
  Trash2,
} from 'lucide-react'

import { getSignalFieldSchema } from '../lib/dslSchemas'
import { configCollectionItemYaml } from '../lib/configYaml'
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
  Field,
  IconTile,
  Input,
  MiniStat,
  SectionLabel,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
  Tooltip,
  useCopy,
  toast,
} from '../ui'
import ConfirmDialog from '../components/ConfirmDialog'
import ConfigPageSchemaFieldsEditor from './ConfigPageSchemaFieldsEditor'
import { OptionSelect } from './ConfigPageSchemaFieldsEditorControls'
import { EntityYamlPanel, TryPromptPanel } from './configPageRoutingScopeViews'
import { ruleExpression } from './configPageDecisionRuleSupport'
import {
  SIGNAL_CATALOG,
  findSignalCatalogEntry,
  signalTypeIcon,
  signalTypeLabel,
} from './configPageSignalCatalog'
import { signalEvaluatorLabel, type SignalEvaluator } from './configPageSignalEvaluator'
import {
  signalDraftErrors,
  signalRecordForSave,
  signalSummary,
  type SignalDraft,
  type SignalRef,
} from './configPageSignalForm'
import type { SignalUsage } from './configPageSignalReferences'
import {
  decisionQuestionTypes,
  signalAvailability,
  signalCapabilitySchema,
  type SignalCapabilityScope,
} from './signalCapabilities'
import type { ConfigData } from './configPageSupport'
import type { DecisionTasks } from './useDecisionTasks'

export type SignalSheetView = 'overview' | 'try' | 'used' | 'yaml'

export interface SignalSheetTarget {
  mode: 'create' | 'edit'
  /** The saved signal, when editing. */
  original?: SignalRef & { record: Record<string, unknown> }
  draft: SignalDraft
}

interface SignalTasks {
  data: DecisionTasks | null
  error: string | null
  refresh: () => void
}

interface ConfigPageSignalSheetProps {
  target: SignalSheetTarget | null
  open: boolean
  view: SignalSheetView
  onViewChange: (view: SignalSheetView) => void
  onOpenChange: (open: boolean) => void
  scopeLabel: string
  scopedConfig: ConfigData | null
  readOnly: boolean
  usage?: SignalUsage
  evaluator?: SignalEvaluator
  tasks: SignalTasks
  capabilityScope: SignalCapabilityScope
  entrypointModel?: string
  onSave: (draft: SignalDraft, original?: SignalRef) => Promise<void>
  onDelete: (signal: SignalRef) => void
  onDuplicate: (signal: SignalRef & { record: Record<string, unknown> }) => void
  onOpenDecision: (name: string) => void
  onOpenProjection: (kind: 'score' | 'partition', name: string) => void
  onUseInDecision: (signal: SignalRef) => void
}

function typeOption(type: string) {
  const Icon = signalTypeIcon(type)
  return (
    <span className="flex items-center gap-2">
      <Icon className="size-3.5 text-chart-2" />
      <span className="font-sans">{signalTypeLabel(type)}</span>
      <span className="font-mono text-[11px] text-subtle-foreground in-data-[slot=select-trigger]:hidden">
        {type}
      </span>
    </span>
  )
}

function UsageRow({
  icon,
  tone,
  title,
  detail,
  badge,
  onOpen,
}: {
  icon: typeof Split
  tone: 'violet' | 'teal' | 'rose'
  title: string
  detail: string
  badge?: React.ReactNode
  onOpen: () => void
}) {
  return (
    <button
      type="button"
      onClick={onOpen}
      className="group flex items-center gap-3 rounded-lg border border-border bg-surface px-3 py-2.5 text-left transition-colors hover:border-border-strong hover:bg-surface-subtle focus-visible:ring-[3px] focus-visible:ring-ring/40 focus-visible:outline-none"
    >
      <IconTile icon={icon} tone={tone} size="sm" />
      <span className="flex min-w-0 flex-1 flex-col">
        <span className="truncate font-mono text-[13px] font-medium">{title}</span>
        <span className="truncate font-mono text-[11.5px] text-subtle-foreground">{detail}</span>
      </span>
      {badge}
      <ArrowUpRight className="size-4 shrink-0 text-subtle-foreground transition-colors group-hover:text-foreground" />
    </button>
  )
}

export default function ConfigPageSignalSheet({
  target,
  open,
  view,
  onViewChange,
  onOpenChange,
  scopeLabel,
  scopedConfig,
  readOnly,
  usage,
  evaluator,
  tasks,
  capabilityScope,
  entrypointModel,
  onSave,
  onDelete,
  onDuplicate,
  onOpenDecision,
  onOpenProjection,
  onUseInDecision,
}: ConfigPageSignalSheetProps) {
  const { copy } = useCopy()
  const [draft, setDraft] = React.useState<SignalDraft | null>(target?.draft ?? null)
  const [saving, setSaving] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)
  const [confirmDiscard, setConfirmDiscard] = React.useState(false)
  const targetKey = target
    ? `${target.mode}:${target.original?.type ?? target.draft.type}:${target.original?.name ?? ''}`
    : ''
  React.useEffect(() => {
    setDraft(target ? JSON.parse(JSON.stringify(target.draft)) : null)
    setError(null)
    setSaving(false)
    // Reset only when a different signal opens, not when the saved config refreshes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [targetKey])

  if (!target || !draft) return null
  const entry = findSignalCatalogEntry(draft.type)
  const Icon = signalTypeIcon(draft.type)
  const original = target.original
  const dirty = JSON.stringify(draft) !== JSON.stringify(target.draft)
  const availability = signalAvailability(tasks.data, capabilityScope, draft.type, draft.fields)
  const baseSchema = getSignalFieldSchema(draft.type)
  const schema =
    draft.type === 'systemone'
      ? signalCapabilitySchema(
          baseSchema,
          decisionQuestionTypes(tasks.data, capabilityScope, draft.fields),
        )
      : baseSchema
  const yamlPath = entry ? ['signals', entry.collection] : ['signals', draft.type]
  const yaml = configCollectionItemYaml(yamlPath, signalRecordForSave(draft), entry?.schemaRef)
  const savedYaml = original
    ? configCollectionItemYaml(yamlPath, original.record, entry?.schemaRef)
    : undefined
  const typeChanged = Boolean(original && original.type !== draft.type)
  const renamed = Boolean(original && original.name !== draft.name.trim() && draft.name.trim())
  const used = usage?.total ?? 0
  const title = draft.name.trim() || `New ${signalTypeLabel(draft.type).toLowerCase()} signal`
  const stage = typeof draft.fields.stage === 'string' ? draft.fields.stage : undefined

  const requestClose = () => {
    if (dirty && !readOnly) setConfirmDiscard(true)
    else onOpenChange(false)
  }

  const save = async () => {
    const problems = signalDraftErrors(scopedConfig, draft, original)
    if (problems.length) {
      setError(problems[0])
      onViewChange('overview')
      return
    }
    if (!availability.supported) {
      setError(availability.reason ?? 'This signal type is not available.')
      onViewChange('overview')
      return
    }
    setSaving(true)
    setError(null)
    try {
      await onSave(draft, original)
      toast.success(`${draft.name.trim()} saved`, { description: `Saved in recipe ${scopeLabel}.` })
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : 'The signal could not be saved.')
    } finally {
      setSaving(false)
    }
  }

  return (
    <>
      <DetailSheet
        open={open}
        onOpenChange={(next) => (next ? onOpenChange(true) : requestClose())}
      >
        <DetailSheetHeader
          icon={<IconTile icon={Icon} tone="gold" size="lg" />}
          title={<span className="font-mono">{title}</span>}
          badges={
            <>
              <Badge tone="gold">{signalTypeLabel(draft.type)}</Badge>
              {stage === 'response' || entry?.decisionReferenceable === false ? (
                <Badge tone="outline">response stage</Badge>
              ) : null}
              {target.mode === 'create' ? <Badge tone="info">New</Badge> : null}
            </>
          }
          description={
            <>
              <span className="max-w-[420px] truncate">{signalSummary({ ...draft.fields })}</span>
              <span className="text-subtle-foreground">·</span>
              <span>
                recipe <span className="font-mono">{scopeLabel}</span>
              </span>
            </>
          }
          actions={
            original ? (
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="ghost" size="icon-sm" aria-label="Signal actions">
                    <MoreHorizontal />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="w-48">
                  {readOnly ? null : (
                    <DropdownMenuItem onSelect={() => onDuplicate(original)}>
                      <CopyPlus />
                      Duplicate
                    </DropdownMenuItem>
                  )}
                  <DropdownMenuItem
                    onSelect={() => {
                      copy(savedYaml ?? yaml)
                      toast.success('Signal YAML copied')
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
                        disabled={used > 0}
                        onSelect={() => onDelete(original)}
                      >
                        <Trash2 />
                        {used > 0 ? 'In use, can’t delete' : 'Delete'}
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
          onValueChange={(next) => onViewChange(next as SignalSheetView)}
          className="min-h-0 flex-1 gap-0"
        >
          <TabsList className="px-5">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="try">Try it</TabsTrigger>
            <TabsTrigger value="used" disabled={!original}>
              Used by
              <Badge size="sm" tone="neutral" className="tabular">
                {used}
              </Badge>
            </TabsTrigger>
            <TabsTrigger value="yaml">YAML</TabsTrigger>
          </TabsList>
          <DetailSheetBody>
            <TabsContent value="overview" className="flex flex-col gap-5 px-5 py-5">
              {error ? (
                <Alert tone="danger" title="The signal was not saved">
                  {error}
                </Alert>
              ) : null}
              <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                <MiniStat
                  label="Type"
                  value={<span className="text-[15px]">{signalTypeLabel(draft.type)}</span>}
                  hint={<span className="font-mono">{draft.type}</span>}
                />
                <MiniStat
                  label="Used by"
                  value={original ? used : '—'}
                  hint={
                    !original
                      ? 'not saved yet'
                      : usage?.decisions.length
                        ? `${usage.decisions.length} ${usage.decisions.length === 1 ? 'decision' : 'decisions'}`
                        : usage?.projections.length
                          ? 'projections only'
                          : usage?.guards.length
                            ? 'response guards'
                            : 'nothing reads it'
                  }
                />
                <MiniStat
                  label={evaluator?.kind === 'model' ? 'Answered by' : 'Evaluated by'}
                  value={
                    <span className="block truncate font-mono text-[14px]">
                      {evaluator ? signalEvaluatorLabel(evaluator) : '—'}
                    </span>
                  }
                  hint={
                    evaluator?.source ?? (evaluator?.kind === 'rule' ? 'no model call' : undefined)
                  }
                />
                <MiniStat
                  label="Stage"
                  value={
                    <span className="text-[15px]">
                      {stage ?? (entry?.decisionReferenceable === false ? 'response' : 'request')}
                    </span>
                  }
                  hint={
                    entry?.decisionReferenceable === false
                      ? 'read by response guards'
                      : 'read by decisions'
                  }
                />
              </div>
              {original && used === 0 ? (
                <Alert
                  tone="neutral"
                  title="Skipped at request time"
                  actions={
                    readOnly || entry?.decisionReferenceable === false ? null : (
                      <Button size="sm" variant="outline" onClick={() => onUseInDecision(original)}>
                        <Plus />
                        Use in a decision
                      </Button>
                    )
                  }
                >
                  The Router evaluates only signals that a decision, projection or response guard
                  reads.
                </Alert>
              ) : null}
              {tasks.error ? (
                <Alert
                  tone="warning"
                  title="Model capabilities could not be loaded"
                  actions={
                    <Button size="sm" variant="outline" onClick={tasks.refresh}>
                      <RefreshCw />
                      Retry
                    </Button>
                  }
                >
                  {tasks.error}
                </Alert>
              ) : !availability.supported && tasks.data ? (
                <Alert tone="warning" title="This signal can't be answered yet">
                  {availability.reason}
                </Alert>
              ) : null}
              <section className="flex flex-col gap-3">
                <SectionLabel>Identity</SectionLabel>
                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                  <Field
                    label="Name"
                    htmlFor="signal-name"
                    hint="Decisions and projections reference the signal by type and name."
                  >
                    <Input
                      id="signal-name"
                      value={draft.name}
                      disabled={readOnly}
                      placeholder="my_signal"
                      className="font-mono text-[12.5px]"
                      aria-invalid={!draft.name.trim() || undefined}
                      onChange={(event) => setDraft({ ...draft, name: event.target.value })}
                    />
                  </Field>
                  <Field
                    label="Type"
                    htmlFor="signal-type"
                    hint={
                      entry ? (
                        <span className="font-mono">signals.{entry.collection}</span>
                      ) : undefined
                    }
                  >
                    <OptionSelect
                      id="signal-type"
                      ariaLabel="Signal type"
                      value={draft.type}
                      required
                      disabled={readOnly}
                      options={SIGNAL_CATALOG.map((candidate) => candidate.type)}
                      disabledOptions={Object.fromEntries(
                        SIGNAL_CATALOG.flatMap((candidate) => {
                          const status = signalAvailability(
                            tasks.data,
                            capabilityScope,
                            candidate.type,
                          )
                          return tasks.data && !status.supported && candidate.type !== draft.type
                            ? [[candidate.type, 'unavailable']]
                            : []
                        }),
                      )}
                      renderOption={typeOption}
                      onChange={(next) => {
                        if (!next || next === draft.type) return
                        setDraft({ type: next, name: draft.name, fields: {} })
                      }}
                    />
                  </Field>
                </div>
                {typeChanged ? (
                  <Alert tone="warning" title="Changing the type clears its settings">
                    Saving moves this rule to signals.{entry?.collection}. Rules that reference it
                    as {original?.type} stop matching.
                  </Alert>
                ) : renamed && used > 0 ? (
                  <Alert tone="warning" title="References keep the old name">
                    {used} {used === 1 ? 'reference uses' : 'references use'} “{original?.name}”.
                    Update them after saving, or they stop matching.
                  </Alert>
                ) : null}
              </section>
              <section className="flex flex-col gap-3">
                <SectionLabel>Definition</SectionLabel>
                <ConfigPageSchemaFieldsEditor
                  schema={schema}
                  value={draft.fields}
                  readOnly={readOnly}
                  path={['signals', draft.type]}
                  onChange={(fields) => setDraft({ ...draft, fields })}
                />
              </section>
            </TabsContent>

            <TabsContent value="try" className="px-5 py-5">
              {original ? (
                entrypointModel ? (
                  <TryPromptPanel
                    entrypointModel={entrypointModel}
                    dirty={dirty}
                    target={{
                      signals: [{ type: original.type, name: original.name }],
                      rule: { type: original.type, name: original.name, record: original.record },
                    }}
                  />
                ) : (
                  <Alert tone="info" title="No chat model name reaches this recipe">
                    Add an entrypoint that routes a model name to this recipe on the Recipes page,
                    then try prompts here.
                  </Alert>
                )
              ) : (
                <Alert tone="info" title="Save the signal to try it">
                  The Router evaluates saved signals; save this one, then send it a prompt.
                </Alert>
              )}
            </TabsContent>

            <TabsContent value="used" className="flex flex-col gap-5 px-5 py-5">
              {usage && used === 0 ? (
                <div className="rounded-xl border border-dashed border-border-strong">
                  <EmptyState
                    compact
                    icon={Split}
                    tone="violet"
                    title="Nothing reads this signal yet"
                    description="The Router skips signals that no decision, projection or response guard reads. Add it to a decision's rules to start routing on it."
                    actions={
                      readOnly || !original || entry?.decisionReferenceable === false ? null : (
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => onUseInDecision(original)}
                        >
                          <Plus />
                          Use in a decision
                        </Button>
                      )
                    }
                  />
                </div>
              ) : null}
              {usage?.decisions.length ? (
                <section className="flex flex-col gap-2">
                  <SectionLabel>Decisions</SectionLabel>
                  {usage.decisions.map((decision) => (
                    <UsageRow
                      key={decision.name}
                      icon={Split}
                      tone="violet"
                      title={decision.name}
                      detail={ruleExpression(decision.rules)}
                      badge={
                        <span className="shrink-0 rounded-md bg-secondary px-1.5 py-0.5 font-mono text-xs tabular">
                          P{decision.priority ?? 0}
                        </span>
                      }
                      onOpen={() => onOpenDecision(decision.name)}
                    />
                  ))}
                </section>
              ) : null}
              {usage?.guards.length ? (
                <section className="flex flex-col gap-2">
                  <SectionLabel>Response guards</SectionLabel>
                  {usage.guards.map((decision) => (
                    <UsageRow
                      key={decision.name}
                      icon={Split}
                      tone="rose"
                      title={decision.name}
                      detail="response_guard plugin"
                      onOpen={() => onOpenDecision(decision.name)}
                    />
                  ))}
                </section>
              ) : null}
              {usage?.masking.length ? (
                <section className="flex flex-col gap-2">
                  <SectionLabel>Masking</SectionLabel>
                  {usage.masking.map((decision) => (
                    <UsageRow
                      key={decision.name}
                      icon={Split}
                      tone="rose"
                      title={decision.name}
                      detail="masking plugin"
                      onOpen={() => onOpenDecision(decision.name)}
                    />
                  ))}
                </section>
              ) : null}
              {usage?.projections.length ? (
                <section className="flex flex-col gap-2">
                  <SectionLabel>Projections</SectionLabel>
                  {usage.projections.map((projection) => (
                    <UsageRow
                      key={`${projection.kind}:${projection.name}`}
                      icon={Blend}
                      tone="teal"
                      title={projection.name}
                      detail={
                        projection.kind === 'partition'
                          ? 'partition member'
                          : `weighted_sum input · weight ${projection.weight ?? '?'}`
                      }
                      onOpen={() => onOpenProjection(projection.kind, projection.name)}
                    />
                  ))}
                </section>
              ) : null}
            </TabsContent>

            <TabsContent value="yaml" className="px-5 py-5">
              <EntityYamlPanel
                path={`recipes[${scopeLabel}].signals.${entry?.collection ?? draft.type}[${draft.name.trim() || 'new'}]`}
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
                  onClick={() =>
                    dirty ? setDraft(JSON.parse(JSON.stringify(target.draft))) : onOpenChange(false)
                  }
                >
                  {dirty ? 'Discard' : 'Close'}
                </Button>
                <Tooltip
                  content={
                    !availability.supported && tasks.data
                      ? availability.reason
                      : dirty || target.mode === 'create'
                        ? 'Validate and save'
                        : 'No changes'
                  }
                >
                  <span>
                    <Button
                      size="sm"
                      onClick={() => void save()}
                      loading={saving}
                      disabled={!dirty && target.mode !== 'create'}
                    >
                      {saving ? null : <Save />}
                      {target.mode === 'create' ? 'Create signal' : 'Save and apply'}
                    </Button>
                  </span>
                </Tooltip>
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
