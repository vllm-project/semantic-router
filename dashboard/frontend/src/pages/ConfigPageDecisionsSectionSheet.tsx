import * as React from 'react'
import {
  ArrowRight,
  CopyPlus,
  FileDown,
  MessagesSquare,
  MoreHorizontal,
  Puzzle,
  Save,
  Split,
  Trash2,
} from 'lucide-react'

import ConfirmDialog from '../components/ConfirmDialog'
import { configCollectionItemYaml } from '../lib/configYaml'
import {
  DECISION_ACTION_SCHEMA,
  DECISION_ADAPTATIONS_SCHEMA,
  DECISION_DECLARATIVE_SCHEMA,
  DECISION_FALLBACK_SCHEMA,
  DECISION_IDENTITY_SCHEMA,
  DECISION_OUTPUT_CONTRACT_SCHEMA,
  DECISION_RELIABILITY_SCHEMA,
} from '../lib/decisionSchemas'
import type { FieldSchema } from '../lib/dslSchemas'
import { getAlgorithmFieldSchema } from '../lib/dslAlgorithmSchemas'
import { requiredFieldErrors } from '../lib/schemaFieldValidation'
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
  IconTile,
  MiniStat,
  ProviderLogo,
  SectionLabel,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
  toast,
  useCopy,
} from '../ui'
import ConfigPageDecisionAlgorithmEditor from './ConfigPageDecisionAlgorithmEditor'
import ConfigPageDecisionModelRefsEditor from './ConfigPageDecisionModelRefsEditor'
import ConfigPageDecisionPluginsEditor from './ConfigPageDecisionPluginsEditor'
import ConfigPageDecisionRulesEditor from './ConfigPageDecisionRulesEditor'
import ConfigPageSchemaFieldsEditor from './ConfigPageSchemaFieldsEditor'
import { EntityYamlPanel, SignalChip, TryPromptPanel } from './configPageRoutingScopeViews'
import { algorithmFields, algorithmType } from './configPageDecisionAlgorithmSupport'
import {
  DecisionDraftError,
  cloneDecision,
  decisionFromDraft,
  type DecisionTab,
} from './configPageDecisionFormSupport'
import {
  DECISION_KIND_HINT,
  DECISION_KIND_LABEL,
  algorithmMeta,
  decisionKind,
  modelDisplayName,
  modelLogo,
  pluginLabel,
} from './configPageDecisionPresentation'
import {
  countRuleConditions,
  ruleConnective,
  ruleExpression,
  ruleLeaves,
} from './configPageDecisionRuleSupport'
import {
  decisionOutputForEditor,
  decisionOutputFromEditor,
  type DecisionConfig,
  type NormalizedModel,
  type ReasoningFamily,
} from './configPageSupport'

export type DecisionSheetView = DecisionTab | 'yaml' | 'try'

export interface DecisionSheetTarget {
  mode: 'create' | 'edit'
  original?: DecisionConfig
  draft: DecisionConfig
}

const kindTone = { routing: 'violet', guardrail: 'rose', fallback: 'neutral' } as const

const ADVANCED_SCHEMA: FieldSchema[] = [
  {
    key: 'reliability',
    label: 'Timeouts & Retries',
    type: 'object',
    description:
      "Overrides the provider model's timeouts and retries for this decision. Retry conditions and status codes add to the provider model's.",
    fields: DECISION_RELIABILITY_SCHEMA,
  },
  {
    key: 'fallback',
    label: 'Fallback',
    type: 'object',
    description:
      "Overrides the recipe's cross-model fallback for this decision, field by field. The candidates stay this decision's ranked models.",
    fields: DECISION_FALLBACK_SCHEMA,
  },
  {
    key: 'output_contract',
    label: 'Output Contract',
    type: 'string',
    description: 'Optional model-facing output instructions.',
  },
  {
    key: 'output_contract_spec',
    label: 'Output Contract Specification',
    type: 'object',
    description: 'Typed extraction, normalization, rendering, and post-processing behavior.',
    fields: DECISION_OUTPUT_CONTRACT_SCHEMA,
  },
  ...DECISION_DECLARATIVE_SCHEMA,
]

const LEARNING_SCHEMA: FieldSchema[] = [
  {
    key: 'adaptations',
    label: 'Learning & Protection',
    type: 'object',
    description: 'Decision-level overrides for online adaptation and model-switch protection.',
    fields: DECISION_ADAPTATIONS_SCHEMA,
  },
]

const IDENTITY_SCHEMA = DECISION_IDENTITY_SCHEMA.filter((field) => field.key !== 'output_contract')
const ACTION_FIELD: FieldSchema = {
  key: 'action',
  label: 'Direct Action',
  type: 'object',
  description: 'Route straight to one destination model instead of ranking the candidates.',
  fields: DECISION_ACTION_SCHEMA,
}

function validators() {
  return {
    actionErrors: (action: Record<string, unknown>) =>
      requiredFieldErrors(DECISION_ACTION_SCHEMA, action),
    algorithmErrors: (algorithm: Record<string, unknown>) =>
      requiredFieldErrors(
        getAlgorithmFieldSchema(algorithmType(algorithm)),
        algorithmFields(algorithm),
      ),
    declarativeErrors: (decision: DecisionConfig) =>
      requiredFieldErrors(DECISION_DECLARATIVE_SCHEMA, {
        candidateIterations: decision.candidateIterations,
        retention: decision.retention,
        output: decision.output ? decisionOutputForEditor(decision.output) : undefined,
        annotations: decision.annotations,
      }),
  }
}

function Summary({
  draft,
  models,
  onOpenTab,
}: {
  draft: DecisionConfig
  models: readonly NormalizedModel[]
  onOpenTab: (tab: DecisionSheetView) => void
}) {
  const leaves = ruleLeaves(draft.rules)
  const algorithm = algorithmMeta(draft.algorithm?.type)
  const refs = draft.modelRefs ?? []
  const plugins = draft.plugins ?? []
  return (
    <div className="flex flex-col gap-5">
      <section className="flex flex-col gap-2">
        <div className="flex items-center justify-between">
          <SectionLabel>When</SectionLabel>
          <Button variant="link" size="xs" onClick={() => onOpenTab('rules')}>
            Edit rules
            <ArrowRight />
          </Button>
        </div>
        <div className="flex flex-col gap-2 rounded-lg border border-border bg-surface-subtle/60 p-3">
          {leaves.length ? (
            <div className="flex flex-wrap items-center gap-1">
              {leaves.slice(0, 8).map((leaf, index) => (
                <React.Fragment key={index}>
                  {index > 0 ? (
                    <span className="px-0.5 font-mono text-[10px] font-semibold text-chart-3">
                      {ruleConnective(draft.rules)}
                    </span>
                  ) : null}
                  <SignalChip
                    type={String(leaf.node.type ?? '')}
                    name={String(leaf.node.name ?? '')}
                    label={typeof leaf.node.label === 'string' ? leaf.node.label : undefined}
                    negated={leaf.negated}
                  />
                </React.Fragment>
              ))}
              {leaves.length > 8 ? (
                <span className="text-xs text-subtle-foreground">+{leaves.length - 8}</span>
              ) : null}
            </div>
          ) : null}
          <p className="font-mono text-[12px] leading-relaxed text-muted-foreground">
            {ruleExpression(draft.rules)}
          </p>
        </div>
      </section>
      <section className="flex flex-col gap-3">
        <SectionLabel>Then</SectionLabel>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <button
            type="button"
            onClick={() => onOpenTab('algorithm')}
            className="flex flex-col gap-1 rounded-lg text-left focus-visible:ring-[3px] focus-visible:ring-ring/40 focus-visible:outline-none"
          >
            <span className="text-xs text-subtle-foreground">Algorithm</span>
            {algorithm ? (
              <Badge tone="violet">{algorithm.label}</Badge>
            ) : (
              <span className="text-[13px] text-muted-foreground">Not set</span>
            )}
          </button>
          <div className="flex flex-col gap-1">
            <span className="text-xs text-subtle-foreground">Direct action</span>
            {draft.action?.destination ? (
              <span className="font-mono text-[12.5px]">
                {draft.action.type} → {draft.action.destination}
              </span>
            ) : (
              <span className="text-[13px] text-muted-foreground">None</span>
            )}
          </div>
        </div>
        <div className="flex flex-col gap-1.5">
          <span className="text-xs text-subtle-foreground">Candidate models</span>
          {refs.length ? (
            <div className="flex flex-wrap gap-1.5">
              {refs.map((ref, index) => {
                const model = models.find((candidate) => candidate.name === ref.model)
                return (
                  <button
                    key={`${ref.model}-${index}`}
                    type="button"
                    onClick={() => onOpenTab('models')}
                    className="inline-flex h-7 items-center gap-1.5 rounded-md border border-border bg-surface px-2 text-xs transition-colors hover:border-border-strong"
                  >
                    <ProviderLogo provider={modelLogo(model)} size={14} />
                    {modelDisplayName(model, ref.model || 'unnamed')}
                    {ref.use_reasoning ? (
                      <span className="text-[10.5px] text-subtle-foreground">reasoning</span>
                    ) : null}
                  </button>
                )
              })}
            </div>
          ) : (
            <span className="text-[13px] text-muted-foreground">
              {plugins.some((plugin) => plugin.type === 'fast_response')
                ? 'Answers with a fixed response'
                : 'None'}
            </span>
          )}
        </div>
        <div className="flex flex-col gap-1.5">
          <span className="text-xs text-subtle-foreground">Plugins</span>
          {plugins.length ? (
            <div className="flex flex-wrap gap-1.5">
              {plugins.map((plugin, index) => (
                <button
                  key={`${plugin.type}-${index}`}
                  type="button"
                  onClick={() => onOpenTab('plugins')}
                >
                  <Badge tone="rose">
                    <Puzzle />
                    {pluginLabel(plugin.type)}
                  </Badge>
                </button>
              ))}
            </div>
          ) : (
            <span className="text-[13px] text-muted-foreground">None</span>
          )}
        </div>
      </section>
    </div>
  )
}

interface ConfigPageDecisionSheetProps {
  target: DecisionSheetTarget | null
  open: boolean
  view: DecisionSheetView
  onViewChange: (view: DecisionSheetView) => void
  onOpenChange: (open: boolean) => void
  scopeLabel: string
  readOnly: boolean
  models: NormalizedModel[]
  reasoningFamilies: Record<string, ReasoningFamily>
  entrypointModel?: string
  recordedRequests?: number
  onSave: (decision: DecisionConfig, originalName?: string) => Promise<void>
  onDelete: (decision: DecisionConfig) => void
  onDuplicate: (decision: DecisionConfig) => void
  onTestInPlayground: () => void
}

export default function ConfigPageDecisionSheet({
  target,
  open,
  view,
  onViewChange,
  onOpenChange,
  scopeLabel,
  readOnly,
  models,
  reasoningFamilies,
  entrypointModel,
  recordedRequests,
  onSave,
  onDelete,
  onDuplicate,
  onTestInPlayground,
}: ConfigPageDecisionSheetProps) {
  const { copy } = useCopy()
  const [draft, setDraft] = React.useState<DecisionConfig | null>(
    target ? cloneDecision(target.draft) : null,
  )
  const [saving, setSaving] = React.useState(false)
  const [error, setError] = React.useState<{ message: string; tab: DecisionTab } | null>(null)
  const [confirmDiscard, setConfirmDiscard] = React.useState(false)
  const targetKey = target
    ? `${target.mode}:${target.original?.name ?? ''}:${target.mode === 'create' ? JSON.stringify(target.draft) : ''}`
    : ''
  React.useEffect(() => {
    setDraft(target ? cloneDecision(target.draft) : null)
    setError(null)
    setSaving(false)
    // Reset only when a different decision opens, not when the saved config refreshes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [targetKey])

  if (!target || !draft) return null
  const original = target.original
  const dirty = JSON.stringify(draft) !== JSON.stringify(target.draft)
  const kind = decisionKind(draft)
  const set = (key: keyof DecisionConfig, value: unknown) => {
    const next = { ...draft } as unknown as Record<string, unknown>
    if (value === undefined) delete next[key]
    else next[key] = value
    setDraft(next as unknown as DecisionConfig)
  }
  const preview = (() => {
    try {
      return decisionFromDraft(draft)
    } catch {
      return draft
    }
  })()
  const yaml = configCollectionItemYaml(['decisions'], preview, '#/$defs/Decision')
  const savedYaml = original
    ? configCollectionItemYaml(['decisions'], original, '#/$defs/Decision')
    : undefined
  const conditions = countRuleConditions(draft.rules)
  const title = draft.name.trim() || 'New decision'

  const requestClose = () => {
    if (dirty && !readOnly) setConfirmDiscard(true)
    else onOpenChange(false)
  }

  const save = async () => {
    let decision: DecisionConfig
    try {
      decision = decisionFromDraft(draft, validators())
    } catch (failure) {
      const tab = failure instanceof DecisionDraftError ? failure.tab : 'overview'
      setError({ message: failure instanceof Error ? failure.message : String(failure), tab })
      onViewChange(tab)
      return
    }
    setSaving(true)
    setError(null)
    try {
      await onSave(decision, original?.name)
      toast.success(`${decision.name} saved`, { description: `Saved in recipe ${scopeLabel}.` })
    } catch (failure) {
      const tab =
        failure instanceof DecisionDraftError
          ? failure.tab
          : view === 'yaml' || view === 'try'
            ? 'overview'
            : view
      setError({
        message: failure instanceof Error ? failure.message : 'The decision could not be saved.',
        tab,
      })
      if (view === 'yaml' || view === 'try') onViewChange('overview')
    } finally {
      setSaving(false)
    }
  }

  const errorAlert = (tab: DecisionTab) =>
    error && error.tab === tab ? (
      <Alert tone="danger" title="The decision was not saved">
        {error.message}
      </Alert>
    ) : null

  const advancedValue = {
    ...draft,
    output: draft.output ? decisionOutputForEditor(draft.output) : undefined,
  } as Record<string, unknown>
  const advancedCount = ADVANCED_SCHEMA.filter((field) => {
    const value = (draft as unknown as Record<string, unknown>)[field.key]
    return value !== undefined && !(Array.isArray(value) && value.length === 0)
  }).length

  return (
    <>
      <DetailSheet
        open={open}
        size="xl"
        onOpenChange={(next) => (next ? onOpenChange(true) : requestClose())}
      >
        <DetailSheetHeader
          icon={
            <IconTile
              icon={Split}
              tone={kind === 'guardrail' ? 'rose' : kind === 'fallback' ? 'slate' : 'violet'}
              size="lg"
            />
          }
          title={<span className="font-mono">{title}</span>}
          badges={
            <>
              <Badge tone={kindTone[kind]}>{DECISION_KIND_LABEL[kind]}</Badge>
              {target.mode === 'create' ? <Badge tone="info">New</Badge> : null}
            </>
          }
          description={
            <>
              {draft.description ? (
                <span className="max-w-[420px] truncate">{draft.description}</span>
              ) : null}
              {draft.description ? <span className="text-subtle-foreground">·</span> : null}
              <span className="font-mono">P{draft.priority ?? 0}</span>
              {draft.tier !== undefined ? (
                <>
                  <span className="text-subtle-foreground">·</span>
                  <span className="font-mono">tier {draft.tier}</span>
                </>
              ) : null}
              <span className="text-subtle-foreground">·</span>
              <span>
                recipe <span className="font-mono">{scopeLabel}</span>
              </span>
            </>
          }
          actions={
            <>
              <Button variant="outline" size="sm" onClick={() => onViewChange('try')}>
                <MessagesSquare />
                Test
              </Button>
              {original ? (
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="ghost" size="icon-sm" aria-label="Decision actions">
                      <MoreHorizontal />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end" className="w-44">
                    {readOnly ? null : (
                      <DropdownMenuItem onSelect={() => onDuplicate(original)}>
                        <CopyPlus />
                        Duplicate
                      </DropdownMenuItem>
                    )}
                    <DropdownMenuItem
                      onSelect={() => {
                        copy(savedYaml ?? yaml)
                        toast.success('Decision YAML copied')
                      }}
                    >
                      <FileDown />
                      Copy YAML
                    </DropdownMenuItem>
                    {readOnly ? null : (
                      <>
                        <DropdownMenuSeparator />
                        <DropdownMenuItem variant="destructive" onSelect={() => onDelete(original)}>
                          <Trash2 />
                          Delete
                        </DropdownMenuItem>
                      </>
                    )}
                  </DropdownMenuContent>
                </DropdownMenu>
              ) : null}
            </>
          }
        />
        <Tabs
          value={view}
          onValueChange={(next) => onViewChange(next as DecisionSheetView)}
          className="min-h-0 flex-1 gap-0"
        >
          <TabsList className="overflow-x-auto px-5 [scrollbar-width:none]">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="rules">
              Rules
              <Badge size="sm" tone="neutral" className="tabular">
                {conditions}
              </Badge>
            </TabsTrigger>
            <TabsTrigger value="models">
              Models
              <Badge size="sm" tone="neutral" className="tabular">
                {draft.modelRefs?.length ?? 0}
              </Badge>
            </TabsTrigger>
            <TabsTrigger value="algorithm">Algorithm</TabsTrigger>
            <TabsTrigger value="plugins">
              Plugins
              <Badge size="sm" tone="neutral" className="tabular">
                {draft.plugins?.length ?? 0}
              </Badge>
            </TabsTrigger>
            <TabsTrigger value="advanced">
              Advanced
              {advancedCount ? (
                <Badge size="sm" tone="neutral" className="tabular">
                  {advancedCount}
                </Badge>
              ) : null}
            </TabsTrigger>
            <TabsTrigger value="try">Try it</TabsTrigger>
            <TabsTrigger value="yaml">YAML</TabsTrigger>
          </TabsList>
          <DetailSheetBody>
            <TabsContent value="overview" className="flex flex-col gap-5 px-5 py-5">
              {errorAlert('overview')}
              <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                <MiniStat
                  label="Conditions"
                  value={conditions}
                  hint={
                    conditions
                      ? ruleConnective(draft.rules) === 'OR'
                        ? 'any must match'
                        : 'all must match'
                      : 'matches every request'
                  }
                />
                <MiniStat
                  label="Models"
                  value={draft.modelRefs?.length ?? 0}
                  hint={draft.action?.destination ? 'plus a direct action' : undefined}
                />
                <MiniStat label="Plugins" value={draft.plugins?.length ?? 0} />
                <MiniStat
                  label="Recorded requests"
                  value={recordedRequests === undefined ? '—' : recordedRequests.toLocaleString()}
                  hint={
                    recordedRequests === undefined
                      ? 'Router Replay unavailable'
                      : 'from Router Replay'
                  }
                />
              </div>
              {kind !== 'routing' ? (
                <Alert tone="neutral" title={DECISION_KIND_LABEL[kind]}>
                  {DECISION_KIND_HINT[kind]}
                </Alert>
              ) : null}
              <section className="flex flex-col gap-3">
                <SectionLabel>Identity</SectionLabel>
                <ConfigPageSchemaFieldsEditor
                  schema={IDENTITY_SCHEMA}
                  value={draft as unknown as Record<string, unknown>}
                  readOnly={readOnly}
                  showAllByDefault
                  hideAdditional
                  path={['decision']}
                  onChange={(next) => setDraft(next as unknown as DecisionConfig)}
                />
              </section>
              <Summary draft={draft} models={models} onOpenTab={onViewChange} />
            </TabsContent>

            <TabsContent value="rules" className="flex flex-col gap-4 px-5 py-5">
              {errorAlert('rules')}
              <p className="text-[13px] text-muted-foreground">
                The decision matches when this tree is true. Signals are evaluated once per request;
                among matching decisions the recipe&apos;s strategy picks the winner.
              </p>
              <ConfigPageDecisionRulesEditor
                value={draft.rules ?? {}}
                readOnly={readOnly}
                onChange={(rules) => set('rules', rules)}
              />
            </TabsContent>

            <TabsContent value="models" className="flex flex-col gap-5 px-5 py-5">
              {errorAlert('models')}
              <section className="flex flex-col gap-3">
                <div className="flex items-baseline justify-between gap-3">
                  <SectionLabel>Candidate models</SectionLabel>
                  <span className="font-mono text-[11px] text-subtle-foreground">modelRefs</span>
                </div>
                <ConfigPageDecisionModelRefsEditor
                  value={draft.modelRefs ?? []}
                  models={models}
                  reasoningFamilies={reasoningFamilies}
                  readOnly={readOnly}
                  onChange={(modelRefs) => set('modelRefs', modelRefs)}
                />
              </section>
              <ConfigPageSchemaFieldsEditor
                schema={[ACTION_FIELD]}
                value={draft as unknown as Record<string, unknown>}
                readOnly={readOnly}
                hideAdditional
                showAllByDefault
                path={['decision']}
                onChange={(next) => setDraft(next as unknown as DecisionConfig)}
              />
            </TabsContent>

            <TabsContent value="algorithm" className="flex flex-col gap-5 px-5 py-5">
              {errorAlert('algorithm')}
              <ConfigPageDecisionAlgorithmEditor
                value={draft.algorithm}
                readOnly={readOnly}
                onChange={(algorithm) => set('algorithm', algorithm)}
              />
              <ConfigPageSchemaFieldsEditor
                schema={LEARNING_SCHEMA}
                value={draft as unknown as Record<string, unknown>}
                readOnly={readOnly}
                hideAdditional
                showAllByDefault
                path={['decision']}
                onChange={(next) => setDraft(next as unknown as DecisionConfig)}
              />
            </TabsContent>

            <TabsContent value="plugins" className="flex flex-col gap-4 px-5 py-5">
              {errorAlert('plugins')}
              <ConfigPageDecisionPluginsEditor
                value={draft.plugins ?? []}
                readOnly={readOnly}
                onChange={(plugins) => set('plugins', plugins)}
              />
            </TabsContent>

            <TabsContent value="advanced" className="flex flex-col gap-4 px-5 py-5">
              {errorAlert('advanced')}
              <ConfigPageSchemaFieldsEditor
                schema={ADVANCED_SCHEMA}
                value={advancedValue}
                readOnly={readOnly}
                hideAdditional
                showAllByDefault
                path={['decision']}
                onChange={(next) => {
                  const updated = { ...next } as Record<string, unknown>
                  const output = decisionOutputFromEditor(updated.output)
                  if (output) updated.output = output
                  else delete updated.output
                  setDraft(updated as unknown as DecisionConfig)
                }}
              />
            </TabsContent>

            <TabsContent value="try" className="flex flex-col gap-4 px-5 py-5">
              {original && entrypointModel ? (
                <TryPromptPanel
                  entrypointModel={entrypointModel}
                  dirty={dirty}
                  target={{ decision: original.name }}
                />
              ) : (
                <Alert
                  tone="info"
                  title={
                    original
                      ? 'No chat model name reaches this recipe'
                      : 'Save the decision to try it'
                  }
                >
                  {original
                    ? 'Add an entrypoint that routes a model name to this recipe on the Recipes page, then try prompts here.'
                    : 'The Router evaluates saved decisions; save this one, then send it a prompt.'}
                </Alert>
              )}
              <Button variant="link" size="xs" className="w-fit" onClick={onTestInPlayground}>
                Chat with the Router in the Playground
                <ArrowRight />
              </Button>
            </TabsContent>

            <TabsContent value="yaml" className="px-5 py-5">
              <EntityYamlPanel
                path={`recipes[${scopeLabel}].decisions[${draft.name.trim() || 'new'}]`}
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
                    dirty ? setDraft(cloneDecision(target.draft)) : onOpenChange(false)
                  }
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
                  {target.mode === 'create' ? 'Create decision' : 'Save and apply'}
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
