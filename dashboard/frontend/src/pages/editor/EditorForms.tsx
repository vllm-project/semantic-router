import * as React from 'react'
import { CircleAlert, RotateCcw, Save } from 'lucide-react'
import {
  Alert,
  Badge,
  Button,
  CopyButton,
  Input,
  Label,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  cn,
} from '../../ui'
import { formatDslName } from '@/lib/dslMutations'
import { PLUGIN_DESCRIPTIONS, PLUGIN_TYPES, SIGNAL_TYPES } from '@/lib/dslSchemas'
import type { FieldSchema } from '@/lib/dslSchemaTypes'
import type {
  ASTModelDecl,
  ASTPluginDecl,
  ASTProjectionMappingDecl,
  ASTProjectionPartitionDecl,
  ASTProjectionScoreDecl,
  ASTSignalDecl,
  DSLFieldObject,
} from '@/types/dsl'
import ConfigPageSchemaFieldsEditor from '../ConfigPageSchemaFieldsEditor'
import { signalTypeIcon, signalTypeLabel } from '../configPageSignalCatalog'
import { pluginLabel } from '../configPageDecisionPresentation'
import { useSignalCapabilities } from '../signalCapabilityState'
import {
  decisionQuestionTypes,
  signalAvailability,
  signalCapabilitySchema,
} from '../signalCapabilities'
import {
  NEW_PROJECTION_FIELDS,
  dslBlockPreview,
  fieldProblems,
  modelFieldSchema,
  normalizeEntityName,
  pluginFieldSchema,
  projectionFieldSchema,
  projectionFields,
  signalFieldSchema,
  type ProjectionDeclKind,
} from './editorSchemas'

type Fields = Record<string, unknown>

const clone = (fields: Fields): Fields => JSON.parse(JSON.stringify(fields ?? {})) as Fields

export function FormSection({
  title,
  description,
  actions,
  children,
  className,
}: {
  title: React.ReactNode
  description?: React.ReactNode
  actions?: React.ReactNode
  children: React.ReactNode
  className?: string
}) {
  return (
    <section className={cn('rounded-xl border border-border bg-card shadow-xs', className)}>
      <header className="flex items-start justify-between gap-3 border-b border-border px-4 py-3">
        <div className="flex min-w-0 flex-col gap-0.5">
          <h3 className="text-[13px] font-semibold">{title}</h3>
          {description ? <p className="text-xs text-muted-foreground">{description}</p> : null}
        </div>
        {actions ? <div className="flex shrink-0 items-center gap-1.5">{actions}</div> : null}
      </header>
      <div className="px-4 py-4">{children}</div>
    </section>
  )
}

const DSL_TOKEN =
  /(#.*$)|("(?:[^"\\]|\\.)*")|\b(SIGNAL|ROUTE|MODEL|PLUGIN|PROJECTION|PRIORITY|WHEN|ALGORITHM|TIER|AND|OR|NOT)\b|(\b-?\d+(?:\.\d+)?\b)|([A-Za-z_][\w.-]*)(?=\s*:)/g

function highlightDsl(line: string): React.ReactNode[] {
  const nodes: React.ReactNode[] = []
  let last = 0
  for (const match of line.matchAll(DSL_TOKEN)) {
    const index = match.index ?? 0
    if (index > last) nodes.push(line.slice(last, index))
    const [text, comment, string, keyword, number] = match
    const className = comment
      ? 'italic text-subtle-foreground'
      : string
        ? 'text-chart-6'
        : keyword
          ? 'font-semibold text-chart-3'
          : number
            ? 'text-chart-7'
            : 'text-chart-1'
    nodes.push(
      <span key={index} className={className}>
        {text}
      </span>,
    )
    last = index + text.length
  }
  if (last < line.length) nodes.push(line.slice(last))
  return nodes
}

/** The DSL block the form writes, so authors learn the source while they click. */
export function DslPreview({ code, title = 'DSL preview' }: { code: string; title?: string }) {
  const lines = code.split('\n')
  return (
    <section className="overflow-hidden rounded-xl border border-border bg-surface-sunken/50 dark:bg-surface-sunken">
      <header className="flex h-9 items-center justify-between border-b border-border pl-4 pr-1.5 text-xs font-medium text-muted-foreground">
        {title}
        <CopyButton value={code} label="Copy DSL" />
      </header>
      <pre className="max-h-80 overflow-auto px-4 py-3 font-mono text-[12.5px] leading-5 text-foreground/90">
        {lines.map((line, index) => (
          <React.Fragment key={index}>
            {highlightDsl(line)}
            {index < lines.length - 1 ? '\n' : null}
          </React.Fragment>
        ))}
      </pre>
    </section>
  )
}

export function ProblemList({ problems }: { problems: string[] }) {
  if (problems.length === 0) return null
  return (
    <ul className="flex flex-col gap-1 rounded-lg border border-destructive/25 bg-destructive-soft px-3 py-2.5 text-xs text-destructive-foreground">
      {problems.map((problem) => (
        <li key={problem} className="flex items-start gap-1.5">
          <CircleAlert className="mt-px size-3.5 shrink-0" />
          {problem}
        </li>
      ))}
    </ul>
  )
}

/** The sticky action bar of an entity form. */
export function FormFooter({
  mode,
  dirty,
  disabled,
  problems,
  onSubmit,
  onReset,
  onCancel,
  submitLabel,
}: {
  mode: 'create' | 'edit'
  dirty: boolean
  disabled?: boolean
  problems: number
  onSubmit: () => void
  onReset?: () => void
  onCancel?: () => void
  submitLabel?: string
}) {
  return (
    <div className="sticky bottom-0 z-10 -mx-5 mt-2 flex items-center justify-between gap-3 border-t border-border bg-card/95 px-5 py-3 backdrop-blur-sm">
      <span className="min-w-0 truncate text-xs text-muted-foreground">
        {problems > 0
          ? `${problems} ${problems === 1 ? 'problem' : 'problems'} to fix before ${mode === 'create' ? 'creating' : 'saving'}`
          : mode === 'create'
            ? 'Adds the block to the working copy; nothing is deployed yet.'
            : dirty
              ? 'Unsaved form changes. Save writes them to the working copy.'
              : 'Matches the working copy.'}
      </span>
      <div className="flex shrink-0 items-center gap-2">
        {mode === 'create' ? (
          <Button variant="ghost" size="sm" onClick={onCancel}>
            Cancel
          </Button>
        ) : (
          <Button variant="ghost" size="sm" onClick={onReset} disabled={!dirty}>
            <RotateCcw />
            Revert
          </Button>
        )}
        <Button size="sm" onClick={onSubmit} disabled={disabled || problems > 0}>
          {mode === 'edit' ? <Save /> : null}
          {submitLabel ?? (mode === 'create' ? 'Create' : 'Save')}
        </Button>
      </div>
    </div>
  )
}

export function NameField({
  id,
  value,
  onChange,
  placeholder,
}: {
  id: string
  value: string
  onChange: (value: string) => void
  placeholder: string
}) {
  return (
    <div className="grid gap-1.5">
      <Label htmlFor={id}>
        Name{' '}
        <span aria-hidden className="text-destructive-foreground">
          *
        </span>
      </Label>
      <Input
        id={id}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder={placeholder}
        autoFocus
        spellCheck={false}
        className="font-mono text-[12.5px]"
      />
      <p className="text-xs text-muted-foreground">
        Letters, digits, <span className="font-mono">_ - . /</span>; other names are quoted in the
        DSL.
      </p>
    </div>
  )
}

function useDraft(initial: Fields) {
  const [draft, setDraft] = React.useState<Fields>(() => clone(initial))
  const baseline = React.useRef(JSON.stringify(initial ?? {}))
  const dirty = JSON.stringify(draft) !== baseline.current
  return {
    draft,
    setDraft,
    dirty,
    reset: () => setDraft(clone(initial)),
  }
}

/* Signals ------------------------------------------------------------------- */

function signalSchema(
  type: string,
  fields: Fields,
  capabilities: ReturnType<typeof useSignalCapabilities>,
): FieldSchema[] {
  return type === 'systemone'
    ? signalCapabilitySchema(
        signalFieldSchema(type),
        decisionQuestionTypes(capabilities.data, capabilities.scope, fields as DSLFieldObject),
      )
    : signalFieldSchema(type)
}

export function SignalForm({
  signal,
  onSave,
  onCreate,
  onCancel,
}: {
  signal?: ASTSignalDecl
  onSave?: (fields: DSLFieldObject) => void
  onCreate?: (signalType: string, name: string, fields: DSLFieldObject) => void
  onCancel?: () => void
}) {
  const capabilities = useSignalCapabilities()
  const [type, setType] = React.useState(signal?.signalType ?? 'domain')
  const [name, setName] = React.useState('')
  const { draft, setDraft, dirty, reset } = useDraft(signal?.fields ?? {})
  const schema = signalSchema(type, draft, capabilities)
  const availability = signalAvailability(
    capabilities.data,
    capabilities.scope,
    type,
    draft as DSLFieldObject,
  )
  const creating = !signal
  const problems = [
    ...(creating && !normalizeEntityName(name) ? ['Name is required.'] : []),
    ...fieldProblems(schema, draft),
  ]
  const disabledTypes = Object.fromEntries(
    SIGNAL_TYPES.flatMap((candidate) => {
      const result = signalAvailability(capabilities.data, capabilities.scope, candidate)
      return result.supported ? [] : [[candidate, result.reason || 'Unavailable']]
    }),
  ) as Record<string, string>
  const previewName = creating ? normalizeEntityName(name) || 'name_here' : signal.name

  return (
    <div className="flex flex-col gap-4">
      {creating ? (
        <FormSection
          title="Signal"
          description="What the signal detects and how decisions name it."
        >
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="grid gap-1.5">
              <Label htmlFor="editor-signal-type">
                Signal type{' '}
                <span aria-hidden className="text-destructive-foreground">
                  *
                </span>
              </Label>
              <Select
                value={type}
                onValueChange={(next) => {
                  setType(next)
                  setDraft({})
                }}
              >
                <SelectTrigger id="editor-signal-type" aria-label="Signal type">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="max-h-80">
                  {SIGNAL_TYPES.map((candidate) => {
                    const Icon = signalTypeIcon(candidate)
                    return (
                      <SelectItem
                        key={candidate}
                        value={candidate}
                        disabled={Boolean(disabledTypes[candidate])}
                      >
                        <span className="flex items-center gap-2">
                          <Icon className="size-3.5 text-chart-2" />
                          {signalTypeLabel(candidate)}
                          <span className="font-mono text-[11px] text-subtle-foreground">
                            {candidate}
                          </span>
                        </span>
                      </SelectItem>
                    )
                  })}
                </SelectContent>
              </Select>
            </div>
            <NameField
              id="editor-signal-name"
              value={name}
              onChange={setName}
              placeholder="my_signal_name"
            />
          </div>
        </FormSection>
      ) : null}
      {!availability.supported || capabilities.error ? (
        <Alert
          tone="warning"
          title={capabilities.error ? 'Signal capabilities unavailable' : 'Signal unavailable'}
          actions={
            capabilities.error ? (
              <Button variant="outline" size="xs" onClick={capabilities.refresh}>
                Retry
              </Button>
            ) : null
          }
        >
          {capabilities.error || availability.reason}
        </Alert>
      ) : null}
      <FormSection
        title="Fields"
        description={`${signalTypeLabel(type)} settings from the Router contract; fields the form does not describe are kept as written.`}
      >
        <ConfigPageSchemaFieldsEditor
          schema={schema}
          value={draft}
          onChange={setDraft}
          path={['signal', type]}
          showAllByDefault={creating}
        />
      </FormSection>
      <ProblemList problems={problems} />
      <DslPreview
        code={dslBlockPreview(
          `SIGNAL ${type} ${formatDslName(previewName)}`,
          draft as DSLFieldObject,
        )}
      />
      <FormFooter
        mode={creating ? 'create' : 'edit'}
        dirty={dirty}
        disabled={!availability.supported}
        problems={problems.length}
        onReset={reset}
        onCancel={onCancel}
        onSubmit={() =>
          creating
            ? onCreate?.(type, normalizeEntityName(name), draft as DSLFieldObject)
            : onSave?.(draft as DSLFieldObject)
        }
      />
    </div>
  )
}

/* Plugins ------------------------------------------------------------------- */

export function PluginForm({
  plugin,
  onSave,
  onCreate,
  onCancel,
}: {
  plugin?: ASTPluginDecl
  onSave?: (fields: DSLFieldObject) => void
  onCreate?: (name: string, pluginType: string, fields: DSLFieldObject) => void
  onCancel?: () => void
}) {
  const creating = !plugin
  const [type, setType] = React.useState<string>(plugin?.pluginType ?? PLUGIN_TYPES[0])
  const [name, setName] = React.useState('')
  const { draft, setDraft, dirty, reset } = useDraft(plugin?.fields ?? { enabled: true })
  const schema = pluginFieldSchema(type)
  const problems = [
    ...(creating && !normalizeEntityName(name) ? ['Name is required.'] : []),
    ...fieldProblems(schema, draft),
  ]
  const previewName = creating ? normalizeEntityName(name) || 'my_plugin' : plugin.name
  return (
    <div className="flex flex-col gap-4">
      {creating ? (
        <FormSection
          title="Plugin"
          description="A reusable plugin template that decisions turn on by name."
        >
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="grid gap-1.5">
              <Label htmlFor="editor-plugin-type">
                Plugin type{' '}
                <span aria-hidden className="text-destructive-foreground">
                  *
                </span>
              </Label>
              <Select
                value={type}
                onValueChange={(next) => {
                  setType(next)
                  setDraft({ enabled: true })
                }}
              >
                <SelectTrigger id="editor-plugin-type" aria-label="Plugin type">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="max-h-80">
                  {PLUGIN_TYPES.map((candidate) => (
                    <SelectItem key={candidate} value={candidate}>
                      <span className="flex items-center gap-2">
                        {pluginLabel(candidate)}
                        <span className="font-mono text-[11px] text-subtle-foreground">
                          {candidate}
                        </span>
                      </span>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {PLUGIN_DESCRIPTIONS[type] ? (
                <p className="text-xs text-muted-foreground">{PLUGIN_DESCRIPTIONS[type]}</p>
              ) : null}
            </div>
            <NameField
              id="editor-plugin-name"
              value={name}
              onChange={setName}
              placeholder="my_plugin"
            />
          </div>
        </FormSection>
      ) : null}
      <FormSection
        title="Configuration"
        description={
          PLUGIN_DESCRIPTIONS[type] ?? `${pluginLabel(type)} settings from the Router contract.`
        }
      >
        <ConfigPageSchemaFieldsEditor
          schema={schema}
          value={draft}
          onChange={setDraft}
          path={['plugin', type]}
          showAllByDefault={creating}
        />
      </FormSection>
      <ProblemList problems={problems} />
      <DslPreview
        code={dslBlockPreview(
          `PLUGIN ${formatDslName(previewName)} ${type}`,
          draft as DSLFieldObject,
        )}
      />
      <FormFooter
        mode={creating ? 'create' : 'edit'}
        dirty={dirty}
        problems={problems.length}
        onReset={reset}
        onCancel={onCancel}
        onSubmit={() =>
          creating
            ? onCreate?.(normalizeEntityName(name), type, draft as DSLFieldObject)
            : onSave?.(draft as DSLFieldObject)
        }
      />
    </div>
  )
}

/* Models -------------------------------------------------------------------- */

export function ModelForm({
  model,
  onSave,
  onCreate,
  onCancel,
}: {
  model?: ASTModelDecl
  onSave?: (fields: DSLFieldObject) => void
  onCreate?: (name: string, fields: DSLFieldObject) => void
  onCancel?: () => void
}) {
  const creating = !model
  const [name, setName] = React.useState('')
  const { draft, setDraft, dirty, reset } = useDraft(model?.fields ?? {})
  const schema = React.useMemo(() => modelFieldSchema(), [])
  const problems = [
    ...(creating && !name.trim() ? ['Name is required.'] : []),
    ...fieldProblems(schema, draft),
  ]
  const previewName = creating ? name.trim() || 'model-name' : model.name
  return (
    <div className="flex flex-col gap-4">
      {creating ? (
        <FormSection
          title="Model"
          description="A routing model card: what decisions may pick. Backends and credentials stay under Backend models."
        >
          <NameField
            id="editor-model-name"
            value={name}
            onChange={setName}
            placeholder="qwen3-32b"
          />
        </FormSection>
      ) : null}
      <FormSection
        title="Model card"
        description="Capabilities, limits and modalities decisions read when they choose this model."
      >
        <ConfigPageSchemaFieldsEditor
          schema={schema}
          value={draft}
          onChange={setDraft}
          path={['model']}
          showAllByDefault={creating}
        />
      </FormSection>
      <ProblemList problems={problems} />
      <DslPreview
        code={dslBlockPreview(`MODEL ${formatDslName(previewName)}`, draft as DSLFieldObject)}
      />
      <FormFooter
        mode={creating ? 'create' : 'edit'}
        dirty={dirty}
        problems={problems.length}
        onReset={reset}
        onCancel={onCancel}
        onSubmit={() =>
          creating
            ? onCreate?.(name.trim(), draft as DSLFieldObject)
            : onSave?.(draft as DSLFieldObject)
        }
      />
    </div>
  )
}

/* Projections --------------------------------------------------------------- */

const projectionCopy: Record<ProjectionDeclKind, { title: string; description: string }> = {
  partition: {
    title: 'Partition',
    description: 'Picks one member among competing domain signals, so overlapping matches resolve.',
  },
  score: {
    title: 'Score',
    description: 'Weighs signal evidence into one number that mappings band into outputs.',
  },
  mapping: {
    title: 'Mapping',
    description: 'Bands a score into named outputs that decisions reference as projection signals.',
  },
}

export function ProjectionForm({
  kind,
  entity,
  onSave,
  onCreate,
  onCancel,
}: {
  kind: ProjectionDeclKind
  entity?: ASTProjectionPartitionDecl | ASTProjectionScoreDecl | ASTProjectionMappingDecl
  onSave?: (fields: DSLFieldObject) => void
  onCreate?: (name: string, fields: DSLFieldObject) => void
  onCancel?: () => void
}) {
  const creating = !entity
  const [name, setName] = React.useState('')
  const initial = React.useMemo(
    () => (entity ? projectionFields(kind, entity) : NEW_PROJECTION_FIELDS[kind]),
    [entity, kind],
  )
  const { draft, setDraft, dirty, reset } = useDraft(initial)
  const schema = React.useMemo(() => projectionFieldSchema(kind), [kind])
  const problems = [
    ...(creating && !normalizeEntityName(name) ? ['Name is required.'] : []),
    ...fieldProblems(schema, draft),
  ]
  const previewName = creating ? normalizeEntityName(name) || 'name_here' : entity.name
  const copy = projectionCopy[kind]
  return (
    <div className="flex flex-col gap-4">
      {creating ? (
        <FormSection
          title={`Projection ${copy.title.toLowerCase()}`}
          description={copy.description}
        >
          <NameField
            id={`editor-projection-${kind}-name`}
            value={name}
            onChange={setName}
            placeholder={
              kind === 'partition'
                ? 'balance_domain_partition'
                : kind === 'score'
                  ? 'request_difficulty'
                  : 'request_band'
            }
          />
        </FormSection>
      ) : null}
      <FormSection
        title="Fields"
        description={creating ? 'The same fields the Projections page edits.' : copy.description}
        actions={
          <Badge tone="teal" size="sm">
            {kind}
          </Badge>
        }
      >
        <ConfigPageSchemaFieldsEditor
          schema={schema}
          value={draft}
          onChange={setDraft}
          path={['projections', `${kind}s`]}
          showAllByDefault
        />
      </FormSection>
      <ProblemList problems={problems} />
      <DslPreview
        code={dslBlockPreview(
          `PROJECTION ${kind} ${formatDslName(previewName)}`,
          draft as DSLFieldObject,
        )}
      />
      <FormFooter
        mode={creating ? 'create' : 'edit'}
        dirty={dirty}
        problems={problems.length}
        onReset={reset}
        onCancel={onCancel}
        onSubmit={() =>
          creating
            ? onCreate?.(normalizeEntityName(name), draft as DSLFieldObject)
            : onSave?.(draft as DSLFieldObject)
        }
      />
    </div>
  )
}
