import * as React from 'react'
import { ChevronRight, Plus, RotateCcw, Trash2, Undo2 } from 'lucide-react'
import {
  Badge,
  Button,
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
  Input,
  Tooltip,
  cn,
} from '../ui'
import { pluginFieldsFromRouterSchema } from '../lib/routerConfigSchema'
import {
  BooleanControl,
  ChipsControl,
  ChoiceControl,
  CodeControl,
  FieldActions,
  FieldShell,
  KeyValueControl,
  NumberControl,
  OptionSetControl,
  SelectControl,
  TextControl,
} from './ConfigPageRouterConfigSectionControls'
import {
  deepEqual,
  getAt,
  isRecord,
  leafPaths,
  pathKey,
  type SettingPath,
} from './configPageRouterSettingsDraft'
import {
  changedCount,
  displayValue,
  effectiveKind,
  fieldVisible,
  isScalarSchema,
  selectOptions,
  useRouterSettingsEditor,
  type RouterSettingsEditor,
} from './configPageRouterSettingsEditor'
import {
  isSecretSetting,
  restartReason,
  settingLabel,
  settingPresentation,
} from './configPageRouterSettingsModel'
import { generatedStructuredSchema } from './configPageRouterStructuredFields'
import {
  createRouterStructuredValue,
  type RouterStructuredSchema,
} from './configPageRouterStructuredSchema'

function domId(path: SettingPath): string {
  return `setting-${path.join('-').replace(/[^a-zA-Z0-9_-]/g, '_')}`
}

function explicitCount(editor: RouterSettingsEditor, path: SettingPath): number {
  return leafPaths(getAt(editor.draft, path)).length
}

function inheritedPlaceholder(
  editor: RouterSettingsEditor,
  path: SettingPath,
  schema: RouterStructuredSchema,
): { value: unknown; text: string } {
  // An effective value is the default only when config.yaml did not set the field.
  const value = getAt(editor.saved, path) === undefined ? getAt(editor.effective, path) : undefined
  if (value !== undefined && !isRecord(value)) return { value, text: displayValue(value) }
  return {
    value: undefined,
    text: schema.placeholder ?? (editor.effective ? 'Not set' : 'Router default'),
  }
}

function errorFor(editor: RouterSettingsEditor, path: SettingPath): string | undefined {
  return editor.errors.get(pathKey(path))
}

interface PathFieldProps {
  schema: RouterStructuredSchema
  path: string[]
  keyName: string
  percent?: boolean
  disabled?: boolean
  /** The only field of its card: the card header already names it. */
  bare?: boolean
}

/** One setting bound to a path of the global override, with inherited and edited states. */
export function PathField(props: PathFieldProps) {
  const editor = useRouterSettingsEditor()
  const { schema, path, keyName } = props
  if (!fieldVisible(editor, schema, keyName, path)) return null
  const kind = effectiveKind(schema)
  if (kind === 'object') return <GroupField {...props} />
  if (kind === 'object-list') return <ListField {...props} />
  if (kind === 'object-map') return <MapField {...props} />
  if (kind === 'string-map') return <KeyValueField {...props} />
  if (kind === 'json') return <CodeField {...props} />
  return <ScalarField {...props} />
}

function ScalarField({ schema, path, keyName, percent, disabled: blocked }: PathFieldProps) {
  const editor = useRouterSettingsEditor()
  const id = domId(path)
  const kind = effectiveKind(schema)
  const explicitValue = getAt(editor.draft, path)
  const savedValue = getAt(editor.saved, path)
  const explicit = explicitValue !== undefined
  const edited = !deepEqual(explicitValue, savedValue)
  const { label, unit } = settingPresentation(keyName, schema.label)
  const inherited = inheritedPlaceholder(editor, path, schema)
  const disabled = editor.readOnly || blocked
  const error = errorFor(editor, path)
  const set = (value: unknown) =>
    value === undefined ? editor.unset(path) : editor.set(path, value)

  let control: React.ReactNode
  if (kind === 'boolean') {
    const checked = explicit ? explicitValue === true : inherited.value === true
    control = (
      <BooleanControl
        id={id}
        label={label}
        checked={checked}
        inherited={!explicit}
        disabled={disabled}
        onChange={set}
      />
    )
  } else if (kind === 'select') {
    const options = selectOptions(schema)
    const current =
      typeof explicitValue === 'string' && explicitValue !== '' ? explicitValue : undefined
    const short =
      options.length >= 2 && options.length <= 3 && options.every((option) => option.length <= 14)
    control = short ? (
      <ChoiceControl
        id={id}
        label={label}
        value={current}
        inheritedValue={typeof inherited.value === 'string' ? inherited.value : undefined}
        options={options}
        disabled={disabled}
        onChange={set}
      />
    ) : (
      <SelectControl
        id={id}
        label={label}
        value={current}
        placeholder={inherited.text}
        options={options}
        disabled={disabled}
        invalid={Boolean(error)}
        onChange={set}
      />
    )
  } else if (kind === 'number') {
    const placeholder =
      typeof inherited.value === 'number' && percent
        ? String(Math.round(inherited.value * 10000) / 100)
        : inherited.text
    control = (
      <NumberControl
        id={id}
        value={explicitValue}
        placeholder={placeholder}
        unit={unit}
        percent={percent}
        disabled={disabled}
        invalid={Boolean(error)}
        onChange={set}
      />
    )
  } else if (kind === 'string-list' && schema.options?.length) {
    const inheritedList =
      getAt(editor.saved, path) === undefined ? getAt(editor.effective, path) : undefined
    control = (
      <OptionSetControl
        id={id}
        label={label}
        options={schema.options}
        values={Array.isArray(explicitValue) ? explicitValue : undefined}
        inheritedValues={Array.isArray(inheritedList) ? inheritedList : undefined}
        disabled={disabled}
        onChange={(next) => (next.length === 0 && !explicit ? editor.unset(path) : set(next))}
      />
    )
  } else if (kind === 'string-list' || kind === 'number-list') {
    const inheritedList =
      getAt(editor.saved, path) === undefined ? getAt(editor.effective, path) : undefined
    control = (
      <ChipsControl
        id={id}
        label={label}
        numeric={kind === 'number-list'}
        values={
          Array.isArray(explicitValue) ? (explicitValue as Array<string | number>) : undefined
        }
        inheritedValues={Array.isArray(inheritedList) ? inheritedList : undefined}
        placeholder={schema.placeholder}
        disabled={disabled}
        onChange={set}
      />
    )
  } else {
    control = (
      <TextControl
        id={id}
        label={label}
        value={
          typeof explicitValue === 'string'
            ? explicitValue
            : explicitValue === undefined
              ? undefined
              : String(explicitValue)
        }
        placeholder={inherited.text}
        secret={isSecretSetting(schema, keyName)}
        unit={unit}
        disabled={disabled}
        invalid={Boolean(error)}
        onChange={set}
      />
    )
  }

  return (
    <FieldShell
      id={id}
      label={label}
      inherited={!explicit}
      edited={edited}
      restart={restartReason(path)}
      hint={schema.description}
      error={error}
      actions={
        <FieldActions
          label={label}
          edited={edited}
          explicit={explicit}
          disabled={disabled}
          onUndo={() => editor.restore(path)}
          onReset={() => editor.unset(path)}
        />
      }
    >
      {control}
    </FieldShell>
  )
}

/** Splits a schema's children into the two-column grid and full-width blocks. */
function childEntries(schema: RouterStructuredSchema): Array<[string, RouterStructuredSchema]> {
  return Object.entries(schema.fields ?? {})
}

export function FieldGrid({
  children,
  className,
}: {
  children: React.ReactNode
  className?: string
}) {
  return (
    <div className={cn('grid grid-cols-1 gap-x-8 gap-y-5 @[34rem]:grid-cols-2', className)}>
      {children}
    </div>
  )
}

export function PathFields({
  entries,
  disabled,
  bare,
}: {
  entries: Array<{ key: string; path: string[]; schema: RouterStructuredSchema; percent?: boolean }>
  disabled?: boolean
  bare?: boolean
}) {
  const scalars = entries.filter(
    (entry) => isScalarSchema(entry.schema) || effectiveKind(entry.schema) === 'select',
  )
  const blocks = entries.filter((entry) => !scalars.includes(entry))
  return (
    <>
      {scalars.length ? (
        <FieldGrid>
          {scalars.map((entry) => (
            <PathField
              key={entry.path.join('.')}
              schema={entry.schema}
              path={entry.path}
              keyName={entry.key}
              percent={entry.percent}
              disabled={disabled}
            />
          ))}
        </FieldGrid>
      ) : null}
      {blocks.map((entry) => (
        <PathField
          key={entry.path.join('.')}
          schema={entry.schema}
          path={entry.path}
          keyName={entry.key}
          percent={entry.percent}
          disabled={disabled}
          bare={bare && entries.length === 1}
        />
      ))}
    </>
  )
}

function GroupSummary({ explicit, edited }: { explicit: number; edited: number }) {
  return (
    <span className="flex shrink-0 items-center gap-1.5">
      {edited ? (
        <Badge size="sm" tone="brand">
          {edited} edited
        </Badge>
      ) : null}
      {explicit ? (
        <span className="text-xs tabular text-muted-foreground">{explicit} set</span>
      ) : (
        <span className="text-xs text-subtle-foreground">Defaults</span>
      )}
    </span>
  )
}

function GroupField({ schema, path, keyName, disabled }: PathFieldProps) {
  const editor = useRouterSettingsEditor()
  const forced = editor.query !== '' || editor.show !== 'all'
  const [open, setOpen] = React.useState(false)
  const isOpen = forced || open
  const label = settingLabel(schema.label || keyName)
  const edited = changedCount(editor, path)
  const explicit = explicitCount(editor, path)
  const entries = childEntries(schema).map(([key, child]) => ({
    key,
    path: [...path, key],
    schema: child,
  }))
  const error = errorFor(editor, path)

  return (
    <Collapsible
      open={isOpen}
      onOpenChange={setOpen}
      className="group/group rounded-lg border border-border bg-surface-subtle/40 dark:bg-surface-subtle/30"
    >
      <div className="flex min-h-11 items-center gap-2 pr-2">
        <CollapsibleTrigger asChild>
          <button
            type="button"
            className="flex min-w-0 flex-1 items-center gap-2 rounded-lg py-2.5 pl-3 text-left outline-none focus-visible:ring-[3px] focus-visible:ring-ring/40"
          >
            <ChevronRight
              className={cn(
                'size-4 shrink-0 text-subtle-foreground transition-transform duration-200',
                isOpen && 'rotate-90',
              )}
            />
            <span className="flex min-w-0 flex-col">
              <span className="flex items-center gap-1.5 text-[13px] font-medium">
                {edited ? (
                  <span
                    className="size-1.5 shrink-0 rounded-full bg-primary"
                    role="img"
                    aria-label="Edited"
                  />
                ) : null}
                <span className="truncate">{label}</span>
                {restartReason(path) ? (
                  <Badge size="sm" tone="outline" className="font-normal">
                    restart
                  </Badge>
                ) : null}
              </span>
              {schema.description ? (
                <span className="truncate text-xs text-muted-foreground">{schema.description}</span>
              ) : null}
            </span>
          </button>
        </CollapsibleTrigger>
        <GroupSummary explicit={explicit} edited={edited} />
        {!editor.readOnly && !disabled && (edited || explicit) ? (
          <Tooltip
            content={edited ? 'Undo changes in this group' : 'Use router defaults for this group'}
          >
            <Button
              variant="ghost"
              size="icon-xs"
              aria-label={edited ? `Undo changes in ${label}` : `Reset ${label} to defaults`}
              onClick={() => (edited ? editor.restore(path) : editor.unset(path))}
              className={cn(
                !edited &&
                  'opacity-0 transition-opacity focus-visible:opacity-100 group-hover/group:opacity-100',
              )}
            >
              {edited ? <Undo2 /> : <RotateCcw />}
            </Button>
          </Tooltip>
        ) : null}
      </div>
      <CollapsibleContent>
        <div className="flex flex-col gap-4 border-t border-border px-3 pb-4 pt-4 @container">
          {error ? <p className="text-xs text-destructive-foreground">{error}</p> : null}
          <PathFields entries={entries} disabled={disabled} />
        </div>
      </CollapsibleContent>
    </Collapsible>
  )
}

function BlockHeader({
  label,
  hint,
  inherited,
  edited,
  count,
  actions,
  error,
}: {
  label: string
  hint?: string
  inherited: boolean
  edited: boolean
  count?: string
  actions?: React.ReactNode
  error?: string
}) {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex min-h-6 items-center gap-1.5">
        {edited ? (
          <span
            className="size-1.5 shrink-0 rounded-full bg-primary"
            role="img"
            aria-label="Edited"
          />
        ) : null}
        {label ? <span className="text-[13px] font-medium">{label}</span> : null}
        {count ? (
          <span className="text-xs tabular text-subtle-foreground">
            {label ? count : `${count} configured`}
          </span>
        ) : null}
        {inherited ? (
          <Badge size="sm" tone="outline" className="font-normal">
            inherited
          </Badge>
        ) : null}
        {actions ? <span className="ml-auto flex items-center gap-1">{actions}</span> : null}
      </div>
      {error ? (
        <p className="text-xs leading-relaxed text-destructive-foreground">{error}</p>
      ) : hint ? (
        <p className="text-xs leading-relaxed text-muted-foreground">{hint}</p>
      ) : null}
    </div>
  )
}

function BlockActions({
  label,
  edited,
  explicit,
  disabled,
  onUndo,
  onReset,
}: {
  label: string
  edited: boolean
  explicit: boolean
  disabled?: boolean
  onUndo: () => void
  onReset: () => void
}) {
  if (disabled) return null
  if (edited) {
    return (
      <Button variant="ghost" size="xs" onClick={onUndo} aria-label={`Undo ${label}`}>
        <Undo2 />
        Undo
      </Button>
    )
  }
  if (!explicit) return null
  return (
    <Button
      variant="ghost"
      size="xs"
      onClick={onReset}
      aria-label={`Reset ${label} to its default`}
    >
      <RotateCcw />
      Use default
    </Button>
  )
}

function addLabel(schema: RouterStructuredSchema, label: string): string {
  const words = settingLabel(
    schema.item?.label && schema.item.label !== schema.label ? schema.item.label : label,
  )
    .toLowerCase()
    .split(' ')
  const last = words[words.length - 1]
  if (last.length > 3 && last.endsWith('s') && !last.endsWith('ss'))
    words[words.length - 1] = last.slice(0, -1)
  return `Add ${words.join(' ')}`
}

function itemTitle(schema: RouterStructuredSchema, item: unknown, index: number): string {
  const record = isRecord(item) ? item : {}
  for (const key of [
    schema.itemLabelKey,
    'name',
    'label',
    'type',
    'model_alias',
    'mapper_id',
    'env',
  ].filter(Boolean) as string[]) {
    const value = record[key]
    if (typeof value === 'string' && value.trim()) return value
  }
  return `${settingLabel(schema.item?.label ?? schema.label)} ${index + 1}`
}

function pluginItemSchema(base: RouterStructuredSchema, item: unknown): RouterStructuredSchema {
  const type = isRecord(item) && typeof item.type === 'string' ? item.type : ''
  const fields = type ? pluginFieldsFromRouterSchema(type) : []
  if (fields.length === 0 || base.kind !== 'object') return base
  return {
    ...base,
    fields: {
      ...base.fields,
      configuration: {
        kind: 'object',
        label: 'Configuration',
        defaultValue: {},
        fields: Object.fromEntries(
          fields.map((field) => [field.key, generatedStructuredSchema(field)]),
        ),
      },
    },
  }
}

function ListField({ schema, path, keyName, disabled: blocked, bare }: PathFieldProps) {
  const editor = useRouterSettingsEditor()
  const explicitValue = getAt(editor.draft, path)
  const savedValue = getAt(editor.saved, path)
  const explicit = explicitValue !== undefined
  const edited = !deepEqual(explicitValue, savedValue)
  const inheritedValue = savedValue === undefined ? getAt(editor.effective, path) : undefined
  const items = (
    Array.isArray(explicitValue)
      ? explicitValue
      : Array.isArray(inheritedValue)
        ? inheritedValue
        : []
  ) as unknown[]
  const label = settingLabel(schema.label || keyName)
  const disabled = editor.readOnly || blocked
  const isPlugins = path.length === 1 && path[0] === 'plugins'
  const write = (next: unknown[]) => editor.set(path, next)

  return (
    <div className="flex flex-col gap-2.5" data-setting={domId(path)}>
      <BlockHeader
        label={bare ? '' : label}
        hint={bare ? undefined : schema.description}
        inherited={!explicit}
        edited={edited}
        count={items.length ? String(items.length) : undefined}
        error={errorFor(editor, path)}
        actions={
          <BlockActions
            label={label}
            edited={edited}
            explicit={explicit}
            disabled={disabled}
            onUndo={() => editor.restore(path)}
            onReset={() => editor.unset(path)}
          />
        }
      />
      {items.length ? (
        <div className="flex flex-col gap-2">
          {items.map((item, index) => {
            const itemSchema = schema.item
              ? isPlugins
                ? pluginItemSchema(schema.item, item)
                : schema.item
              : undefined
            return (
              <EntryCard
                key={index}
                title={itemTitle(schema, item, index)}
                inherited={!explicit}
                disabled={disabled}
                removeLabel={`Remove ${itemTitle(schema, item, index)}`}
                onRemove={() => write(items.filter((_, itemIndex) => itemIndex !== index))}
              >
                {itemSchema ? (
                  <ValueObject
                    schema={itemSchema}
                    value={item}
                    disabled={disabled}
                    idPrefix={`${domId(path)}-${index}`}
                    onChange={(next) =>
                      write(items.map((entry, entryIndex) => (entryIndex === index ? next : entry)))
                    }
                  />
                ) : (
                  <CodeControl
                    id={`${domId(path)}-${index}`}
                    label={`${label} ${index + 1}`}
                    value={item}
                    disabled={disabled}
                    onChange={(next) =>
                      write(items.map((entry, entryIndex) => (entryIndex === index ? next : entry)))
                    }
                  />
                )}
              </EntryCard>
            )
          })}
        </div>
      ) : (
        <p className="rounded-lg border border-dashed border-border px-3 py-3 text-xs text-subtle-foreground">
          {schema.emptyLabel ?? `No ${label.toLowerCase()} configured.`}
        </p>
      )}
      {!disabled && schema.item ? (
        <Button
          variant="outline"
          size="xs"
          className="w-fit"
          onClick={() =>
            write([...items, createRouterStructuredValue(schema.item as RouterStructuredSchema)])
          }
        >
          <Plus />
          {addLabel(schema, label)}
        </Button>
      ) : null}
    </div>
  )
}

function MapField({ schema, path, keyName, disabled: blocked, bare }: PathFieldProps) {
  const editor = useRouterSettingsEditor()
  const [name, setName] = React.useState('')
  const draftEntries = getAt(editor.draft, path)
  const savedEntries = getAt(editor.saved, path)
  const effectiveEntries = getAt(editor.effective, path)
  const explicitMap = isRecord(draftEntries) ? draftEntries : {}
  const inheritedMap = isRecord(effectiveEntries) ? effectiveEntries : {}
  const savedMap = isRecord(savedEntries) ? savedEntries : {}
  const names = [
    ...new Set([
      ...Object.keys(explicitMap),
      ...Object.keys(inheritedMap).filter((key) => !(key in savedMap)),
    ]),
  ]
  const label = settingLabel(schema.label || keyName)
  const disabled = editor.readOnly || blocked
  const edited = !deepEqual(draftEntries, savedEntries)
  const trimmed = name.trim()
  const nameProblem = trimmed && names.includes(trimmed) ? `${trimmed} already exists.` : null

  return (
    <div className="flex flex-col gap-2.5" data-setting={domId(path)}>
      <BlockHeader
        label={bare ? '' : label}
        hint={bare ? undefined : schema.description}
        inherited={false}
        edited={edited}
        count={names.length ? String(names.length) : undefined}
        error={errorFor(editor, path)}
        actions={
          <BlockActions
            label={label}
            edited={edited}
            explicit={false}
            disabled={disabled}
            onUndo={() => editor.restore(path)}
            onReset={() => editor.unset(path)}
          />
        }
      />
      {names.length ? (
        <div className="flex flex-col gap-2">
          {names.map((entry) => {
            const isExplicit = entry in explicitMap
            const value = isExplicit ? explicitMap[entry] : inheritedMap[entry]
            const entryPath = [...path, entry]
            const entryEdited = !deepEqual(
              getAt(editor.draft, entryPath),
              getAt(editor.saved, entryPath),
            )
            return (
              <EntryCard
                key={entry}
                title={entry}
                mono
                inherited={!isExplicit}
                edited={entryEdited}
                isNew={isExplicit && !(entry in savedMap)}
                disabled={disabled}
                removeLabel={`Remove ${entry}`}
                onRemove={isExplicit ? () => editor.unset(entryPath) : undefined}
                onUndo={entryEdited ? () => editor.restore(entryPath) : undefined}
              >
                {schema.item ? (
                  <ValueObject
                    schema={schema.item}
                    value={value}
                    disabled={disabled}
                    idPrefix={`${domId(path)}-${entry}`}
                    onChange={(next) => editor.set(entryPath, next)}
                  />
                ) : null}
              </EntryCard>
            )
          })}
        </div>
      ) : (
        <p className="rounded-lg border border-dashed border-border px-3 py-3 text-xs text-subtle-foreground">
          No entries yet.
        </p>
      )}
      {!disabled && schema.item ? (
        <form
          className="flex items-start gap-2"
          onSubmit={(event) => {
            event.preventDefault()
            if (!trimmed || nameProblem) return
            editor.set(
              [...path, trimmed],
              createRouterStructuredValue(schema.item as RouterStructuredSchema),
            )
            setName('')
          }}
        >
          <div className="flex flex-col gap-1">
            <Input
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="New entry name"
              aria-label={`New ${label} entry name`}
              aria-invalid={Boolean(nameProblem) || undefined}
              className="h-8 w-56 font-mono text-[12.5px]"
            />
            {nameProblem ? (
              <p className="text-xs text-destructive-foreground">{nameProblem}</p>
            ) : null}
          </div>
          <Button
            type="submit"
            variant="outline"
            size="sm"
            disabled={!trimmed || Boolean(nameProblem)}
          >
            <Plus />
            Add entry
          </Button>
        </form>
      ) : null}
    </div>
  )
}

function KeyValueField({ schema, path, keyName, disabled: blocked, bare }: PathFieldProps) {
  const editor = useRouterSettingsEditor()
  const explicitValue = getAt(editor.draft, path)
  const savedValue = getAt(editor.saved, path)
  const explicit = explicitValue !== undefined
  const edited = !deepEqual(explicitValue, savedValue)
  const inheritedValue = savedValue === undefined ? getAt(editor.effective, path) : undefined
  const label = settingLabel(schema.label || keyName)
  const disabled = editor.readOnly || blocked
  return (
    <div className="flex flex-col gap-2" data-setting={domId(path)}>
      <BlockHeader
        label={bare ? '' : label}
        hint={bare ? undefined : schema.description}
        inherited={!explicit}
        edited={edited}
        error={errorFor(editor, path)}
        actions={
          <BlockActions
            label={label}
            edited={edited}
            explicit={explicit}
            disabled={disabled}
            onUndo={() => editor.restore(path)}
            onReset={() => editor.unset(path)}
          />
        }
      />
      <KeyValueControl
        label={label}
        value={isRecord(explicitValue) ? (explicitValue as Record<string, string>) : undefined}
        inheritedValue={
          isRecord(inheritedValue) ? (inheritedValue as Record<string, string>) : undefined
        }
        disabled={disabled}
        onChange={(next) => editor.set(path, next)}
      />
    </div>
  )
}

function CodeField({ schema, path, keyName, disabled: blocked, bare }: PathFieldProps) {
  const editor = useRouterSettingsEditor()
  const explicitValue = getAt(editor.draft, path)
  const savedValue = getAt(editor.saved, path)
  const explicit = explicitValue !== undefined
  const edited = !deepEqual(explicitValue, savedValue)
  const inheritedValue = savedValue === undefined ? getAt(editor.effective, path) : undefined
  const label = settingLabel(schema.label || keyName)
  const disabled = editor.readOnly || blocked
  const key = pathKey(path)
  const { reportInvalid } = editor
  const onInvalid = React.useCallback(
    (message: string | null) => reportInvalid(key, message),
    [key, reportInvalid],
  )
  return (
    <div className="flex flex-col gap-2" data-setting={domId(path)}>
      <BlockHeader
        label={bare ? '' : label}
        hint={bare ? undefined : schema.description}
        inherited={!explicit}
        edited={edited}
        error={errorFor(editor, path)}
        actions={
          <BlockActions
            label={label}
            edited={edited}
            explicit={explicit}
            disabled={disabled}
            onUndo={() => editor.restore(path)}
            onReset={() => editor.unset(path)}
          />
        }
      />
      <CodeControl
        id={domId(path)}
        label={label}
        value={explicitValue}
        inheritedValue={inheritedValue}
        disabled={disabled}
        onInvalid={onInvalid}
        onChange={(next) => (next === undefined ? editor.unset(path) : editor.set(path, next))}
      />
    </div>
  )
}

export function EntryCard({
  title,
  mono,
  inherited,
  edited,
  isNew,
  disabled,
  removeLabel,
  onRemove,
  onUndo,
  defaultOpen,
  children,
}: {
  title: string
  mono?: boolean
  inherited?: boolean
  edited?: boolean
  isNew?: boolean
  disabled?: boolean
  removeLabel: string
  onRemove?: () => void
  onUndo?: () => void
  defaultOpen?: boolean
  children: React.ReactNode
}) {
  const [open, setOpen] = React.useState(Boolean(defaultOpen))
  return (
    <Collapsible
      open={open}
      onOpenChange={setOpen}
      className="rounded-lg border border-border bg-card"
    >
      <div className="flex min-h-10 items-center gap-2 pr-1.5">
        <CollapsibleTrigger asChild>
          <button
            type="button"
            className="flex min-w-0 flex-1 items-center gap-2 rounded-lg py-2 pl-3 text-left outline-none focus-visible:ring-[3px] focus-visible:ring-ring/40"
          >
            <ChevronRight
              className={cn(
                'size-4 shrink-0 text-subtle-foreground transition-transform duration-200',
                open && 'rotate-90',
              )}
            />
            {edited ? (
              <span
                className="size-1.5 shrink-0 rounded-full bg-primary"
                role="img"
                aria-label="Edited"
              />
            ) : null}
            <span
              className={cn('truncate text-[13px] font-medium', mono && 'font-mono text-[12.5px]')}
            >
              {title}
            </span>
            {inherited ? (
              <Badge size="sm" tone="outline" className="font-normal">
                inherited
              </Badge>
            ) : null}
            {isNew ? (
              <Badge size="sm" tone="brand">
                new
              </Badge>
            ) : null}
          </button>
        </CollapsibleTrigger>
        {!disabled && onUndo ? (
          <Tooltip content="Undo changes to this entry">
            <Button variant="ghost" size="icon-xs" onClick={onUndo} aria-label={`Undo ${title}`}>
              <Undo2 />
            </Button>
          </Tooltip>
        ) : null}
        {!disabled && onRemove ? (
          <Tooltip content="Remove">
            <Button
              variant="ghost"
              size="icon-xs"
              onClick={onRemove}
              aria-label={removeLabel}
              className="hover:text-destructive-foreground"
            >
              <Trash2 />
            </Button>
          </Tooltip>
        ) : null}
      </div>
      <CollapsibleContent>
        <div className="flex flex-col gap-4 border-t border-border px-3 pb-4 pt-4 @container">
          {children}
        </div>
      </CollapsibleContent>
    </Collapsible>
  )
}

/** A value-bound object editor for list items and map entries, where the whole value is explicit. */
export function ValueObject({
  schema,
  value,
  disabled,
  idPrefix,
  onChange,
}: {
  schema: RouterStructuredSchema
  value: unknown
  disabled?: boolean
  idPrefix: string
  onChange: (next: unknown) => void
}) {
  const record = isRecord(value) ? value : {}
  const entries = childEntries(schema)
  const update = (key: string, next: unknown) => {
    const copy = { ...record }
    if (next === undefined) delete copy[key]
    else copy[key] = next
    onChange(copy)
  }
  const scalars = entries.filter(
    ([, child]) => isScalarSchema(child) || effectiveKind(child) === 'select',
  )
  const blocks = entries.filter((entry) => !scalars.includes(entry))
  return (
    <>
      {scalars.length ? (
        <FieldGrid>
          {scalars.map(([key, child]) => (
            <ValueField
              key={key}
              schema={child}
              keyName={key}
              value={record[key]}
              disabled={disabled}
              id={`${idPrefix}-${key}`}
              onChange={(next) => update(key, next)}
            />
          ))}
        </FieldGrid>
      ) : null}
      {blocks.map(([key, child]) => (
        <ValueField
          key={key}
          schema={child}
          keyName={key}
          value={record[key]}
          disabled={disabled}
          id={`${idPrefix}-${key}`}
          onChange={(next) => update(key, next)}
        />
      ))}
    </>
  )
}

function ValueField({
  schema,
  keyName,
  value,
  disabled,
  id,
  onChange,
}: {
  schema: RouterStructuredSchema
  keyName: string
  value: unknown
  disabled?: boolean
  id: string
  onChange: (next: unknown) => void
}) {
  const kind = effectiveKind(schema)
  const { label, unit } = settingPresentation(keyName, schema.label || keyName)
  const missing =
    schema.required &&
    (value === undefined || value === '' || (Array.isArray(value) && value.length === 0))
  const error = typeof value === 'string' && kind === 'number' ? 'Enter a number.' : undefined

  if (kind === 'object') {
    return (
      <EntryCard title={label} removeLabel={label} disabled={disabled}>
        <ValueObject
          schema={schema}
          value={value}
          disabled={disabled}
          idPrefix={id}
          onChange={onChange}
        />
      </EntryCard>
    )
  }
  if (kind === 'object-list') {
    const items = Array.isArray(value) ? value : []
    return (
      <div className="flex flex-col gap-2">
        <BlockHeader
          label={label}
          hint={schema.description}
          inherited={false}
          edited={false}
          count={items.length ? String(items.length) : undefined}
        />
        {items.map((item, index) => (
          <EntryCard
            key={index}
            title={itemTitle(schema, item, index)}
            disabled={disabled}
            removeLabel={`Remove ${itemTitle(schema, item, index)}`}
            onRemove={() => onChange(items.filter((_, itemIndex) => itemIndex !== index))}
          >
            {schema.item ? (
              <ValueObject
                schema={schema.item}
                value={item}
                disabled={disabled}
                idPrefix={`${id}-${index}`}
                onChange={(next) =>
                  onChange(items.map((entry, entryIndex) => (entryIndex === index ? next : entry)))
                }
              />
            ) : null}
          </EntryCard>
        ))}
        {!disabled && schema.item ? (
          <Button
            variant="outline"
            size="xs"
            className="w-fit"
            onClick={() =>
              onChange([
                ...items,
                createRouterStructuredValue(schema.item as RouterStructuredSchema),
              ])
            }
          >
            <Plus />
            {addLabel(schema, label)}
          </Button>
        ) : null}
      </div>
    )
  }
  if (kind === 'string-map') {
    return (
      <div className="flex flex-col gap-2">
        <BlockHeader label={label} hint={schema.description} inherited={false} edited={false} />
        <KeyValueControl
          label={label}
          value={isRecord(value) ? (value as Record<string, string>) : {}}
          disabled={disabled}
          onChange={onChange}
        />
      </div>
    )
  }
  if (kind === 'json' || kind === 'object-map') {
    return (
      <div className="flex flex-col gap-2">
        <BlockHeader label={label} hint={schema.description} inherited={false} edited={false} />
        <CodeControl id={id} label={label} value={value} disabled={disabled} onChange={onChange} />
      </div>
    )
  }

  let control: React.ReactNode
  if (kind === 'boolean') {
    control = (
      <BooleanControl
        id={id}
        label={label}
        checked={value === true}
        disabled={disabled}
        onChange={onChange}
      />
    )
  } else if (kind === 'select') {
    control = (
      <SelectControl
        id={id}
        label={label}
        value={typeof value === 'string' && value !== '' ? value : undefined}
        placeholder={schema.placeholder ?? 'Choose…'}
        options={selectOptions(schema)}
        disabled={disabled}
        invalid={Boolean(missing)}
        onChange={onChange}
      />
    )
  } else if (kind === 'number') {
    control = (
      <NumberControl
        id={id}
        value={value}
        unit={unit}
        placeholder={schema.placeholder}
        disabled={disabled}
        invalid={Boolean(error || missing)}
        onChange={onChange}
      />
    )
  } else if (kind === 'string-list' || kind === 'number-list') {
    control = (
      <ChipsControl
        id={id}
        label={label}
        numeric={kind === 'number-list'}
        values={Array.isArray(value) ? (value as Array<string | number>) : []}
        placeholder={schema.placeholder}
        disabled={disabled}
        onChange={onChange}
      />
    )
  } else {
    control = (
      <TextControl
        id={id}
        label={label}
        value={typeof value === 'string' ? value : value === undefined ? undefined : String(value)}
        placeholder={schema.placeholder}
        secret={isSecretSetting(schema, keyName)}
        unit={unit}
        disabled={disabled}
        invalid={Boolean(missing)}
        onChange={onChange}
      />
    )
  }
  return (
    <FieldShell
      id={id}
      label={label}
      required={schema.required}
      hint={schema.description}
      error={error ?? (missing ? `${label} is required.` : undefined)}
    >
      {control}
    </FieldShell>
  )
}
