import * as React from 'react'
import {
  ArrowDown,
  ArrowUp,
  ChevronDown,
  ChevronRight,
  Plus,
  RotateCcw,
  Trash2,
} from 'lucide-react'

import type { FieldSchema } from '../lib/dslSchemas'
import {
  Badge,
  Button,
  ChipInput,
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
  Slider,
  Textarea,
  Tooltip,
  cn,
} from '../ui'
import { RuleTreeEditor } from './ConfigPageDecisionRulesEditor'
import {
  BooleanSwitch,
  BoundsInput,
  JsonInput,
  KeyValueInput,
  NumberInput,
  OptionSelect,
  PatternListInput,
  SuggestInput,
  SuggestionChips,
} from './ConfigPageSchemaFieldsEditorControls'
import {
  boundsSummary,
  isLongTextField,
  isPatternList,
  isPredicateSchema,
  isSliderField,
  useSchemaFormHints,
} from './ConfigPageSchemaFieldsEditorSupport'

type Value = Record<string, unknown>

function isRecord(value: unknown): value is Value {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function hasValue(value: unknown): boolean {
  return value !== undefined
}

const WIDE_TYPES = new Set<FieldSchema['type']>([
  'string[]',
  'number[]',
  'object',
  'object[]',
  'key-value',
  'json',
  'rule',
])

function isWide(field: FieldSchema): boolean {
  return (
    WIDE_TYPES.has(field.type) || isLongTextField(field) || (field.description?.length ?? 0) > 96
  )
}

function fieldId(path: readonly string[]): string {
  return `field-${path.join('-').replace(/[^\w-]/g, '_')}`
}

function humanKey(label: string): string {
  return label
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_|_$/g, '')
}

function displayScalar(value: unknown): string {
  if (value === undefined) return 'Not set'
  if (typeof value === 'string') return value || '""'
  return JSON.stringify(value)
}

export interface SchemaFieldsEditorProps {
  schema: FieldSchema[]
  value: Value
  onChange?: (value: Value) => void
  readOnly?: boolean
  /** Path of this value in its entity, used for ids and suggestions. */
  path?: readonly string[]
  /** Starts with every optional setting listed, not only the configured ones. */
  showAllByDefault?: boolean
  /** Hides the note about fields this schema does not describe. */
  hideAdditional?: boolean
  className?: string
}

/**
 * Edits a configuration object from its field schema. Configured and required
 * fields come first; the rest stay one click away. Fields the schema does not
 * describe are preserved and listed, never dropped.
 */
export default function ConfigPageSchemaFieldsEditor({
  schema,
  value,
  onChange,
  readOnly = false,
  path = [],
  showAllByDefault = false,
  hideAdditional = false,
  className,
}: SchemaFieldsEditorProps) {
  const [showAll, setShowAll] = React.useState(showAllByDefault)
  const known = new Set(schema.map((field) => field.key))
  const configured = schema.filter((field) => field.required || hasValue(value[field.key]))
  const optional = schema.filter((field) => !field.required && !hasValue(value[field.key]))
  const listed = readOnly
    ? schema.filter((field) => hasValue(value[field.key]))
    : showAll
      ? schema
      : configured
  // Switches read best together after the fields they qualify.
  const visible = [
    ...listed.filter((field) => field.type !== 'boolean'),
    ...listed.filter((field) => field.type === 'boolean'),
  ]
  const additional = Object.entries(value).filter(([key]) => !known.has(key))

  const setField = (key: string, next: unknown) => {
    const updated = { ...value }
    if (next === undefined) delete updated[key]
    else updated[key] = next
    onChange?.(updated)
  }

  return (
    <div className={cn('flex flex-col gap-4', className)}>
      {visible.length > 0 ? (
        <div className="grid grid-cols-1 gap-x-4 gap-y-4 sm:grid-cols-2">
          {visible.map((field) => (
            <SchemaField
              key={field.key}
              field={field}
              value={value[field.key]}
              onChange={(next) => setField(field.key, next)}
              readOnly={readOnly}
              path={[...path, field.key]}
              patterns={isPatternList(field, value)}
            />
          ))}
        </div>
      ) : readOnly ? (
        <p className="text-[13px] text-muted-foreground">Uses the defaults for every setting.</p>
      ) : null}
      {!readOnly && !showAllByDefault && optional.length > 0 ? (
        <button
          type="button"
          onClick={() => setShowAll((current) => !current)}
          className="flex w-fit items-center gap-1.5 rounded-md px-1 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground focus-visible:ring-[3px] focus-visible:ring-ring/40 focus-visible:outline-none"
          aria-expanded={showAll}
        >
          {showAll ? <ChevronDown className="size-3.5" /> : <ChevronRight className="size-3.5" />}
          {showAll
            ? 'Show configured settings only'
            : `${optional.length} more ${optional.length === 1 ? 'setting' : 'settings'} at their defaults`}
        </button>
      ) : null}
      {!hideAdditional && additional.length > 0 ? <AdditionalFields entries={additional} /> : null}
    </div>
  )
}

function AdditionalFields({ entries }: { entries: [string, unknown][] }) {
  return (
    <section
      aria-label="Additional configuration fields"
      className="rounded-lg border border-dashed border-border-strong bg-surface-subtle/60 px-3 py-2.5"
    >
      <p className="text-xs font-medium text-foreground">Kept as written</p>
      <p className="mt-0.5 text-xs text-muted-foreground">
        The form does not describe these fields; saving keeps them unchanged.
      </p>
      <dl className="mt-2 flex flex-col gap-1">
        {entries.map(([key, item]) => (
          <div key={key} className="flex min-w-0 gap-2 font-mono text-[12px]">
            <dt className="shrink-0 text-muted-foreground">{key}:</dt>
            <dd className="min-w-0 truncate text-foreground/90">{displayScalar(item)}</dd>
          </div>
        ))}
      </dl>
    </section>
  )
}

function FieldLabel({
  field,
  htmlFor,
  onReset,
}: {
  field: FieldSchema
  htmlFor?: string
  onReset?: () => void
}) {
  const showKey = humanKey(field.label) !== field.key.toLowerCase()
  return (
    <div className="flex min-h-5 items-center justify-between gap-2">
      <label
        htmlFor={htmlFor}
        className="flex min-w-0 items-center gap-1 text-[13px] font-medium leading-none text-foreground"
      >
        <span className="truncate">{field.label}</span>
        {field.required ? (
          <span aria-hidden className="text-destructive-foreground">
            *
          </span>
        ) : null}
      </label>
      <span className="flex shrink-0 items-center gap-1">
        {showKey ? (
          <span className="font-mono text-[11px] text-subtle-foreground">{field.key}</span>
        ) : null}
        {onReset ? (
          <Tooltip content="Reset to default">
            <button
              type="button"
              aria-label={`Reset ${field.label}`}
              onClick={onReset}
              className="grid size-5 place-items-center rounded text-subtle-foreground transition-colors hover:bg-accent hover:text-foreground focus-visible:ring-[3px] focus-visible:ring-ring/40 focus-visible:outline-none"
            >
              <RotateCcw className="size-3" />
            </button>
          </Tooltip>
        ) : null}
      </span>
    </div>
  )
}

function FieldHint({ field }: { field: FieldSchema }) {
  return field.description ? (
    <p className="text-xs leading-relaxed text-muted-foreground">{field.description}</p>
  ) : null
}

export interface SchemaFieldProps {
  field: FieldSchema
  value: unknown
  onChange: (value: unknown) => void
  readOnly: boolean
  path: readonly string[]
  /** Entries may contain commas, so only Enter separates them. */
  patterns?: boolean
}

/** One field of a schema-driven form, for editors that lay out their own fields. */
export function SchemaField(props: SchemaFieldProps) {
  const { field, value, onChange, readOnly, path, patterns = false } = props
  const hints = useSchemaFormHints()
  const id = fieldId(path)
  const resettable = !readOnly && !field.required && value !== undefined
  const reset = resettable ? () => onChange(undefined) : undefined
  const wide = isWide(field) ? 'sm:col-span-2' : undefined

  if (field.type === 'object') {
    return isPredicateSchema(field) ? (
      <div className={cn('flex flex-col gap-1.5', 'sm:col-span-2')}>
        <FieldLabel field={field} onReset={reset} />
        <BoundsInput
          idPrefix={id}
          value={isRecord(value) ? value : {}}
          onChange={(next) => onChange(next ?? (field.required ? {} : undefined))}
          disabled={readOnly}
        />
        <FieldHint field={field} />
      </div>
    ) : (
      <ObjectField {...props} />
    )
  }
  if (field.type === 'object[]') return <ObjectListField {...props} />

  if (field.type === 'boolean') {
    return (
      <div
        className={cn(
          'flex items-start justify-between gap-3 rounded-lg border border-border bg-surface px-3 py-2.5',
          wide,
        )}
      >
        <div className="flex min-w-0 flex-col gap-1">
          <FieldLabel field={field} htmlFor={id} onReset={reset} />
          <FieldHint field={field} />
        </div>
        <BooleanSwitch
          id={id}
          value={value}
          onChange={onChange}
          disabled={readOnly}
          ariaLabel={field.label}
        />
      </div>
    )
  }

  let control: React.ReactNode
  switch (field.type) {
    case 'number':
      control = isSliderField(field) ? (
        <div className="flex items-center gap-3">
          <Slider
            aria-label={field.label}
            value={typeof value === 'number' ? value : (field.min ?? 0)}
            min={field.min}
            max={field.max}
            step={
              (field.max ?? 1) - (field.min ?? 0) <= 1
                ? 0.01
                : (field.max ?? 1) - (field.min ?? 0) <= 10
                  ? 0.1
                  : 1
            }
            tone={hints.accent ?? 'blue'}
            disabled={readOnly}
            className={cn(value === undefined && 'opacity-60')}
            onValueChange={(next) => onChange(next)}
          />
          <NumberInput
            id={id}
            value={value}
            onChange={onChange}
            min={field.min}
            max={field.max}
            placeholder={field.placeholder ?? 'default'}
            disabled={readOnly}
            className="h-8 w-20 shrink-0 text-center"
          />
        </div>
      ) : (
        <NumberInput
          id={id}
          value={value}
          onChange={onChange}
          min={field.min}
          max={field.max}
          placeholder={field.placeholder}
          disabled={readOnly}
        />
      )
      break
    case 'select':
      control = (
        <OptionSelect
          id={id}
          ariaLabel={field.label}
          value={value}
          options={field.options ?? []}
          disabledOptions={field.disabledOptions}
          required={field.required}
          disabled={readOnly}
          onChange={onChange}
        />
      )
      break
    case 'string[]': {
      const items = Array.isArray(value)
        ? value.filter((item): item is string => typeof item === 'string')
        : []
      const suggestions = hints.suggest?.(field, path) ?? field.options ?? []
      control = (
        <div className="flex flex-col gap-1.5">
          {patterns ? (
            <PatternListInput
              id={id}
              ariaLabel={field.label}
              value={items}
              onChange={onChange}
              placeholder={field.placeholder}
              disabled={readOnly}
            />
          ) : (
            <ChipInput
              id={id}
              aria-label={field.label}
              value={items}
              caseSensitive
              onChange={onChange}
              placeholder={field.placeholder || 'Type and press Enter'}
              disabled={readOnly}
            />
          )}
          {readOnly ? null : (
            <SuggestionChips value={items} suggestions={suggestions} onAdd={onChange} />
          )}
        </div>
      )
      break
    }
    case 'number[]': {
      const items = Array.isArray(value)
        ? value.filter((item): item is number => typeof item === 'number')
        : []
      control = (
        <ChipInput
          id={id}
          aria-label={field.label}
          value={items.map(String)}
          caseSensitive
          validate={(candidate) =>
            Number.isFinite(Number(candidate)) ? null : `${candidate} is not a number`
          }
          onChange={(next) => onChange(next.map(Number))}
          placeholder={field.placeholder || 'Add a number and press Enter'}
          disabled={readOnly}
        />
      )
      break
    }
    case 'key-value':
      control = (
        <KeyValueInput
          value={isRecord(value) ? value : {}}
          onChange={(next) =>
            onChange(Object.keys(next).length || field.required ? next : undefined)
          }
          disabled={readOnly}
          keyLabel={field.keyLabel}
          valueLabel={field.valueLabel}
        />
      )
      break
    case 'json':
      control = (
        <JsonInput
          id={id}
          value={value}
          onChange={onChange}
          disabled={readOnly}
          placeholder={field.placeholder}
        />
      )
      break
    case 'rule':
      control = (
        <RuleTreeEditor
          value={isRecord(value) ? value : {}}
          onChange={(next) => onChange(next)}
          readOnly={readOnly}
          signalTypes={field.options}
          allowAlways={false}
        />
      )
      break
    default: {
      const text = typeof value === 'string' ? value : value === undefined ? '' : String(value)
      const setText = (next: string) => onChange(next === '' ? undefined : next)
      control = isLongTextField(field) ? (
        <Textarea
          id={id}
          value={text}
          disabled={readOnly}
          placeholder={field.placeholder}
          rows={Math.min(8, Math.max(2, Math.ceil(text.length / 80)))}
          onChange={(event) => setText(event.target.value)}
        />
      ) : (
        <SuggestInput
          id={id}
          value={text}
          onChange={setText}
          suggestions={hints.suggest?.(field, path) ?? field.options ?? []}
          placeholder={field.placeholder}
          disabled={readOnly}
          mono={!field.key.endsWith('description')}
        />
      )
    }
  }

  return (
    <div className={cn('flex min-w-0 flex-col gap-1.5', wide)}>
      <FieldLabel field={field} htmlFor={id} onReset={reset} />
      {control}
      <FieldHint field={field} />
    </div>
  )
}

function ObjectField({ field, value, onChange, readOnly, path }: SchemaFieldProps) {
  const record = isRecord(value) ? value : undefined
  if (!record) {
    if (readOnly) return null
    return (
      <div className="flex items-center justify-between gap-3 rounded-lg border border-dashed border-border-strong px-3 py-2.5 sm:col-span-2">
        <div className="flex min-w-0 flex-col gap-1">
          <FieldLabel field={field} />
          {field.description ? (
            <FieldHint field={field} />
          ) : (
            <p className="text-xs text-subtle-foreground">Not set</p>
          )}
        </div>
        <Button type="button" variant="outline" size="xs" onClick={() => onChange({})}>
          <Plus />
          Configure
        </Button>
      </div>
    )
  }
  const fields = field.fields ?? []
  return (
    <section
      aria-label={field.label}
      className="flex flex-col gap-3 rounded-lg border border-border bg-surface-subtle/50 px-3 pb-3 pt-2.5 sm:col-span-2"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex min-w-0 flex-col gap-1">
          <FieldLabel field={field} />
          <FieldHint field={field} />
        </div>
        {!readOnly && !field.required ? (
          <Button
            type="button"
            variant="ghost"
            size="xs"
            onClick={() => onChange(undefined)}
            aria-label={`Remove ${field.label}`}
          >
            <Trash2 />
            Remove
          </Button>
        ) : null}
      </div>
      {fields.length > 0 ? (
        <ConfigPageSchemaFieldsEditor
          schema={fields}
          value={record}
          onChange={(next) => onChange(next)}
          readOnly={readOnly}
          path={path}
        />
      ) : (
        <JsonInput
          value={record}
          onChange={(next) => onChange(isRecord(next) ? next : {})}
          disabled={readOnly}
        />
      )}
    </section>
  )
}

function itemTitle(field: FieldSchema, item: unknown, index: number): string {
  const base = field.itemLabel || field.label.replace(/s$/, '')
  const candidate = field.itemLabelKey && isRecord(item) ? item[field.itemLabelKey] : undefined
  return typeof candidate === 'string' && candidate.trim()
    ? candidate.trim()
    : `${base} ${index + 1}`
}

function itemSubtitle(field: FieldSchema, item: unknown): string {
  if (!isRecord(item)) return ''
  return (field.fields ?? [])
    .filter((nested) => nested.key !== field.itemLabelKey && item[nested.key] !== undefined)
    .slice(0, 3)
    .map((nested) => {
      const nestedValue = item[nested.key]
      if (isPredicateSchema(nested)) return boundsSummary(nestedValue)
      if (isRecord(nestedValue) || Array.isArray(nestedValue)) return nested.key
      return `${nested.key}: ${String(nestedValue)}`
    })
    .filter(Boolean)
    .join(' · ')
}

function ObjectListField({ field, value, onChange, readOnly, path }: SchemaFieldProps) {
  const items = Array.isArray(value) ? value : []
  const [open, setOpen] = React.useState<Set<number>>(
    () => new Set(items.length <= 2 ? items.map((_, index) => index) : []),
  )
  const itemLabel = (field.itemLabel || field.label.replace(/s$/, '')).toLowerCase()
  const update = (next: unknown[]) => onChange(next)
  const move = (index: number, delta: number) => {
    const target = index + delta
    if (target < 0 || target >= items.length) return
    const next = [...items]
    ;[next[index], next[target]] = [next[target], next[index]]
    update(next)
    setOpen((current) => {
      const swapped = new Set<number>()
      current.forEach((position) =>
        swapped.add(position === index ? target : position === target ? index : position),
      )
      return swapped
    })
  }
  const remove = (index: number) => {
    update(items.filter((_, position) => position !== index))
    setOpen(
      (current) =>
        new Set(
          [...current]
            .filter((position) => position !== index)
            .map((position) => (position > index ? position - 1 : position)),
        ),
    )
  }
  const toggle = (index: number) =>
    setOpen((current) => {
      const next = new Set(current)
      if (next.has(index)) next.delete(index)
      else next.add(index)
      return next
    })

  if (readOnly && items.length === 0) return null
  return (
    <section aria-label={field.label} className="flex flex-col gap-2 sm:col-span-2">
      <div className="flex items-end justify-between gap-3">
        <div className="flex min-w-0 flex-col gap-1">
          <div className="flex items-center gap-2">
            <FieldLabel field={field} />
            <Badge size="sm" tone="neutral" className="font-mono tabular">
              {items.length}
            </Badge>
          </div>
          <FieldHint field={field} />
        </div>
      </div>
      {items.length === 0 ? (
        <p className="rounded-lg border border-dashed border-border-strong px-3 py-3 text-xs text-muted-foreground">
          {field.emptyLabel || `No ${field.label.toLowerCase()} configured.`}
        </p>
      ) : (
        <ol className="flex flex-col gap-2">
          {items.map((item, index) => {
            const expanded = open.has(index)
            return (
              <li key={index} className="rounded-lg border border-border bg-surface shadow-xs">
                <Collapsible open={expanded} onOpenChange={() => toggle(index)}>
                  <div className="flex items-center gap-2 px-2.5 py-2">
                    <CollapsibleTrigger asChild>
                      <button
                        type="button"
                        className="flex min-w-0 flex-1 items-center gap-2 rounded-md text-left focus-visible:ring-[3px] focus-visible:ring-ring/40 focus-visible:outline-none"
                        aria-label={`${expanded ? 'Collapse' : 'Expand'} ${itemTitle(field, item, index)}`}
                      >
                        {expanded ? (
                          <ChevronDown className="size-3.5 shrink-0 text-subtle-foreground" />
                        ) : (
                          <ChevronRight className="size-3.5 shrink-0 text-subtle-foreground" />
                        )}
                        <span className="w-5 shrink-0 font-mono text-[11px] tabular text-subtle-foreground">
                          {index + 1}
                        </span>
                        <span className="truncate font-mono text-[12.5px] font-medium">
                          {itemTitle(field, item, index)}
                        </span>
                        <span className="min-w-0 truncate text-xs text-subtle-foreground">
                          {itemSubtitle(field, item)}
                        </span>
                      </button>
                    </CollapsibleTrigger>
                    {readOnly ? null : (
                      <span className="flex shrink-0 items-center">
                        <Button
                          type="button"
                          variant="ghost"
                          size="icon-xs"
                          aria-label={`Move ${itemTitle(field, item, index)} up`}
                          disabled={index === 0}
                          onClick={() => move(index, -1)}
                        >
                          <ArrowUp />
                        </Button>
                        <Button
                          type="button"
                          variant="ghost"
                          size="icon-xs"
                          aria-label={`Move ${itemTitle(field, item, index)} down`}
                          disabled={index === items.length - 1}
                          onClick={() => move(index, 1)}
                        >
                          <ArrowDown />
                        </Button>
                        <Button
                          type="button"
                          variant="ghost"
                          size="icon-xs"
                          aria-label={`Remove ${itemTitle(field, item, index)}`}
                          onClick={() => remove(index)}
                        >
                          <Trash2 />
                        </Button>
                      </span>
                    )}
                  </div>
                  <CollapsibleContent>
                    <div className="border-t border-border px-3 py-3">
                      {(field.fields ?? []).length > 0 ? (
                        <ConfigPageSchemaFieldsEditor
                          schema={field.fields ?? []}
                          value={isRecord(item) ? item : {}}
                          onChange={(next) =>
                            update(
                              items.map((entry, position) => (position === index ? next : entry)),
                            )
                          }
                          readOnly={readOnly}
                          path={[...path, String(index)]}
                        />
                      ) : (
                        <JsonInput
                          value={item}
                          onChange={(next) =>
                            update(
                              items.map((entry, position) => (position === index ? next : entry)),
                            )
                          }
                          disabled={readOnly}
                        />
                      )}
                    </div>
                  </CollapsibleContent>
                </Collapsible>
              </li>
            )
          })}
        </ol>
      )}
      {readOnly ? null : (
        <Button
          type="button"
          variant="outline"
          size="xs"
          className="w-fit"
          onClick={() => {
            update([...items, {}])
            setOpen((current) => new Set(current).add(items.length))
          }}
        >
          <Plus />
          {field.addLabel || `Add ${itemLabel}`}
        </Button>
      )}
    </section>
  )
}
