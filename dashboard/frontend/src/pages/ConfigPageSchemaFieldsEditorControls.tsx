import * as React from 'react'
import { Plus, Trash2, X } from 'lucide-react'

import {
  Button,
  Input,
  Segmented,
  Select,
  SelectContent,
  SelectItem,
  SelectSeparator,
  SelectTrigger,
  SelectValue,
  Switch,
  Textarea,
  cn,
  mergeChips,
} from '../ui'
import { BOUND_LABEL, type BoundKey } from './ConfigPageSchemaFieldsEditorSupport'

// Radix Select reserves the empty value, so "unset" travels under this key.
const UNSET = '__unset__'
const SEGMENTED_MAX_OPTIONS = 3
const SEGMENTED_MAX_LABEL = 14

/** Free text with suggestions from the configuration, such as model or deployment names. */
export function SuggestInput({
  id,
  value,
  onChange,
  suggestions = [],
  placeholder,
  disabled,
  mono = false,
  className,
  invalid,
  ariaLabel,
}: {
  id?: string
  value: string
  onChange: (value: string) => void
  suggestions?: readonly string[]
  placeholder?: string
  disabled?: boolean
  mono?: boolean
  className?: string
  invalid?: boolean
  ariaLabel?: string
}) {
  const [open, setOpen] = React.useState(false)
  const [active, setActive] = React.useState(0)
  const listId = React.useId()
  const query = value.trim().toLowerCase()
  const matches = React.useMemo(
    () =>
      [...new Set(suggestions)]
        .filter((item) => item && item !== value && (!query || item.toLowerCase().includes(query)))
        .slice(0, 50),
    [suggestions, value, query],
  )
  const showList = open && matches.length > 0 && !disabled
  const pick = (item: string) => {
    onChange(item)
    setOpen(false)
  }
  return (
    <div className={cn('relative', className)}>
      <Input
        id={id}
        value={value}
        disabled={disabled}
        placeholder={placeholder}
        aria-label={ariaLabel}
        aria-invalid={invalid || undefined}
        role={suggestions.length ? 'combobox' : undefined}
        aria-expanded={suggestions.length ? showList : undefined}
        aria-controls={suggestions.length ? listId : undefined}
        aria-autocomplete={suggestions.length ? 'list' : undefined}
        autoComplete="off"
        className={cn(mono && 'font-mono text-[12.5px]')}
        onFocus={() => setOpen(true)}
        onBlur={() => window.setTimeout(() => setOpen(false), 120)}
        onChange={(event) => {
          onChange(event.target.value)
          setOpen(true)
          setActive(0)
        }}
        onKeyDown={(event) => {
          if (!showList) return
          if (event.key === 'ArrowDown') {
            event.preventDefault()
            setActive((index) => Math.min(matches.length - 1, index + 1))
          } else if (event.key === 'ArrowUp') {
            event.preventDefault()
            setActive((index) => Math.max(0, index - 1))
          } else if (event.key === 'Enter') {
            event.preventDefault()
            pick(matches[Math.min(active, matches.length - 1)])
          } else if (event.key === 'Escape') {
            setOpen(false)
          }
        }}
      />
      {showList ? (
        <div
          id={listId}
          role="listbox"
          className="absolute inset-x-0 top-full z-50 mt-1 max-h-56 overflow-y-auto rounded-lg border border-border bg-popover p-1 shadow-lg"
        >
          {matches.map((item, index) => (
            <button
              key={item}
              type="button"
              role="option"
              aria-selected={index === active}
              onMouseDown={(event) => event.preventDefault()}
              onMouseEnter={() => setActive(index)}
              onClick={() => pick(item)}
              className={cn(
                'flex w-full items-center rounded-md px-2 py-1.5 text-left font-mono text-[12.5px] text-foreground',
                index === active && 'bg-accent',
              )}
            >
              <span className="truncate">{item}</span>
            </button>
          ))}
        </div>
      ) : null}
    </div>
  )
}

/** Number input that keeps partial entries such as "-" or "0." until they parse. */
export function NumberInput({
  id,
  value,
  onChange,
  min,
  max,
  placeholder,
  disabled,
  className,
  ariaLabel,
}: {
  id?: string
  value: unknown
  onChange: (value: number | undefined) => void
  min?: number
  max?: number
  placeholder?: string
  disabled?: boolean
  className?: string
  ariaLabel?: string
}) {
  const external = typeof value === 'number' && Number.isFinite(value) ? String(value) : ''
  const [draft, setDraft] = React.useState(external)
  React.useEffect(() => {
    setDraft((current) =>
      Number(current) === Number(external) && current !== '' ? current : external,
    )
  }, [external])
  const parsed = draft.trim() === '' ? undefined : Number(draft)
  const outOfRange =
    parsed !== undefined &&
    Number.isFinite(parsed) &&
    ((min !== undefined && parsed < min) || (max !== undefined && parsed > max))
  return (
    <Input
      id={id}
      inputMode="decimal"
      value={draft}
      disabled={disabled}
      placeholder={placeholder}
      aria-label={ariaLabel}
      aria-invalid={(parsed !== undefined && !Number.isFinite(parsed)) || outOfRange || undefined}
      className={cn('font-mono text-[12.5px] tabular', className)}
      onChange={(event) => {
        const next = event.target.value
        setDraft(next)
        if (next.trim() === '') onChange(undefined)
        else if (Number.isFinite(Number(next))) onChange(Number(next))
      }}
      onBlur={() => setDraft(external)}
    />
  )
}

/** A select over schema options; a value the schema does not list stays selectable. */
export function OptionSelect({
  id,
  value,
  options,
  disabledOptions = {},
  onChange,
  disabled,
  required,
  ariaLabel,
  className,
  renderOption,
}: {
  id?: string
  value: unknown
  options: readonly string[]
  disabledOptions?: Record<string, string>
  onChange: (value: string | undefined) => void
  disabled?: boolean
  required?: boolean
  ariaLabel?: string
  className?: string
  renderOption?: (option: string) => React.ReactNode
}) {
  const current = typeof value === 'string' ? value : ''
  const listed = options.filter((option) => option !== '')
  const offersDefault = options.includes('') || !required
  const unlisted = current && !listed.includes(current) ? current : null
  const compact =
    !unlisted &&
    listed.length > 1 &&
    listed.length + (options.includes('') ? 1 : 0) <= SEGMENTED_MAX_OPTIONS &&
    listed.every((option) => option.length <= SEGMENTED_MAX_LABEL) &&
    !renderOption &&
    Object.keys(disabledOptions).length === 0
  if (compact) {
    return (
      <Segmented
        aria-label={ariaLabel}
        value={current || UNSET}
        onValueChange={(next) => onChange(next === UNSET ? undefined : next)}
        className={cn('w-fit max-w-full', disabled && 'pointer-events-none opacity-60', className)}
        options={[
          ...(options.includes('') ? [{ value: UNSET, label: 'Default' }] : []),
          ...listed.map((option) => ({ value: option, label: option })),
        ]}
      />
    )
  }
  return (
    <Select
      value={current || UNSET}
      disabled={disabled}
      onValueChange={(next) => onChange(next === UNSET ? undefined : next)}
    >
      <SelectTrigger
        id={id}
        aria-label={ariaLabel}
        className={cn('w-full font-mono text-[12.5px]', className)}
      >
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {offersDefault || !current ? (
          <SelectItem value={UNSET}>
            <span className="font-sans text-muted-foreground">
              {required ? 'Choose…' : 'Default'}
            </span>
          </SelectItem>
        ) : null}
        {listed.map((option) => (
          <SelectItem key={option} value={option} disabled={Boolean(disabledOptions[option])}>
            {renderOption ? (
              renderOption(option)
            ) : (
              <span className="font-mono text-[12.5px]">
                {option}
                {disabledOptions[option] ? (
                  <span className="ml-2 font-sans text-xs text-subtle-foreground">
                    {disabledOptions[option]}
                  </span>
                ) : null}
              </span>
            )}
          </SelectItem>
        ))}
        {unlisted ? (
          <>
            <SelectSeparator />
            <SelectItem value={unlisted}>
              <span className="font-mono text-[12.5px]">{unlisted}</span>
              <span className="ml-2 text-xs text-subtle-foreground">as written</span>
            </SelectItem>
          </>
        ) : null}
      </SelectContent>
    </Select>
  )
}

export function BooleanSwitch({
  id,
  value,
  onChange,
  disabled,
  ariaLabel,
}: {
  id?: string
  value: unknown
  onChange: (value: boolean) => void
  disabled?: boolean
  ariaLabel?: string
}) {
  return (
    <span className="flex items-center gap-2">
      <span className="text-xs text-subtle-foreground">
        {value === undefined ? 'Default' : value ? 'On' : 'Off'}
      </span>
      <Switch
        id={id}
        aria-label={ariaLabel}
        checked={value === true}
        disabled={disabled}
        onCheckedChange={(checked) => onChange(checked)}
      />
    </span>
  )
}

/** Configured values a list can take, offered under a chip input; one click adds one. */
export function SuggestionChips({
  value,
  suggestions,
  onAdd,
  caseSensitive = true,
  limit = 12,
}: {
  value: readonly string[]
  suggestions: readonly string[]
  onAdd: (next: string[]) => void
  caseSensitive?: boolean
  limit?: number
}) {
  const remaining = suggestions.filter(
    (item) => item && mergeChips(value, [item], { caseSensitive }).next.length > value.length,
  )
  if (remaining.length === 0) return null
  return (
    <div className="flex flex-wrap items-center gap-1">
      <span className="mr-0.5 text-[11px] text-subtle-foreground">Add</span>
      {remaining.slice(0, limit).map((item) => (
        <button
          key={item}
          type="button"
          onClick={() => onAdd(mergeChips(value, [item], { caseSensitive }).next)}
          className="inline-flex h-6 items-center gap-1 rounded-md px-1.5 font-mono text-[11.5px] text-muted-foreground ring-1 ring-inset ring-border transition-colors hover:bg-accent hover:text-foreground focus-visible:ring-[3px] focus-visible:ring-ring/40 focus-visible:outline-none"
        >
          <Plus className="size-3" />
          {item}
        </button>
      ))}
      {remaining.length > limit ? (
        <span className="text-[11px] text-subtle-foreground">+{remaining.length - limit} more</span>
      ) : null}
    </div>
  )
}

/**
 * A list of patterns, such as regular expressions, that may contain commas:
 * only Enter and new lines separate entries.
 */
export function PatternListInput({
  id,
  value,
  onChange,
  placeholder,
  disabled,
  ariaLabel,
}: {
  id?: string
  value: readonly string[]
  onChange: (value: string[]) => void
  placeholder?: string
  disabled?: boolean
  ariaLabel?: string
}) {
  const [draft, setDraft] = React.useState('')
  const commit = (raw: string) => {
    const { next } = mergeChips(value, raw.split('\n'), { caseSensitive: true })
    if (next.length !== value.length) onChange(next)
    setDraft('')
  }
  return (
    <label
      htmlFor={id}
      className={cn(
        'flex min-h-9 w-full flex-wrap items-center gap-1.5 rounded-md border border-input bg-surface px-2 py-1.5 shadow-xs transition-[border-color,box-shadow] focus-within:border-ring focus-within:ring-[3px] focus-within:ring-ring/25',
        disabled && 'cursor-not-allowed opacity-60',
      )}
    >
      {value.map((item, index) => (
        <span
          key={`${item}-${index}`}
          className="inline-flex max-w-full items-center gap-1 rounded-md border border-border bg-surface-subtle py-0.5 pr-0.5 pl-2 font-mono text-[12px] text-foreground"
        >
          <span className="truncate">{item}</span>
          {disabled ? null : (
            <button
              type="button"
              aria-label={`Remove ${item}`}
              onClick={(event) => {
                event.preventDefault()
                onChange(value.filter((_, position) => position !== index))
              }}
              className="grid size-4 place-items-center rounded-sm text-subtle-foreground outline-none hover:bg-accent hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/40"
            >
              <X className="size-3" />
            </button>
          )}
        </span>
      ))}
      <input
        id={id}
        aria-label={ariaLabel}
        value={draft}
        disabled={disabled}
        placeholder={value.length ? '' : placeholder || 'Type a pattern and press Enter'}
        onChange={(event) => setDraft(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === 'Enter' && draft.trim()) {
            event.preventDefault()
            commit(draft)
          } else if (event.key === 'Backspace' && !draft && value.length) {
            event.preventDefault()
            onChange(value.slice(0, -1))
          }
        }}
        onPaste={(event) => {
          const text = event.clipboardData.getData('text')
          if (!text.includes('\n')) return
          event.preventDefault()
          commit(text)
        }}
        onBlur={() => draft.trim() && commit(draft)}
        className="h-6 min-w-[12ch] flex-1 bg-transparent font-mono text-[12.5px] outline-none placeholder:font-sans placeholder:text-subtle-foreground disabled:cursor-not-allowed"
      />
    </label>
  )
}

export function KeyValueInput({
  value,
  onChange,
  disabled,
  keyLabel = 'Key',
  valueLabel = 'Value',
}: {
  value: Record<string, unknown>
  onChange: (value: Record<string, unknown>) => void
  disabled?: boolean
  keyLabel?: string
  valueLabel?: string
}) {
  const entries = Object.entries(value)
  const [pending, setPending] = React.useState<{ key: string; value: string }>({
    key: '',
    value: '',
  })
  const rename = (index: number, nextKey: string) =>
    onChange(
      Object.fromEntries(
        entries.map(([key, item], position) =>
          position === index ? [nextKey, item] : [key, item],
        ),
      ),
    )
  const update = (index: number, nextValue: string) =>
    onChange(
      Object.fromEntries(
        entries.map(([key, item], position) =>
          position === index ? [key, nextValue] : [key, item],
        ),
      ),
    )
  const add = () => {
    const key = pending.key.trim()
    if (!key || key in value) return
    onChange({ ...value, [key]: pending.value })
    setPending({ key: '', value: '' })
  }
  return (
    <div className="flex flex-col gap-1.5">
      {entries.map(([key, item], index) => (
        <div
          key={index}
          className="grid grid-cols-[minmax(0,2fr)_minmax(0,3fr)_auto] items-center gap-1.5"
        >
          <Input
            aria-label={`${keyLabel} ${index + 1}`}
            value={key}
            disabled={disabled}
            onChange={(event) => rename(index, event.target.value)}
            className="h-8 font-mono text-[12.5px]"
          />
          <Input
            aria-label={`${valueLabel} for ${key}`}
            value={typeof item === 'string' ? item : JSON.stringify(item)}
            disabled={disabled || typeof item !== 'string'}
            onChange={(event) => update(index, event.target.value)}
            className="h-8 font-mono text-[12.5px]"
          />
          <Button
            type="button"
            variant="ghost"
            size="icon-xs"
            aria-label={`Remove ${key}`}
            disabled={disabled}
            onClick={() =>
              onChange(Object.fromEntries(entries.filter((_, position) => position !== index)))
            }
          >
            <Trash2 />
          </Button>
        </div>
      ))}
      {disabled ? null : (
        <div className="grid grid-cols-[minmax(0,2fr)_minmax(0,3fr)_auto] items-center gap-1.5">
          <Input
            aria-label={`New ${keyLabel.toLowerCase()}`}
            placeholder={keyLabel}
            value={pending.key}
            onChange={(event) => setPending((current) => ({ ...current, key: event.target.value }))}
            onKeyDown={(event) => event.key === 'Enter' && (event.preventDefault(), add())}
            className="h-8 font-mono text-[12.5px]"
          />
          <Input
            aria-label={`New ${valueLabel.toLowerCase()}`}
            placeholder={valueLabel}
            value={pending.value}
            onChange={(event) =>
              setPending((current) => ({ ...current, value: event.target.value }))
            }
            onKeyDown={(event) => event.key === 'Enter' && (event.preventDefault(), add())}
            className="h-8 font-mono text-[12.5px]"
          />
          <Button
            type="button"
            variant="outline"
            size="icon-xs"
            aria-label="Add entry"
            onClick={add}
            disabled={!pending.key.trim()}
          >
            <Plus />
          </Button>
        </div>
      )}
    </div>
  )
}

export function JsonInput({
  id,
  value,
  onChange,
  disabled,
  placeholder,
}: {
  id?: string
  value: unknown
  onChange: (value: unknown) => void
  disabled?: boolean
  placeholder?: string
}) {
  const serialized = value === undefined ? '' : JSON.stringify(value, null, 2)
  const [draft, setDraft] = React.useState(serialized)
  const [error, setError] = React.useState('')
  React.useEffect(() => {
    setDraft(serialized)
    setError('')
  }, [serialized])
  return (
    <div className="flex flex-col gap-1">
      <Textarea
        id={id}
        value={draft}
        disabled={disabled}
        placeholder={placeholder || '{}'}
        aria-invalid={Boolean(error) || undefined}
        className="min-h-24 font-mono text-[12px]"
        onChange={(event) => setDraft(event.target.value)}
        onBlur={() => {
          if (!draft.trim()) {
            setError('')
            onChange(undefined)
            return
          }
          try {
            onChange(JSON.parse(draft))
            setError('')
          } catch (parseError) {
            setError(parseError instanceof Error ? parseError.message : 'Invalid JSON')
          }
        }}
      />
      {error ? <p className="text-xs text-destructive-foreground">{error}</p> : null}
    </div>
  )
}

/** gt/gte and lt/lte as a lower and an upper bound. */
export function BoundsInput({
  value,
  onChange,
  disabled,
  idPrefix,
}: {
  value: Record<string, unknown>
  onChange: (value: Record<string, unknown> | undefined) => void
  disabled?: boolean
  idPrefix: string
}) {
  const bothLower = value.gt !== undefined && value.gte !== undefined
  const bothUpper = value.lt !== undefined && value.lte !== undefined
  const setBound = (
    pair: readonly [BoundKey, BoundKey],
    key: BoundKey,
    number: number | undefined,
  ) => {
    const next = { ...value }
    for (const candidate of pair) delete next[candidate]
    if (number !== undefined) next[key] = number
    onChange(Object.keys(next).length ? next : undefined)
  }
  const side = (pair: readonly [BoundKey, BoundKey], label: string, conflicting: boolean) => {
    const key = value[pair[1]] !== undefined ? pair[1] : pair[0]
    if (conflicting) {
      return (
        <div className="flex flex-col gap-1">
          <span className="text-xs text-subtle-foreground">{label}</span>
          {pair.map((bound) => (
            <div key={bound} className="flex items-center gap-1.5">
              <span className="w-8 text-center font-mono text-xs text-muted-foreground">
                {BOUND_LABEL[bound]}
              </span>
              <NumberInput
                ariaLabel={`${label} ${bound}`}
                value={value[bound]}
                disabled={disabled}
                onChange={(number) => {
                  const next = { ...value }
                  if (number === undefined) delete next[bound]
                  else next[bound] = number
                  onChange(Object.keys(next).length ? next : undefined)
                }}
                className="h-8"
              />
            </div>
          ))}
        </div>
      )
    }
    return (
      <div className="flex flex-col gap-1">
        <span className="text-xs text-subtle-foreground">{label}</span>
        <div className="flex items-center gap-1.5">
          <Segmented
            size="sm"
            aria-label={`${label} comparison`}
            value={key}
            onValueChange={(nextKey) =>
              setBound(
                pair,
                nextKey,
                typeof value[key] === 'number' ? (value[key] as number) : undefined,
              )
            }
            className={cn(disabled && 'pointer-events-none opacity-60')}
            options={pair.map((bound) => ({ value: bound, label: BOUND_LABEL[bound] }))}
          />
          <NumberInput
            id={`${idPrefix}-${pair[0]}`}
            ariaLabel={`${label} value`}
            value={value[key]}
            disabled={disabled}
            placeholder="none"
            onChange={(number) => setBound(pair, key, number)}
            className="h-8"
          />
        </div>
      </div>
    )
  }
  return (
    <div className="grid grid-cols-2 gap-3">
      {side(['gt', 'gte'], 'Lower bound', bothLower)}
      {side(['lt', 'lte'], 'Upper bound', bothUpper)}
    </div>
  )
}
