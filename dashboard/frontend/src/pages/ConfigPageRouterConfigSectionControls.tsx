import * as React from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { Check, Eye, EyeOff, Plus, RotateCcw, Undo2, X } from 'lucide-react'
import {
  Badge,
  Button,
  Input,
  InputGroup,
  Label,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  Switch,
  Textarea,
  Tooltip,
  cn,
} from '../ui'

// Design-system controls for one router setting. Each is controlled; an
// undefined value means the field inherits, and `placeholder` shows what it
// inherits.

export function FieldShell({
  id,
  label,
  required,
  inherited,
  edited,
  restart,
  hint,
  error,
  actions,
  className,
  children,
}: {
  id: string
  label: React.ReactNode
  required?: boolean
  inherited?: boolean
  edited?: boolean
  restart?: string
  hint?: React.ReactNode
  error?: string
  actions?: React.ReactNode
  className?: string
  children: React.ReactNode
}) {
  return (
    <div className={cn('group/field flex min-w-0 flex-col gap-1.5', className)} data-setting={id}>
      <div className="flex h-6 min-w-0 items-center gap-1.5">
        <AnimatePresence initial={false}>
          {edited ? (
            <motion.span
              key="edited"
              initial={{ scale: 0, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0, opacity: 0 }}
              className="size-1.5 shrink-0 rounded-full bg-primary"
              role="img"
              aria-label="Edited"
            />
          ) : null}
        </AnimatePresence>
        <Label htmlFor={id} className="min-w-0 truncate">
          {label}
          {required ? <span className="text-subtle-foreground">*</span> : null}
        </Label>
        {inherited ? (
          <Badge size="sm" tone="outline" className="font-normal">
            inherited
          </Badge>
        ) : null}
        {restart ? (
          <Tooltip content={restart}>
            <span className="inline-flex">
              <Badge size="sm" tone="outline" className="font-normal">
                restart
              </Badge>
            </span>
          </Tooltip>
        ) : null}
        {actions ? <span className="ml-auto flex shrink-0 items-center">{actions}</span> : null}
      </div>
      {children}
      {error ? (
        <p className="text-xs leading-relaxed text-destructive-foreground">{error}</p>
      ) : hint ? (
        <p className="text-xs leading-relaxed text-muted-foreground">{hint}</p>
      ) : null}
    </div>
  )
}

/** Undo for an edited field, or a hover reset for one config.yaml sets. */
export function FieldActions({
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
      <Tooltip content="Undo this change">
        <Button
          variant="ghost"
          size="icon-xs"
          className="size-6"
          onClick={onUndo}
          aria-label={`Undo ${label}`}
        >
          <Undo2 />
        </Button>
      </Tooltip>
    )
  }
  if (!explicit) return null
  return (
    <Tooltip content="Use the router default">
      <Button
        variant="ghost"
        size="icon-xs"
        className="size-6 opacity-0 transition-opacity focus-visible:opacity-100 group-hover/field:opacity-100"
        onClick={onReset}
        aria-label={`Reset ${label} to its default`}
      >
        <RotateCcw />
      </Button>
    </Tooltip>
  )
}

export function TextControl({
  id,
  value,
  placeholder,
  secret,
  mono = true,
  unit,
  disabled,
  invalid,
  label,
  onChange,
}: {
  id: string
  value: string | undefined
  placeholder?: string
  secret?: boolean
  mono?: boolean
  unit?: string
  disabled?: boolean
  invalid?: boolean
  label: string
  onChange: (next: string | undefined) => void
}) {
  const [revealed, setRevealed] = React.useState(false)
  const trailing = secret ? (
    <Button
      variant="ghost"
      size="icon-xs"
      className="size-6"
      onClick={() => setRevealed((current) => !current)}
      aria-label={revealed ? `Hide ${label}` : `Show ${label}`}
      disabled={disabled}
    >
      {revealed ? <EyeOff /> : <Eye />}
    </Button>
  ) : unit ? (
    <span className="pr-1 text-xs text-subtle-foreground">{unit}</span>
  ) : undefined
  return (
    <InputGroup
      id={id}
      value={value ?? ''}
      type={secret && !revealed ? 'password' : 'text'}
      autoComplete="off"
      spellCheck={false}
      disabled={disabled}
      placeholder={placeholder}
      aria-invalid={invalid || undefined}
      onChange={(event) => onChange(event.target.value === '' ? undefined : event.target.value)}
      className={cn(mono && 'font-mono text-[12.5px]', !trailing && 'pr-3')}
      trailing={trailing}
    />
  )
}

function formatNumber(value: number, percent?: boolean): string {
  const shown = percent ? Math.round(value * 10000) / 100 : value
  return String(shown)
}

export function NumberControl({
  id,
  value,
  placeholder,
  unit,
  percent,
  disabled,
  invalid,
  onChange,
}: {
  id: string
  value: unknown
  placeholder?: string
  unit?: string
  percent?: boolean
  disabled?: boolean
  invalid?: boolean
  onChange: (next: number | string | undefined) => void
}) {
  const external =
    typeof value === 'number'
      ? formatNumber(value, percent)
      : typeof value === 'string'
        ? value
        : ''
  const [text, setText] = React.useState(external)
  React.useEffect(() => {
    setText((current) => {
      const parsed = Number(current)
      const same =
        current.trim() !== '' &&
        typeof value === 'number' &&
        Number.isFinite(parsed) &&
        formatNumber(value, percent) === String(parsed)
      return same ? current : external
    })
  }, [external, percent, value])
  const shownUnit = percent ? '%' : unit
  return (
    <InputGroup
      id={id}
      value={text}
      inputMode="decimal"
      autoComplete="off"
      disabled={disabled}
      placeholder={placeholder}
      aria-invalid={invalid || undefined}
      onChange={(event) => {
        const raw = event.target.value
        setText(raw)
        if (raw.trim() === '') {
          onChange(undefined)
          return
        }
        const parsed = Number(raw)
        if (!Number.isFinite(parsed)) {
          onChange(raw)
          return
        }
        onChange(percent ? Math.round(parsed * 100) / 10000 : parsed)
      }}
      className={cn(
        'font-mono text-[12.5px] tabular',
        !shownUnit && 'pr-3',
        shownUnit && shownUnit.length > 3 && 'pr-16',
      )}
      trailing={
        shownUnit ? (
          <span className="pr-1 text-xs text-subtle-foreground">{shownUnit}</span>
        ) : undefined
      }
    />
  )
}

export function BooleanControl({
  id,
  checked,
  inherited,
  disabled,
  label,
  onChange,
}: {
  id: string
  checked: boolean
  inherited?: boolean
  disabled?: boolean
  label: string
  onChange: (next: boolean) => void
}) {
  return (
    <div className="flex h-9 items-center gap-2.5">
      <Switch
        id={id}
        checked={checked}
        onCheckedChange={onChange}
        disabled={disabled}
        aria-label={label}
      />
      <span
        className={cn(
          'text-[13px]',
          inherited ? 'text-subtle-foreground' : 'text-muted-foreground',
        )}
      >
        {checked ? 'On' : 'Off'}
      </span>
    </div>
  )
}

function choiceLabel(option: string): string {
  const words = option.replace(/[_-]+/g, ' ').trim()
  return words ? words.charAt(0).toUpperCase() + words.slice(1) : option
}

/** Segmented choice for two or three short options; an inherited value shows dashed. */
export function ChoiceControl({
  id,
  value,
  inheritedValue,
  options,
  optionLabels,
  disabled,
  label,
  onChange,
}: {
  id: string
  value: string | undefined
  inheritedValue?: string
  options: readonly string[]
  optionLabels?: Record<string, string>
  disabled?: boolean
  label: string
  onChange: (next: string) => void
}) {
  const current = value ?? inheritedValue
  return (
    <div className="flex h-9 items-center">
      <div
        id={id}
        role="radiogroup"
        aria-label={label}
        className="inline-flex items-center gap-0.5 rounded-lg border border-border bg-surface-subtle p-0.5"
      >
        {options.map((option) => {
          const active = option === current
          return (
            <button
              key={option}
              type="button"
              role="radio"
              aria-checked={active}
              disabled={disabled}
              onClick={() => onChange(option)}
              className={cn(
                'inline-flex h-7 items-center rounded-md px-2.5 text-[13px] font-medium text-muted-foreground outline-none transition-[color,background-color,box-shadow] hover:text-foreground focus-visible:ring-[3px] focus-visible:ring-ring/40 disabled:pointer-events-none disabled:opacity-50',
                active &&
                  value !== undefined &&
                  'bg-surface text-foreground shadow-sm dark:bg-accent',
                active &&
                  value === undefined &&
                  'bg-surface/60 text-foreground/70 ring-1 ring-dashed ring-border-strong dark:bg-accent/50',
              )}
            >
              {optionLabels?.[option] ?? choiceLabel(option)}
            </button>
          )
        })}
      </div>
    </div>
  )
}

/** A list limited to known values: each value toggles on or off, kept in contract order. */
export function OptionSetControl({
  id,
  values,
  inheritedValues,
  options,
  disabled,
  label,
  onChange,
}: {
  id: string
  values: readonly unknown[] | undefined
  inheritedValues?: readonly unknown[]
  options: readonly string[]
  disabled?: boolean
  label: string
  onChange: (next: string[]) => void
}) {
  const inherited = values === undefined
  const selected = new Set((values ?? inheritedValues ?? []).map(String))
  const toggle = (option: string) => {
    const next = new Set(selected)
    if (next.has(option)) next.delete(option)
    else next.add(option)
    onChange(options.filter((candidate) => next.has(candidate)))
  }
  return (
    <div
      id={id}
      role="group"
      aria-label={label}
      className="flex min-h-9 flex-wrap items-center gap-1.5"
    >
      {options.map((option) => {
        const on = selected.has(option)
        return (
          <button
            key={option}
            type="button"
            disabled={disabled}
            onClick={() => toggle(option)}
            className={cn(
              'inline-flex h-7 items-center gap-1 rounded-md border px-2 font-mono text-[12px] outline-none transition-[color,background-color,border-color,box-shadow] focus-visible:ring-[3px] focus-visible:ring-ring/40 disabled:pointer-events-none disabled:opacity-50',
              !on &&
                'border-border bg-surface text-muted-foreground hover:bg-surface-subtle hover:text-foreground',
              on &&
                !inherited &&
                'border-border-strong bg-surface-subtle text-foreground shadow-xs',
              on && inherited && 'border-dashed border-border-strong text-subtle-foreground',
            )}
            aria-pressed={on}
          >
            {on ? <Check className="size-3" aria-hidden /> : null}
            {option}
          </button>
        )
      })}
    </div>
  )
}

export function SelectControl({
  id,
  value,
  placeholder,
  options,
  optionLabels,
  disabled,
  invalid,
  mono = true,
  label,
  onChange,
}: {
  id: string
  value: string | undefined
  placeholder?: string
  options: readonly string[]
  optionLabels?: Record<string, string>
  disabled?: boolean
  invalid?: boolean
  mono?: boolean
  label: string
  onChange: (next: string) => void
}) {
  const choices = value && !options.includes(value) ? [value, ...options] : options
  return (
    <Select value={value ?? ''} onValueChange={onChange} disabled={disabled}>
      <SelectTrigger
        id={id}
        aria-label={label}
        aria-invalid={invalid || undefined}
        className={cn(
          'w-full',
          mono && !(value && optionLabels?.[value]) && 'font-mono text-[12.5px]',
        )}
      >
        <SelectValue placeholder={placeholder || 'Choose…'} />
      </SelectTrigger>
      <SelectContent>
        {choices.map((option) => (
          <SelectItem
            key={option}
            value={option}
            className={cn(mono && !optionLabels?.[option] && 'font-mono text-[12.5px]')}
          >
            {optionLabels?.[option] ?? option}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

/** Tags input for a list of short values; Enter or comma adds, Backspace removes the last. */
export function ChipsControl({
  id,
  values,
  inheritedValues,
  numeric,
  disabled,
  label,
  placeholder,
  onChange,
}: {
  id: string
  values: readonly (string | number)[] | undefined
  inheritedValues?: readonly unknown[]
  numeric?: boolean
  disabled?: boolean
  label: string
  placeholder?: string
  onChange: (next: Array<string | number>) => void
}) {
  const [draft, setDraft] = React.useState('')
  const [problem, setProblem] = React.useState<string | null>(null)
  const inherited = values === undefined
  const shown = (values ?? inheritedValues ?? []) as Array<string | number>

  const commit = (raw: string) => {
    const tokens = raw
      .split(/[,\n]/)
      .map((token) => token.trim())
      .filter(Boolean)
    if (tokens.length === 0) return true
    const parsed: Array<string | number> = []
    for (const token of tokens) {
      if (numeric) {
        const number = Number(token)
        if (!Number.isFinite(number)) {
          setProblem(`${token} is not a number.`)
          return false
        }
        parsed.push(number)
      } else {
        parsed.push(token)
      }
    }
    const base = values ?? (inheritedValues as Array<string | number> | undefined) ?? []
    const next = [...base]
    for (const item of parsed)
      if (!next.some((existing) => String(existing) === String(item))) next.push(item)
    onChange(next)
    setProblem(null)
    return true
  }

  return (
    <div className="flex flex-col gap-1">
      <div
        className={cn(
          'flex min-h-9 w-full flex-wrap items-center gap-1 rounded-md border border-input bg-surface px-1.5 py-1 shadow-xs transition-[border-color,box-shadow] focus-within:border-ring focus-within:ring-[3px] focus-within:ring-ring/40',
          disabled && 'cursor-not-allowed opacity-50',
          problem && 'border-destructive',
        )}
      >
        {shown.map((item, index) => (
          <span
            key={`${String(item)}-${index}`}
            className={cn(
              'inline-flex h-6 max-w-full items-center gap-1 rounded-md border pl-2 pr-1 font-mono text-[12px]',
              inherited
                ? 'border-dashed border-border-strong text-subtle-foreground'
                : 'border-border bg-surface-subtle text-foreground',
            )}
          >
            <span className="truncate">{String(item)}</span>
            {!disabled ? (
              <button
                type="button"
                aria-label={`Remove ${String(item)} from ${label}`}
                onClick={() => onChange(shown.filter((_, itemIndex) => itemIndex !== index))}
                className="grid size-4 place-items-center rounded text-subtle-foreground hover:bg-accent hover:text-foreground"
              >
                <X className="size-3" />
              </button>
            ) : null}
          </span>
        ))}
        <input
          id={id}
          value={draft}
          disabled={disabled}
          aria-label={label}
          inputMode={numeric ? 'decimal' : undefined}
          placeholder={
            shown.length === 0 ? (placeholder ?? (numeric ? 'Add a number' : 'Add a value')) : ''
          }
          onChange={(event) => {
            setDraft(event.target.value)
            if (problem) setProblem(null)
          }}
          onKeyDown={(event) => {
            if (event.key === 'Enter' || event.key === ',') {
              event.preventDefault()
              if (commit(draft)) setDraft('')
            } else if (event.key === 'Backspace' && draft === '' && shown.length > 0) {
              onChange(shown.slice(0, -1))
            }
          }}
          onBlur={() => {
            if (draft.trim() && commit(draft)) setDraft('')
          }}
          onPaste={(event) => {
            const text = event.clipboardData.getData('text')
            if (/[,\n]/.test(text)) {
              event.preventDefault()
              commit(text)
            }
          }}
          className="h-6 min-w-24 flex-1 bg-transparent px-1 font-mono text-[12.5px] outline-none placeholder:font-sans placeholder:text-subtle-foreground disabled:cursor-not-allowed"
        />
      </div>
      {problem ? <p className="text-xs text-destructive-foreground">{problem}</p> : null}
    </div>
  )
}

interface KeyValueRow {
  id: number
  key: string
  value: string
}

/** Rows of names and text values, such as extra headers. */
export function KeyValueControl({
  value,
  inheritedValue,
  disabled,
  label,
  keyPlaceholder = 'Name',
  valuePlaceholder = 'Value',
  onChange,
}: {
  value: Record<string, string> | undefined
  inheritedValue?: Record<string, string>
  disabled?: boolean
  label: string
  keyPlaceholder?: string
  valuePlaceholder?: string
  onChange: (next: Record<string, string>) => void
}) {
  const source = value ?? inheritedValue ?? {}
  const nextId = React.useRef(0)
  const toRows = (record: Record<string, string>): KeyValueRow[] =>
    Object.entries(record).map(([key, item]) => ({
      id: nextId.current++,
      key,
      value: String(item ?? ''),
    }))
  const [rows, setRows] = React.useState<KeyValueRow[]>(() => toRows(source))
  const signature = JSON.stringify(source)
  const lastEmitted = React.useRef(signature)
  React.useEffect(() => {
    if (signature !== lastEmitted.current) {
      lastEmitted.current = signature
      setRows(toRows(JSON.parse(signature) as Record<string, string>))
    }
  }, [signature])

  const emit = (next: KeyValueRow[]) => {
    setRows(next)
    const record = Object.fromEntries(
      next.filter((row) => row.key.trim()).map((row) => [row.key.trim(), row.value]),
    )
    lastEmitted.current = JSON.stringify(record)
    onChange(record)
  }
  const inherited = value === undefined

  return (
    <div className="flex flex-col gap-1.5">
      {rows.length > 0 ? (
        <div
          className={cn(
            'overflow-hidden rounded-lg border border-border',
            inherited && 'border-dashed',
          )}
        >
          {rows.map((row, index) => (
            <div
              key={row.id}
              className="grid grid-cols-[minmax(0,2fr)_minmax(0,3fr)_auto] items-center gap-1.5 border-b border-border px-1.5 py-1.5 last:border-0"
            >
              <Input
                value={row.key}
                disabled={disabled}
                placeholder={keyPlaceholder}
                aria-label={`${label} name ${index + 1}`}
                onChange={(event) =>
                  emit(
                    rows.map((item) =>
                      item.id === row.id ? { ...item, key: event.target.value } : item,
                    ),
                  )
                }
                className={cn('h-8 font-mono text-[12.5px]', inherited && 'text-subtle-foreground')}
              />
              <Input
                value={row.value}
                disabled={disabled}
                placeholder={valuePlaceholder}
                aria-label={`${label} value ${index + 1}`}
                onChange={(event) =>
                  emit(
                    rows.map((item) =>
                      item.id === row.id ? { ...item, value: event.target.value } : item,
                    ),
                  )
                }
                className={cn('h-8 font-mono text-[12.5px]', inherited && 'text-subtle-foreground')}
              />
              <Button
                variant="ghost"
                size="icon-xs"
                disabled={disabled}
                aria-label={`Remove ${row.key || 'entry'} from ${label}`}
                onClick={() => emit(rows.filter((item) => item.id !== row.id))}
              >
                <X />
              </Button>
            </div>
          ))}
        </div>
      ) : null}
      {!disabled ? (
        <Button
          variant="ghost"
          size="xs"
          className="w-fit"
          onClick={() =>
            setRows((current) => [...current, { id: nextId.current++, key: '', value: '' }])
          }
        >
          <Plus />
          Add entry
        </Button>
      ) : rows.length === 0 ? (
        <p className="text-xs text-subtle-foreground">No entries.</p>
      ) : null}
    </div>
  )
}

/** JSON editor for a value the contract leaves open. */
export function CodeControl({
  id,
  value,
  inheritedValue,
  disabled,
  label,
  onChange,
  onInvalid,
}: {
  id: string
  value: unknown
  inheritedValue?: unknown
  disabled?: boolean
  label: string
  onChange: (next: unknown) => void
  onInvalid?: (message: string | null) => void
}) {
  const external = value === undefined ? '' : JSON.stringify(value, null, 2)
  const [text, setText] = React.useState(external)
  const [problem, setProblem] = React.useState<string | null>(null)
  const lastEmitted = React.useRef(external)
  React.useEffect(() => {
    if (external !== lastEmitted.current) {
      lastEmitted.current = external
      setText(external)
      setProblem(null)
      onInvalid?.(null)
    }
  }, [external, onInvalid])
  return (
    <div className="flex flex-col gap-1">
      <Textarea
        id={id}
        value={text}
        disabled={disabled}
        aria-label={label}
        aria-invalid={Boolean(problem) || undefined}
        spellCheck={false}
        rows={Math.min(
          14,
          Math.max(4, (text || JSON.stringify(inheritedValue ?? {}, null, 2)).split('\n').length),
        )}
        placeholder={inheritedValue === undefined ? '{}' : JSON.stringify(inheritedValue, null, 2)}
        onChange={(event) => {
          const raw = event.target.value
          setText(raw)
          if (raw.trim() === '') {
            setProblem(null)
            onInvalid?.(null)
            lastEmitted.current = ''
            onChange(undefined)
            return
          }
          try {
            const parsed = JSON.parse(raw) as unknown
            setProblem(null)
            onInvalid?.(null)
            lastEmitted.current = JSON.stringify(parsed, null, 2)
            onChange(parsed)
          } catch (error) {
            const message = `Enter valid JSON. ${(error as Error).message}`
            setProblem(message)
            onInvalid?.(message)
          }
        }}
        className="font-mono text-[12px] leading-relaxed"
      />
      {problem ? <p className="text-xs text-destructive-foreground">{problem}</p> : null}
    </div>
  )
}
