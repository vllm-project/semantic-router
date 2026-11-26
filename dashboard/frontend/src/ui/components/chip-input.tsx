import * as React from 'react'
import { X } from 'lucide-react'
import { cn } from '../lib/cn'
import { mergeChips } from './chip-input-support'

/**
 * Editable list of short strings shown as chips (keywords, model names, tags).
 * Enter, Tab or a comma adds the typed value; pasting splits on commas and new
 * lines; Backspace in an empty field removes the last chip. With
 * `splitOnComma={false}` (for values that contain commas, such as regular
 * expressions) only Enter, Tab and new lines separate values. Duplicates are
 * ignored (case-insensitively unless `caseSensitive`), and `validate` may
 * refuse a value with a message.
 */
export function ChipInput({
  value,
  onChange,
  placeholder = 'Type and press Enter',
  caseSensitive = false,
  splitOnComma = true,
  validate,
  disabled,
  invalid,
  className,
  id,
  'aria-label': ariaLabel,
  'aria-describedby': ariaDescribedBy,
}: {
  value: readonly string[]
  onChange: (next: string[]) => void
  placeholder?: string
  caseSensitive?: boolean
  splitOnComma?: boolean
  validate?: (candidate: string) => string | null
  disabled?: boolean
  invalid?: boolean
  className?: string
  id?: string
  'aria-label'?: string
  'aria-describedby'?: string
}) {
  const [draft, setDraft] = React.useState('')
  const [error, setError] = React.useState<string | null>(null)
  const inputRef = React.useRef<HTMLInputElement>(null)
  const errorId = React.useId()
  const generatedId = React.useId()
  const inputId = id ?? generatedId

  const add = (raw: string[]) => {
    const result = mergeChips(value, raw, { caseSensitive, validate })
    setError(result.error)
    if (result.error) return false
    if (result.next.length !== value.length) onChange(result.next)
    return true
  }

  const commitDraft = () => {
    if (!draft.trim()) return false
    if (add([draft])) setDraft('')
    return true
  }

  const remove = (index: number) => {
    onChange(value.filter((_, position) => position !== index))
    inputRef.current?.focus()
  }

  return (
    <div className={cn('flex flex-col gap-1.5', className)}>
      <label
        htmlFor={inputId}
        className={cn(
          'flex min-h-9 w-full flex-wrap items-center gap-1.5 rounded-md border border-input bg-surface px-2 py-1.5 text-[13px] shadow-xs transition-[border-color,box-shadow] focus-within:border-ring focus-within:ring-[3px] focus-within:ring-ring/25',
          (invalid || error) && 'border-destructive focus-within:ring-destructive/20',
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
                  remove(index)
                }}
                className="grid size-4 place-items-center rounded-sm text-subtle-foreground outline-none hover:bg-accent hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/40"
              >
                <X className="size-3" />
              </button>
            )}
          </span>
        ))}
        <input
          ref={inputRef}
          id={inputId}
          value={draft}
          disabled={disabled}
          aria-label={ariaLabel}
          aria-invalid={Boolean(invalid || error) || undefined}
          aria-describedby={
            [error ? errorId : null, ariaDescribedBy].filter(Boolean).join(' ') || undefined
          }
          placeholder={value.length ? '' : placeholder}
          onChange={(event) => {
            const text = event.target.value
            if (splitOnComma && text.includes(',')) {
              const parts = text.split(',')
              const tail = parts.pop() ?? ''
              if (add(parts)) setDraft(tail)
              else setDraft(text)
              return
            }
            setDraft(text)
            if (error) setError(null)
          }}
          onKeyDown={(event) => {
            if (event.key === 'Enter' || (event.key === 'Tab' && draft.trim())) {
              if (commitDraft()) event.preventDefault()
            } else if (event.key === 'Backspace' && !draft && value.length) {
              event.preventDefault()
              onChange(value.slice(0, -1))
            }
          }}
          onPaste={(event) => {
            const text = event.clipboardData.getData('text')
            const separator = splitOnComma ? /[,\n]/ : /\n/
            if (!separator.test(text)) return
            event.preventDefault()
            add(text.split(separator))
          }}
          onBlur={() => commitDraft()}
          className="h-6 min-w-[8ch] flex-1 bg-transparent text-[13px] outline-none placeholder:text-subtle-foreground disabled:cursor-not-allowed"
        />
      </label>
      {error ? (
        <p id={errorId} className="text-xs text-destructive">
          {error}
        </p>
      ) : null}
    </div>
  )
}
