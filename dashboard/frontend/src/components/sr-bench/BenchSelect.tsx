import { useEffect, useId, useRef, useState, type KeyboardEvent } from 'react'
import { Check, ChevronDown, Search } from 'lucide-react'
import { cn, inputBase } from '../../ui'

export interface BenchSelectOption {
  value: string
  label: string
  description?: string
}

interface Props {
  label: string
  value: string
  options: BenchSelectOption[]
  onChange: (value: string) => void
  placeholder?: string
  disabled?: boolean
  searchable?: boolean
  className?: string
  /** `toolbar` renders a compact filter pill whose label sits inside the control. */
  variant?: 'field' | 'toolbar'
}

/** Searchable listbox using the Dashboard's compact model-picker pattern. */
export default function BenchSelect({
  label,
  value,
  options,
  onChange,
  placeholder = 'Choose an option',
  disabled = false,
  searchable = false,
  className = '',
  variant = 'field',
}: Props) {
  const id = useId()
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [opensAbove, setOpensAbove] = useState(false)
  const root = useRef<HTMLDivElement>(null)
  const trigger = useRef<HTMLButtonElement>(null)
  const search = useRef<HTMLInputElement>(null)
  const optionRefs = useRef<Array<HTMLButtonElement | null>>([])
  const initialFocus = useRef<'first' | 'last' | 'selected'>('selected')
  const selected = options.find((option) => option.value === value)
  const toolbar = variant === 'toolbar'
  const filtered = options.filter((option) =>
    `${option.label} ${option.description ?? ''}`
      .toLowerCase()
      .includes(query.trim().toLowerCase()),
  )

  function close(restoreFocus = false) {
    setOpen(false)
    if (restoreFocus) trigger.current?.focus()
  }

  function focusOption(index: number) {
    if (filtered.length) optionRefs.current[(index + filtered.length) % filtered.length]?.focus()
  }

  function show(focus: typeof initialFocus.current = 'selected') {
    initialFocus.current = focus
    const bounds = trigger.current?.getBoundingClientRect()
    if (bounds) {
      const below = window.innerHeight - bounds.bottom
      const height = Math.min(options.length * 52 + (searchable ? 46 : 0) + 12, 340)
      setOpensAbove(below < height && bounds.top > below)
    }
    setQuery('')
    setOpen(true)
  }

  function optionKeyDown(event: KeyboardEvent<HTMLButtonElement>, index: number) {
    if (escape(event)) return
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault()
      focusOption(index + (event.key === 'ArrowDown' ? 1 : -1))
    } else if (event.key === 'Home' || event.key === 'End') {
      event.preventDefault()
      focusOption(event.key === 'Home' ? 0 : filtered.length - 1)
    }
  }

  function escape(event: KeyboardEvent<HTMLElement>) {
    if (!open || event.key !== 'Escape') return false
    event.preventDefault()
    event.stopPropagation()
    close(true)
    return true
  }

  useEffect(() => {
    if (!open) return
    if (searchable) search.current?.focus()
    else {
      const index =
        initialFocus.current === 'last'
          ? options.length - 1
          : initialFocus.current === 'first'
            ? 0
            : Math.max(
                0,
                options.findIndex((option) => option.value === value),
              )
      optionRefs.current[index]?.focus()
    }
    const outside = (event: PointerEvent) => {
      if (!root.current?.contains(event.target as Node)) setOpen(false)
    }
    // Sheets and dialogs dismiss on Escape in the document capture phase; an open
    // list closes first, so Escape must be intercepted before it reaches them.
    const escapeFirst = (event: globalThis.KeyboardEvent) => {
      if (event.key !== 'Escape' || !root.current?.contains(event.target as Node)) return
      event.preventDefault()
      event.stopPropagation()
      setOpen(false)
      trigger.current?.focus()
    }
    document.addEventListener('pointerdown', outside)
    window.addEventListener('keydown', escapeFirst, true)
    return () => {
      document.removeEventListener('pointerdown', outside)
      window.removeEventListener('keydown', escapeFirst, true)
    }
    // Focus only when opening; typing and selection must not reset keyboard focus.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, searchable])

  const filterActive = toolbar && !!options.length && value !== options[0].value
  const unavailable = disabled || options.length === 0

  return (
    <div
      ref={root}
      role="group"
      aria-labelledby={`${id}-label`}
      className={cn(
        'relative min-w-0',
        toolbar
          ? cn(
              'inline-flex h-8 items-center rounded-md border bg-surface text-[13px] shadow-xs transition-colors',
              filterActive ? 'border-border-strong' : 'border-dashed border-border-strong',
              unavailable ? 'opacity-60' : 'hover:bg-accent',
            )
          : 'flex flex-col gap-1.5',
        className,
      )}
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget)) close()
      }}
    >
      <span
        id={`${id}-label`}
        className={
          toolbar
            ? 'whitespace-nowrap pl-2.5 pr-1 text-xs text-muted-foreground'
            : 'text-xs font-medium text-muted-foreground'
        }
      >
        {label}
      </span>
      <div className={cn('relative min-w-0', toolbar && 'flex h-full items-center')}>
        <button
          ref={trigger}
          type="button"
          role="combobox"
          aria-labelledby={`${id}-label`}
          aria-controls={`${id}-options`}
          aria-haspopup="listbox"
          aria-expanded={open && !disabled}
          className={
            toolbar
              ? 'flex h-full min-w-0 max-w-[16rem] items-center gap-1.5 rounded-r-md pl-1 pr-2 text-left font-medium text-foreground outline-none focus-visible:ring-[3px] focus-visible:ring-ring/40 disabled:cursor-not-allowed [&_svg]:size-3.5 [&_svg]:shrink-0 [&_svg]:text-subtle-foreground'
              : cn(
                  inputBase,
                  'h-9 cursor-pointer items-center justify-between gap-2 text-left disabled:cursor-not-allowed [&_svg]:size-4 [&_svg]:shrink-0 [&_svg]:text-subtle-foreground',
                  open && 'border-ring ring-[3px] ring-ring/40',
                )
          }
          disabled={unavailable}
          title={selected?.label}
          onClick={() => (open ? close() : show())}
          onKeyDown={(event) => {
            if (escape(event)) return
            if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
              event.preventDefault()
              if (open) focusOption(event.key === 'ArrowDown' ? 0 : filtered.length - 1)
              else show(event.key === 'ArrowDown' ? 'first' : 'last')
            }
          }}
        >
          <span className={cn('truncate', !selected && 'text-subtle-foreground')}>
            {selected?.label ?? placeholder}
          </span>
          <ChevronDown
            aria-hidden="true"
            className={cn('transition-transform duration-200', open && 'rotate-180')}
          />
        </button>
        {open && !disabled && (
          <div
            className={cn(
              'absolute z-30 overflow-hidden rounded-lg border border-border bg-popover text-popover-foreground shadow-lg animate-in fade-in-0 zoom-in-95',
              toolbar ? 'w-max min-w-[13rem] max-w-[22rem]' : 'w-full min-w-[min(16rem,100%)]',
              opensAbove
                ? 'bottom-[calc(100%+6px)] slide-in-from-bottom-1'
                : 'top-[calc(100%+6px)] slide-in-from-top-1',
              toolbar ? '-left-px' : 'left-0',
            )}
          >
            {searchable && (
              <div className="flex items-center gap-2 border-b border-border px-2.5 text-subtle-foreground focus-within:text-foreground">
                <Search aria-hidden="true" className="size-3.5 shrink-0" />
                <input
                  ref={search}
                  type="search"
                  aria-label={`Search ${label.toLowerCase()}`}
                  placeholder="Search…"
                  value={query}
                  className="h-9 min-w-0 flex-1 bg-transparent text-[13px] text-foreground outline-none placeholder:text-subtle-foreground"
                  onChange={(event) => setQuery(event.target.value)}
                  onKeyDown={(event) => {
                    if (escape(event)) return
                    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
                      event.preventDefault()
                      focusOption(event.key === 'ArrowDown' ? 0 : filtered.length - 1)
                    }
                  }}
                />
              </div>
            )}
            <div
              id={`${id}-options`}
              role="listbox"
              aria-labelledby={`${id}-label`}
              className="max-h-[17.5rem] overflow-y-auto overscroll-contain p-1"
            >
              {filtered.map((option, index) => (
                <button
                  key={option.value}
                  ref={(element) => {
                    optionRefs.current[index] = element
                  }}
                  type="button"
                  role="option"
                  tabIndex={-1}
                  aria-selected={option.value === value}
                  data-value={option.value}
                  className="flex w-full items-center justify-between gap-3 rounded-md px-2.5 py-2 text-left text-[13px] text-foreground outline-none transition-colors hover:bg-accent focus-visible:bg-accent aria-selected:bg-accent/70"
                  onKeyDown={(event) => optionKeyDown(event, index)}
                  onClick={() => {
                    onChange(option.value)
                    close(true)
                  }}
                >
                  <span className="min-w-0">
                    <strong className="block break-words font-medium">{option.label}</strong>
                    {option.description && (
                      <small className="mt-0.5 block text-xs text-muted-foreground">
                        {option.description}
                      </small>
                    )}
                  </span>
                  {option.value === value && (
                    <Check aria-hidden="true" className="size-4 shrink-0 text-primary" />
                  )}
                </button>
              ))}
              {!filtered.length && (
                <p className="px-2.5 py-2 text-xs text-muted-foreground">No matching options.</p>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
