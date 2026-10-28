import { useEffect, useId, useRef, useState, type KeyboardEvent } from 'react'
import ProductIcon from '../ProductIcon'
import styles from './BenchControls.module.css'

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
      const height = Math.min(options.length * 65 + (searchable ? 50 : 0) + 12, 340)
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
    document.addEventListener('pointerdown', outside)
    return () => document.removeEventListener('pointerdown', outside)
    // Focus only when opening; typing and selection must not reset keyboard focus.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, searchable])

  return (
    <div
      ref={root}
      role="group"
      aria-labelledby={`${id}-label`}
      className={`${styles.selectField} ${className}`}
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget)) close()
      }}
    >
      <span id={`${id}-label`} className={styles.fieldLabel}>
        {label}
      </span>
      <div className={styles.selectControl}>
        <button
          ref={trigger}
          type="button"
          role="combobox"
          aria-labelledby={`${id}-label`}
          aria-controls={`${id}-options`}
          aria-haspopup="listbox"
          aria-expanded={open && !disabled}
          className={styles.selectTrigger}
          disabled={disabled || options.length === 0}
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
          <span className={!selected ? styles.placeholder : undefined}>
            {selected?.label ?? placeholder}
          </span>
          <ProductIcon name="chevron-down" />
        </button>
        {open && !disabled && (
          <div className={`${styles.selectMenu} ${opensAbove ? styles.menuAbove : ''}`}>
            {searchable && (
              <div className={styles.selectSearch}>
                <ProductIcon name="search" />
                <input
                  ref={search}
                  type="search"
                  aria-label={`Search ${label.toLowerCase()}`}
                  placeholder="Search…"
                  value={query}
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
              className={styles.options}
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
                  className={styles.selectOption}
                  onKeyDown={(event) => optionKeyDown(event, index)}
                  onClick={() => {
                    onChange(option.value)
                    close(true)
                  }}
                >
                  <span>
                    <strong>{option.label}</strong>
                    {option.description && <small>{option.description}</small>}
                  </span>
                  {option.value === value && <ProductIcon name="check" />}
                </button>
              ))}
              {!filtered.length && <p className={styles.noOptions}>No matching options.</p>}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
