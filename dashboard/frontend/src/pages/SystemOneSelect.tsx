import { useCallback, useEffect, useId, useRef, useState, type KeyboardEvent } from 'react'
import { createPortal } from 'react-dom'
import ProductIcon from '../components/ProductIcon'
import styles from './SystemOneSelect.module.css'

interface Option {
  value: string
  label: string
  description?: string
}

interface Props {
  label: string
  value: string
  options: Option[]
  onChange: (value: string) => void
  placeholder?: string
  disabled?: boolean
  className?: string
}

/** Select-only combobox: keyboard focus stays on the trigger while its active option changes. */
export default function SystemOneSelect({
  label,
  value,
  options,
  onChange,
  placeholder = 'Choose an option',
  disabled = false,
  className = '',
}: Props) {
  const id = useId()
  const root = useRef<HTMLDivElement>(null)
  const menu = useRef<HTMLDivElement>(null)
  const trigger = useRef<HTMLButtonElement>(null)
  const typeahead = useRef({ text: '', time: 0 })
  const [open, setOpen] = useState(false)
  const [active, setActive] = useState(0)
  const [placement, setPlacement] = useState({
    above: false,
    height: 280,
    top: 0,
    left: 0,
    width: 0,
  })
  const selectedIndex = options.findIndex((option) => option.value === value)
  const selected = options[selectedIndex]
  const expanded = open && !disabled && options.length > 0
  const activeIndex = Math.min(active, options.length - 1)

  const updatePlacement = useCallback(() => {
    const bounds = trigger.current?.getBoundingClientRect()
    if (bounds) {
      const below = window.innerHeight - bounds.bottom - 16
      const above = bounds.top - 16
      const opensAbove = below < Math.min(options.length * 70 + 12, 280) && above > below
      const width = Math.min(bounds.width, window.innerWidth - 32)
      setPlacement({
        above: opensAbove,
        height: Math.max(0, Math.min(280, opensAbove ? above : below)),
        top: opensAbove ? bounds.top - 6 : bounds.bottom + 6,
        left: Math.max(16, Math.min(bounds.left, window.innerWidth - width - 16)),
        width,
      })
    }
  }, [options.length])

  function show(index = Math.max(0, selectedIndex)) {
    updatePlacement()
    setActive(index)
    setOpen(true)
  }

  function select(index: number) {
    const option = options[index]
    if (!option) return
    onChange(option.value)
    setOpen(false)
    trigger.current?.focus()
  }

  function onKeyDown(event: KeyboardEvent<HTMLButtonElement>) {
    if (event.key === 'Escape') {
      if (expanded) {
        event.preventDefault()
        event.stopPropagation()
        setOpen(false)
      }
    } else if (event.key === 'Tab') {
      setOpen(false)
    } else if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      if (expanded) select(activeIndex)
      else show()
    } else if (['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) {
      event.preventDefault()
      let next = Math.max(0, selectedIndex)
      if (event.key === 'Home') next = 0
      else if (event.key === 'End') next = options.length - 1
      else if (expanded)
        next = Math.max(
          0,
          Math.min(options.length - 1, activeIndex + (event.key === 'ArrowDown' ? 1 : -1)),
        )
      if (expanded) setActive(next)
      else show(next)
    } else if (event.key.length === 1 && !event.ctrlKey && !event.metaKey && !event.altKey) {
      event.preventDefault()
      const now = Date.now()
      const query =
        now - typeahead.current.time > 700 ? event.key : typeahead.current.text + event.key
      typeahead.current = { text: query.toLowerCase(), time: now }
      const match = options.findIndex((option) =>
        option.label.toLowerCase().startsWith(query.toLowerCase()),
      )
      if (match !== -1) {
        if (expanded) setActive(match)
        else show(match)
      }
    }
  }

  useEffect(() => {
    if (!expanded) return
    const outside = (event: PointerEvent) => {
      const target = event.target as Node
      if (!root.current?.contains(target) && !menu.current?.contains(target)) setOpen(false)
    }
    const reposition = (event: Event) => {
      if (!menu.current?.contains(event.target as Node)) updatePlacement()
    }
    document.addEventListener('pointerdown', outside)
    window.addEventListener('scroll', reposition, true)
    window.addEventListener('resize', updatePlacement)
    return () => {
      document.removeEventListener('pointerdown', outside)
      window.removeEventListener('scroll', reposition, true)
      window.removeEventListener('resize', updatePlacement)
    }
  }, [expanded, updatePlacement])

  useEffect(() => {
    if (expanded)
      document.getElementById(`${id}-option-${activeIndex}`)?.scrollIntoView({ block: 'nearest' })
  }, [activeIndex, expanded, id])

  return (
    <div ref={root} className={`${styles.field} ${className}`}>
      <span id={`${id}-label`} className={styles.label}>
        {label}
      </span>
      <button
        ref={trigger}
        type="button"
        role="combobox"
        aria-labelledby={`${id}-label`}
        aria-expanded={expanded}
        aria-controls={expanded ? `${id}-options` : undefined}
        aria-activedescendant={expanded ? `${id}-option-${activeIndex}` : undefined}
        aria-haspopup="listbox"
        disabled={disabled || options.length === 0}
        className={styles.trigger}
        onClick={() => (expanded ? setOpen(false) : show())}
        onKeyDown={onKeyDown}
        onBlur={() => setOpen(false)}
      >
        <span className={selected ? undefined : styles.placeholder}>
          {selected?.label ?? placeholder}
        </span>
        <ProductIcon name="chevron-down" />
      </button>
      {expanded &&
        createPortal(
          <div
            ref={menu}
            id={`${id}-options`}
            role="listbox"
            aria-labelledby={`${id}-label`}
            className={`${styles.menu} ${placement.above ? styles.above : ''}`}
            style={{
              maxHeight: placement.height,
              top: placement.top,
              left: placement.left,
              width: placement.width,
            }}
          >
            {options.map((option, index) => (
              <button
                key={option.value}
                type="button"
                tabIndex={-1}
                id={`${id}-option-${index}`}
                role="option"
                aria-selected={value === option.value}
                data-value={option.value}
                data-active={activeIndex === index}
                className={styles.option}
                onMouseDown={(event) => event.preventDefault()}
                onClick={() => select(index)}
                onPointerMove={() => setActive(index)}
              >
                <span>
                  <strong>{option.label}</strong>
                  {option.description && <small>{option.description}</small>}
                </span>
                {value === option.value && <ProductIcon name="check" />}
              </button>
            ))}
          </div>,
          document.body,
        )}
    </div>
  )
}
