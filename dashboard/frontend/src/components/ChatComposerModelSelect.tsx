import { useEffect, useId, useRef, useState, type KeyboardEvent as ReactKeyboardEvent } from 'react'

import type { RouterModelOption } from '../utils/routerModelSelection'
import styles from './ChatComposerModelSelect.module.css'

interface ChatComposerModelSelectProps {
  disabled?: boolean
  models: RouterModelOption[]
  onChange: (model: string) => void
  value: string
}

export default function ChatComposerModelSelect({
  disabled = false,
  models,
  onChange,
  value,
}: ChatComposerModelSelectProps) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const rootRef = useRef<HTMLDivElement>(null)
  const triggerRef = useRef<HTMLButtonElement>(null)
  const optionRefs = useRef<Array<HTMLButtonElement | null>>([])
  const listboxId = useId()
  const selected = models.find((model) => model.id === value)
  const filteredModels = models.filter((model) => {
    const normalized = query.trim().toLowerCase()
    return (
      !normalized ||
      model.id.toLowerCase().includes(normalized) ||
      model.description.toLowerCase().includes(normalized)
    )
  })

  const closeMenu = (restoreFocus = false) => {
    setOpen(false)
    if (restoreFocus) requestAnimationFrame(() => triggerRef.current?.focus())
  }

  const focusOption = (index: number) => {
    if (filteredModels.length === 0) return
    const normalizedIndex = (index + filteredModels.length) % filteredModels.length
    optionRefs.current[normalizedIndex]?.focus()
  }

  const handleOptionKeyDown = (event: ReactKeyboardEvent<HTMLButtonElement>, index: number) => {
    if (event.key === 'ArrowDown') {
      event.preventDefault()
      focusOption(index + 1)
    } else if (event.key === 'ArrowUp') {
      event.preventDefault()
      focusOption(index - 1)
    } else if (event.key === 'Home') {
      event.preventDefault()
      focusOption(0)
    } else if (event.key === 'End') {
      event.preventDefault()
      focusOption(filteredModels.length - 1)
    } else if (event.key === 'Escape') {
      event.preventDefault()
      closeMenu(true)
    }
  }

  useEffect(() => {
    if (!open) return
    const handlePointerDown = (event: MouseEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) closeMenu()
    }
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') closeMenu(true)
    }
    document.addEventListener('mousedown', handlePointerDown)
    document.addEventListener('keydown', handleKeyDown)
    return () => {
      document.removeEventListener('mousedown', handlePointerDown)
      document.removeEventListener('keydown', handleKeyDown)
    }
  }, [open])

  return (
    <div ref={rootRef} className={styles.root}>
      <button
        ref={triggerRef}
        type="button"
        className={styles.trigger}
        aria-expanded={open}
        aria-controls={listboxId}
        aria-haspopup="listbox"
        disabled={disabled || models.length === 0}
        onClick={() => {
          setQuery('')
          if (open) closeMenu()
          else setOpen(true)
        }}
        data-testid="playground-composer-model-select"
        title={selected?.description || value || 'Choose model'}
      >
        <span className={styles.brandMark}>MoM</span>
        <span className={styles.triggerLabel}>{value || 'Choose model'}</span>
        <svg className={styles.chevron} viewBox="0 0 16 16" aria-hidden="true">
          <path d="m4 10 4-4 4 4" />
        </svg>
      </button>

      {open ? (
        <div className={styles.menu}>
          <div className={styles.menuHeader}>
            <div>
              <span>Choose a model</span>
              <small>Mixture-of-Models</small>
            </div>
            <small>{models.length} models</small>
          </div>
          <input
            className={styles.search}
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search models"
            aria-label="Search models"
            role="combobox"
            aria-controls={listboxId}
            aria-expanded="true"
            onKeyDown={(event) => {
              if (event.key === 'ArrowDown') {
                event.preventDefault()
                focusOption(0)
              } else if (event.key === 'ArrowUp') {
                event.preventDefault()
                focusOption(filteredModels.length - 1)
              }
            }}
            autoFocus
          />
          <div id={listboxId} role="listbox" aria-label="Select Mixture-of-Models profile">
            {filteredModels.map((model, index) => {
              const active = model.id === value
              return (
                <button
                  ref={(element) => {
                    optionRefs.current[index] = element
                  }}
                  key={model.id}
                  type="button"
                  role="option"
                  aria-selected={active}
                  className={`${styles.option} ${active ? styles.optionActive : ''}`}
                  onKeyDown={(event) => handleOptionKeyDown(event, index)}
                  onClick={() => {
                    onChange(model.id)
                    closeMenu(true)
                  }}
                >
                  <span className={styles.optionIdentity}>
                    <code>{model.id}</code>
                  </span>
                  {model.description ? (
                    <span className={styles.optionDescription}>{model.description}</span>
                  ) : null}
                  {active ? <span className={styles.activeMark}>✓</span> : null}
                </button>
              )
            })}
            {filteredModels.length === 0 ? (
              <span className={styles.empty}>No models match “{query}”.</span>
            ) : null}
          </div>
        </div>
      ) : null}
    </div>
  )
}
