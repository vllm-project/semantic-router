import { useEffect, useId, useRef, useState } from 'react'

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
  const menuId = useId()
  const selected = models.find((model) => model.id === value)
  const filteredModels = models.filter((model) => {
    const normalized = query.trim().toLowerCase()
    return (
      !normalized ||
      model.id.toLowerCase().includes(normalized) ||
      model.description.toLowerCase().includes(normalized)
    )
  })

  useEffect(() => {
    if (!open) return
    const handlePointerDown = (event: MouseEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false)
    }
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setOpen(false)
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
        type="button"
        className={styles.trigger}
        aria-expanded={open}
        aria-controls={menuId}
        aria-haspopup="listbox"
        disabled={disabled || models.length === 0}
        onClick={() => {
          setQuery('')
          setOpen((current) => !current)
        }}
        data-testid="playground-composer-model-select"
        title={selected?.description || value || 'Choose model'}
      >
        <span className={styles.brandMark}>AMD</span>
        <span className={styles.triggerLabel}>{value || 'Choose model'}</span>
        <span className={styles.chevron} aria-hidden="true">
          ⌄
        </span>
      </button>

      {open ? (
        <div
          id={menuId}
          className={styles.menu}
          role="listbox"
          aria-label="Select Mixture-of-Models profile"
        >
          <div className={styles.menuHeader}>
            <div>
              <span>Choose a model</span>
              <small>AMD Mixture-of-Models</small>
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
            autoFocus
          />
          {filteredModels.map((model) => {
            const active = model.id === value
            return (
              <button
                key={model.id}
                type="button"
                role="option"
                aria-selected={active}
                className={`${styles.option} ${active ? styles.optionActive : ''}`}
                onClick={() => {
                  onChange(model.id)
                  setOpen(false)
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
      ) : null}
    </div>
  )
}
