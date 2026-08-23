import { useEffect, useId, useRef, useState, type KeyboardEvent as ReactKeyboardEvent } from 'react'

import type { RouterModelOption } from '../utils/routerModelSelection'
import ProductIcon from './ProductIcon'
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
  const filteredModels = models.filter((model) => {
    const normalized = query.trim().toLowerCase()
    return (
      !normalized ||
      model.id.toLowerCase().includes(normalized) ||
      model.recipe?.toLowerCase().includes(normalized) ||
      model.description.toLowerCase().includes(normalized)
    )
  })
  const mixtureModels = filteredModels.filter((model) => model.kind !== 'individual')
  const individualModels = filteredModels.filter((model) => model.kind === 'individual')
  const orderedModels = [...mixtureModels, ...individualModels]

  const closeMenu = (restoreFocus = false) => {
    setOpen(false)
    if (restoreFocus) requestAnimationFrame(() => triggerRef.current?.focus())
  }

  const focusOption = (index: number) => {
    if (orderedModels.length === 0) return
    const normalizedIndex = (index + orderedModels.length) % orderedModels.length
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
      focusOption(orderedModels.length - 1)
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
        aria-label={`Model: ${value || 'Choose model'}`}
        disabled={disabled || models.length === 0}
        onClick={() => {
          setQuery('')
          if (open) closeMenu()
          else setOpen(true)
        }}
        data-testid="playground-composer-model-select"
        title={value || 'Choose model'}
      >
        <span className={styles.triggerLabel}>{value || 'Choose model'}</span>
        <ProductIcon
          className={styles.chevron}
          name={open ? 'chevron-up' : 'chevron-down'}
          aria-hidden="true"
        />
      </button>

      {open ? (
        <div className={styles.menu}>
          <div className={styles.menuHeader}>
            <span>Choose a model</span>
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
                focusOption(orderedModels.length - 1)
              }
            }}
            autoFocus
          />
          <div id={listboxId} role="listbox" aria-label="Select routing model">
            {mixtureModels.length > 0 ? (
              <div className={styles.groupDivider}>
                <span>Mixture-of-Models</span>
              </div>
            ) : null}
            {orderedModels.map((model, index) => {
              const active = model.id === value
              const startsIndividualModels =
                model.kind === 'individual' && index === mixtureModels.length
              return (
                <div key={model.id}>
                  {startsIndividualModels ? (
                    <div className={styles.groupDivider}>
                      <span>Single Model</span>
                    </div>
                  ) : null}
                  <button
                    ref={(element) => {
                      optionRefs.current[index] = element
                    }}
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
                      <span className={styles.optionModelId}>{model.id}</span>
                      {model.recipe ? (
                        <span className={styles.objectiveChip}>{model.recipe}</span>
                      ) : null}
                    </span>
                    {active ? (
                      <ProductIcon className={styles.activeMark} name="check" aria-hidden="true" />
                    ) : null}
                  </button>
                </div>
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
